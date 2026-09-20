"""Local Hugging Face chat adapter for LangGraph tool calling."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _decode_tool_call(
    value: object, index: int, call_id_prefix: str
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        return None
    name = value["name"]
    arguments = value["arguments"]
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return {
        "name": name,
        "args": arguments,
        "id": f"{call_id_prefix}-{index}",
        "type": "tool_call",
    }


def parse_model_output(
    raw_text: str, call_id_prefix: str = "model-tool-call"
) -> AIMessage:
    """Parse only model-emitted tool-call syntax; never infer calls from prose."""
    candidates = _TOOL_CALL_BLOCK.findall(raw_text)
    if not candidates:
        candidates = [raw_text.strip()]

    tool_calls: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        tool_call = _decode_tool_call(decoded, len(tool_calls), call_id_prefix)
        if tool_call is not None:
            tool_calls.append(tool_call)

    content = "" if tool_calls else raw_text
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        response_metadata={"raw_generated_text": raw_text},
    )


def _message_for_template(message: BaseMessage) -> dict[str, Any]:
    if message.type == "human":
        return {"role": "user", "content": message.content}
    if message.type == "system":
        return {"role": "system", "content": message.content}
    if message.type == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": getattr(message, "tool_call_id", None),
        }
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    tool_calls = getattr(message, "tool_calls", [])
    if tool_calls:
        payload["tool_calls"] = [
            {
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call["args"], ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
            for call in tool_calls
        ]
    return payload


class HuggingFaceLocalChatModel(BaseChatModel):
    """A deterministic local causal-LM adapter with deferred heavy imports."""

    model_path: str
    adapter_path: str | None = None
    max_new_tokens: int = 256
    do_sample: bool = False
    repetition_penalty: float = 1.0
    seed: int = 42

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _bound_tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _generation_index: int = PrivateAttr(default=0)

    def __init__(self, *, tokenizer: Any, model: Any, **data: Any) -> None:
        super().__init__(**data)
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def from_loaded(
        cls, *, tokenizer: Any, model: Any, model_path: str, **kwargs: Any
    ) -> "HuggingFaceLocalChatModel":
        return cls(tokenizer=tokenizer, model=model, model_path=model_path, **kwargs)

    @classmethod
    def from_local_model(
        cls, model_path: str, **kwargs: Any
    ) -> "HuggingFaceLocalChatModel":
        """Load a local base and, when supplied, merge the verified PEFT adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        adapter_path = kwargs.pop("adapter_path", None)
        set_seed(int(kwargs.get("seed", 42)))
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, adapter_path, local_files_only=True
            ).merge_and_unload()
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            model_path=model_path,
            adapter_path=adapter_path,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface-local-tool-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_path": self.model_path, "adapter_path": self.adapter_path}

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> "HuggingFaceLocalChatModel":
        self._bound_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **_: Any,
    ) -> ChatResult:
        del stop, run_manager
        template_messages = [_message_for_template(message) for message in messages]
        input_ids = self._tokenizer.apply_chat_template(
            template_messages,
            tools=self._bound_tools,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        device = getattr(self._model, "device", None)
        if device is not None and hasattr(input_ids, "to"):
            input_ids = input_ids.to(device)
        generated = self._model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        output_tokens = generated[0][input_ids.shape[-1] :]
        raw_text = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
        prefix = f"model-turn-{self._generation_index}-tool-call"
        self._generation_index += 1
        message = parse_model_output(raw_text, call_id_prefix=prefix)
        return ChatResult(generations=[ChatGeneration(message=message)])
