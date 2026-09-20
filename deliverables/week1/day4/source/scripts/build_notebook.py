#!/usr/bin/env python3
"""Build the reproducible Day 4 tokenizer experiment notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_cases() -> list[dict[str, str]]:
    """Return the fixed cases in their declared order."""
    return [
        {
            "id": "pure_chinese",
            "category": "纯中文",
            "text": "今天天气很好，我们正在研究大语言模型的分词机制。",
        },
        {
            "id": "pure_english",
            "category": "纯英文",
            "text": "Tokenization maps raw text into discrete model input units.",
        },
        {
            "id": "mixed_language_version",
            "category": "中英文、数字与版本号混合",
            "text": "Qwen2.5-7B-Instruct 在 Python 3.10 与 CUDA 12.1 环境中运行。",
        },
        {
            "id": "emoji",
            "category": "Emoji 与组合符号",
            "text": "模型表现不错🙂🚀，家庭 emoji：👨‍👩‍👧‍👦，旗帜：🇨🇳🇬🇧。",
        },
        {
            "id": "rare_cjk",
            "category": "罕见 CJK 字符",
            "text": "常见字：龙；扩展汉字：𠮷、𩸽、𰻞。",
        },
        {
            "id": "unicode_composition",
            "category": "Unicode 预组合与 combining",
            "text": "预组合 café；组合 cafe\u0301；Å 与 A\u030a。",
        },
        {
            "id": "fullwidth_halfwidth",
            "category": "全角与半角",
            "text": "半角 ABC123!；全角 ＡＢＣ１２３！；日文 ｶﾀｶﾅ／カタカナ。",
        },
        {
            "id": "whitespace_controls",
            "category": "连续空格、Tab 与换行",
            "text": "第一列    第二列\t第三列\n下一行\n\n最后一行",
        },
        {
            "id": "python_code",
            "category": "Python 代码",
            "text": "def fib(n: int) -> int:\n    return n if n < 2 else fib(n-1) + fib(n-2)",
        },
        {
            "id": "json_url_escape",
            "category": "JSON、URL 与转义符",
            "text": r'{"url":"https://example.com/a?q=大模型&x=1","path":"C:\\tmp\\a.txt","ok":true}',
        },
        {
            "id": "latex_math",
            "category": "LaTeX 数学公式",
            "text": r"欧拉公式 $e^{i\pi}+1=0$，积分 $\int_0^1 x^2\,dx=\frac{1}{3}$。",
        },
        {
            "id": "literal_special_tokens",
            "category": "字面量特殊令牌",
            "text": "用户输入中原样出现 <|im_start|>system 和 <|im_end|> 以及 <|endoftext|>。",
        },
        {
            "id": "repeated_hanzi",
            "category": "单个汉字大量重复",
            "text": "哈" * 128,
        },
        {
            "id": "zero_width",
            "category": "零宽字符",
            "text": "可见文字A\u200bB\u200cC\u200dD\ufeff结束",
        },
        {
            "id": "long_chinese",
            "category": "长中文截断样本",
            "text": (
                "大语言模型把连续文本转换为离散令牌，随后通过嵌入层和多层注意力网络处理。"
                "分词方式会影响序列长度、计算成本、跨语言表达以及罕见字符的处理。"
                "本实验固定最大长度为六十四个令牌，并同时观察右截断与左截断保留的内容。"
            )
            * 12,
        },
    ]


def build_corpus_lines() -> list[str]:
    base = [
        "中文分词需要处理词语边界、标点符号、数字以及罕见汉字。",
        "大语言模型通过 tokenizer 将文本映射到 token identifiers。",
        "北京上海深圳成都杭州西安武汉南京广州苏州都是城市名称。",
        "机器学习深度学习自然语言处理计算机视觉强化学习数据工程。",
        "春夏秋冬东西南北山川湖海日月星辰天地玄黄宇宙洪荒。",
        "The same corpus is used to train both tokenizer algorithms.",
        "Byte level processing can represent arbitrary UTF-8 input without unknown bytes.",
        "SentencePiece Unigram selects pieces with a probabilistic language model objective.",
        "Qwen2.5-7B-Instruct supports Chinese English code math and structured text.",
        "版本 v2.5.1，日期 2026-07-18，温度 23.5°C，网址 https://example.com。",
        "你好 world，模型 tokenizer 测试 mixed-language input 12345。",
        "def add(a, b): return a + b  # Python source code",
        "数学公式 e^(i*pi)+1=0，积分从零到一，结果等于三分之一。",
        "标点测试：，。！？；：“”‘’（）【】《》—…",
        "Emoji samples 🙂 🚀 🌏 and rare characters 𠮷 𩸽 are included.",
    ]
    lines: list[str] = []
    for index in range(40):
        for sentence in base:
            lines.append(f"样本{index:03d} {sentence} 序号{index} variant-{index % 17}")
    hanzi = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔龙师火帝鸟官人皇始制文字乃服衣裳推位让国有虞陶唐吊民伐罪周发殷汤坐朝问道垂拱平章爱育黎首臣伏戎羌遐迩壹体率宾归王鸣凤在竹白驹食场化被草木赖及万方"
    characters = list(dict.fromkeys(hanzi))
    for index, first in enumerate(characters):
        for step in range(1, 13):
            second = characters[(index + step) % len(characters)]
            third = characters[(index + 2 * step) % len(characters)]
            fourth = characters[(index + 3 * step) % len(characters)]
            term = first + second + third + fourth
            reverse = fourth + third + second + first
            lines.append(f"组合术语 {term} {term} {reverse} 研究{term}方法与{reverse}系统。")
    return lines


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cases_json = json.dumps(build_cases(), ensure_ascii=False, indent=2)
    cells = [
        _markdown(
            """# Qwen2.5 Tokenizer 与分词实验

本 Notebook 完成 Week 1 Day 4 的全部要求：15 个固定极端用例、Qwen 特殊令牌、
`add_generation_prompt`、`max_length=64` 长文本截断，以及同语料、同目标词表大小下的
ByteLevel BPE 与 SentencePiece Unigram 受控比较。所有结果均由代码直接生成并保存。
"""
        ),
        _markdown(
            """## 1. 环境与可复现设置

Tokenizer 编码和小型分词器训练均为 CPU 任务，不加载 Qwen 模型权重。
Notebook 使用本地已有的 Qwen2.5-7B-Instruct Tokenizer 文件，并记录关键软件版本。
"""
        ),
        _code(
            """import json
import os
import platform
import sys
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import tokenizers as tokenizers_package
import transformers
from IPython.display import display
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

DAY4_ROOT = Path(os.environ.get("DAY4_ROOT", Path.cwd())).resolve()
MODEL_DIR = Path(os.environ.get(
    "QWEN_MODEL_DIR",
    "/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct",
)).resolve()
OUTPUT_DIR = Path(os.environ.get("DAY4_OUTPUT_DIR", DAY4_ROOT / "source" / "results")).resolve()
CORPUS_PATH = Path(os.environ.get(
    "DAY4_CORPUS_PATH", DAY4_ROOT / "source" / "data" / "tokenizer_comparison_corpus.txt"
)).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

versions = {
    "Python": platform.python_version(),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers_package.__version__,
    "sentencepiece": spm.__version__,
    "pandas": pd.__version__,
    "platform": platform.platform(),
    "model_dir": str(MODEL_DIR),
    "corpus_path": str(CORPUS_PATH),
    "output_dir": str(OUTPUT_DIR),
}
(OUTPUT_DIR / "day4_environment.txt").write_text(
    "\\n".join(f"{key}={value}" for key, value in versions.items()) + "\\n",
    encoding="utf-8",
)
display(pd.DataFrame([versions]).T.rename(columns={0: "value"}))
assert MODEL_DIR.is_dir(), f"Tokenizer directory not found: {MODEL_DIR}"
assert CORPUS_PATH.is_file(), f"Comparison corpus not found: {CORPUS_PATH}"
"""
        ),
        _code(
            """tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
)
print("tokenizer_class:", tokenizer.__class__.__name__)
print("len(tokenizer):", len(tokenizer))
print("vocab_size property:", tokenizer.vocab_size)
print("special_tokens_map:", tokenizer.special_tokens_map)
"""
        ),
        _markdown(
            """## 2. 15 个固定极端用例

所有样本在执行前固定，不根据输出更换。统一观测字符数、UTF-8 字节数、token 数、
token IDs、token 文本、decode 结果和 round-trip 是否一致。
"""
        ),
        _code(f"CASES = {cases_json}\n\nassert len(CASES) == 15\n[case['id'] for case in CASES]\n"),
        _code(
            """def observe_text(case):
    text = case["text"]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "case_id": case["id"],
        "category": case["category"],
        "text": text,
        "character_count": len(text),
        "utf8_byte_count": len(text.encode("utf-8")),
        "token_count": len(token_ids),
        "token_ids": json.dumps(token_ids, ensure_ascii=False),
        "token_strings": json.dumps(token_strings, ensure_ascii=False),
        "decoded": decoded,
        "roundtrip_equal": decoded == text,
    }

extreme_rows = [observe_text(case) for case in CASES]
extreme_df = pd.DataFrame(extreme_rows)
extreme_path = OUTPUT_DIR / "tokenizer_extreme_cases.csv"
extreme_df.to_csv(extreme_path, index=False, encoding="utf-8")
display(extreme_df[[
    "case_id", "category", "character_count", "utf8_byte_count",
    "token_count", "roundtrip_equal"
]])
print("saved:", extreme_path)
"""
        ),
        _code(
            """for row in extreme_rows:
    print("=" * 90)
    print(f"{row['case_id']} | {row['category']}")
    print("TEXT:", repr(row["text"]))
    print("TOKEN_IDS:", row["token_ids"])
    print("TOKEN_STRINGS:", row["token_strings"])
    print("DECODED:", repr(row["decoded"]))
    print("ROUNDTRIP_EQUAL:", row["roundtrip_equal"])
"""
        ),
        _markdown(
            """## 3. 特殊令牌实验

这里同时观察 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 的词表 ID、默认特殊令牌
处理和 `split_special_tokens=True` 的字面量拆分。两种路径回答的问题不同：前者展示
模型协议令牌，后者展示相同字符串作为普通文本时会被如何切分。
"""
        ),
        _code(
            """special_token_names = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
literal_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
    split_special_tokens=True,
)

special_results = {"tokens": {}}
for token_text in special_token_names:
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    protocol_ids = tokenizer.encode(token_text, add_special_tokens=False)
    literal_ids = literal_tokenizer.encode(token_text, add_special_tokens=False)
    special_results["tokens"][token_text] = {
        "vocabulary_id": token_id,
        "protocol_ids": protocol_ids,
        "protocol_token_strings": tokenizer.convert_ids_to_tokens(protocol_ids),
        "protocol_decode_keep_special": tokenizer.decode(
            protocol_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "protocol_decode_skip_special": tokenizer.decode(
            protocol_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ),
        "literal_ids_split_special_tokens_true": literal_ids,
        "literal_token_strings": literal_tokenizer.convert_ids_to_tokens(literal_ids),
        "literal_decode": literal_tokenizer.decode(
            literal_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
    }

display(pd.DataFrame([
    {
        "token": token,
        "vocabulary_id": values["vocabulary_id"],
        "protocol_ids": values["protocol_ids"],
        "literal_ids": values["literal_ids_split_special_tokens_true"],
        "decode_keep_special": values["protocol_decode_keep_special"],
        "decode_skip_special": values["protocol_decode_skip_special"],
    }
    for token, values in special_results["tokens"].items()
]))
"""
        ),
        _markdown(
            """### Chat Template：`add_generation_prompt=True/False`

对完全相同的 messages 只切换一个参数。`True` 在尾部加入 assistant 起始标记，供模型
继续生成；`False` 只序列化已有消息，适合保存完整对话或观察模板本身。
"""
        ),
        _code(
            """messages = [
    {"role": "system", "content": "你是一个严谨的实验助手。"},
    {"role": "user", "content": "请用一句话解释分词。"},
]
chat_template_records = {}
for flag in (False, True):
    text_value = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=flag
    )
    ids_value = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=flag
    )
    chat_template_records[f"add_generation_prompt_{str(flag).lower()}"] = {
        "text": text_value,
        "token_ids": ids_value,
        "token_count": len(ids_value),
        "last_12_token_ids": ids_value[-12:],
        "last_12_token_strings": tokenizer.convert_ids_to_tokens(ids_value[-12:]),
    }

special_results["chat_template"] = chat_template_records
false_record = chat_template_records["add_generation_prompt_false"]
true_record = chat_template_records["add_generation_prompt_true"]
special_results["chat_template_difference"] = {
    "added_token_count": true_record["token_count"] - false_record["token_count"],
    "added_suffix_ids": true_record["token_ids"][false_record["token_count"]:],
    "added_suffix_text": true_record["text"][len(false_record["text"]):],
}

special_path = OUTPUT_DIR / "special_token_results.json"
special_path.write_text(
    json.dumps(special_results, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("False template:\\n", false_record["text"])
print("True template:\\n", true_record["text"])
print("Difference:", special_results["chat_template_difference"])
print("saved:", special_path)
"""
        ),
        _markdown(
            """## 4. 长文本截断（`max_length=64`）

分别验证右截断与左截断。记录截断前后 token 数、保留文本、末尾 token，并恢复
Tokenizer 原始的 `truncation_side`，避免污染后续实验。
"""
        ),
        _code(
            """long_text = next(case["text"] for case in CASES if case["id"] == "long_chinese")
full_ids = tokenizer.encode(long_text, add_special_tokens=False)
original_side = tokenizer.truncation_side
truncation_results = {
    "max_length": 64,
    "before_characters": len(long_text),
    "before_utf8_bytes": len(long_text.encode("utf-8")),
    "before_tokens": len(full_ids),
    "original_truncation_side": original_side,
    "runs": {},
}
for side in ("right", "left"):
    tokenizer.truncation_side = side
    ids = tokenizer.encode(
        long_text,
        add_special_tokens=False,
        truncation=True,
        max_length=64,
    )
    retained = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    truncation_results["runs"][side] = {
        "after_tokens": len(ids),
        "retained_text": retained,
        "retained_characters": len(retained),
        "token_ids": ids,
        "last_token_id": ids[-1],
        "last_token_string": tokenizer.convert_ids_to_tokens(ids[-1]),
    }
tokenizer.truncation_side = original_side

truncation_path = OUTPUT_DIR / "truncation_results.json"
truncation_path.write_text(
    json.dumps(truncation_results, ensure_ascii=False, indent=2), encoding="utf-8"
)
display(pd.DataFrame([
    {
        "side": side,
        "before_tokens": truncation_results["before_tokens"],
        "after_tokens": values["after_tokens"],
        "retained_characters": values["retained_characters"],
        "last_token_id": values["last_token_id"],
        "last_token_string": values["last_token_string"],
        "retained_text": values["retained_text"],
    }
    for side, values in truncation_results["runs"].items()
]))
print("saved:", truncation_path)
"""
        ),
        _markdown(
            """## 5. ByteLevel BPE 与 SentencePiece Unigram 受控对比

SentencePiece 是分词训练框架，不等同于单一算法；本实验明确选择其 Unigram 模型，
并与 ByteLevel BPE 比较。二者读取同一 UTF-8 语料，目标 `vocab_size=800`，都启用
字节级回退能力，减少罕见字符变成 `<unk>` 的可能。
"""
        ),
        _code(
            """TARGET_VOCAB_SIZE = 800
TRAIN_DIR = OUTPUT_DIR / "trained_tokenizers"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

bpe_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
bpe_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
bpe_tokenizer.decoder = ByteLevelDecoder()
bpe_trainer = BpeTrainer(
    vocab_size=TARGET_VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<unk>"],
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=False,
)
bpe_tokenizer.train([str(CORPUS_PATH)], bpe_trainer)
bpe_tokenizer.save(str(TRAIN_DIR / "bytelevel_bpe.json"))

sp_prefix = TRAIN_DIR / "sentencepiece_unigram"
spm.SentencePieceTrainer.train(
    input=str(CORPUS_PATH),
    model_prefix=str(sp_prefix),
    model_type="unigram",
    vocab_size=TARGET_VOCAB_SIZE,
    character_coverage=1.0,
    byte_fallback=True,
    hard_vocab_limit=True,
    unk_id=0,
    bos_id=-1,
    eos_id=-1,
    pad_id=-1,
    shuffle_input_sentence=False,
)
sp_tokenizer = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")

print("BPE actual vocab:", bpe_tokenizer.get_vocab_size())
print("SentencePiece actual vocab:", sp_tokenizer.vocab_size())
"""
        ),
        _code(
            """comparison_samples = [
    {"sample_id": "zh_common", "language": "Chinese", "text": "自然语言处理需要稳定而可复现的分词实验。"},
    {"sample_id": "zh_rare", "language": "Chinese", "text": "罕见字符𠮷和𩸽也应该能够往返解码。"},
    {"sample_id": "en", "language": "English", "text": "Tokenizers trade vocabulary size against sequence length."},
    {"sample_id": "mixed", "language": "Mixed", "text": "Qwen2.5 使用 tokenizer 处理 version-7B 与中文。"},
]

comparison_rows = []
for sample in comparison_samples:
    text_value = sample["text"]
    bpe_encoding = bpe_tokenizer.encode(text_value)
    sp_ids = sp_tokenizer.encode(text_value, out_type=int)
    sp_pieces = sp_tokenizer.encode(text_value, out_type=str)
    denominators = {
        "character_count": len(text_value),
        "word_count": len(text_value.split()),
    }
    comparison_rows.extend([
        {
            **sample,
            **denominators,
            "tokenizer": "ByteLevel BPE",
            "target_vocab_size": TARGET_VOCAB_SIZE,
            "actual_vocab_size": bpe_tokenizer.get_vocab_size(),
            "token_count": len(bpe_encoding.ids),
            "token_ids": json.dumps(bpe_encoding.ids),
            "pieces": json.dumps(bpe_encoding.tokens, ensure_ascii=False),
            "unknown_count": sum(
                token_id == bpe_tokenizer.token_to_id("<unk>")
                for token_id in bpe_encoding.ids
            ),
            "decoded": bpe_tokenizer.decode(bpe_encoding.ids),
            "roundtrip_equal": bpe_tokenizer.decode(bpe_encoding.ids) == text_value,
        },
        {
            **sample,
            **denominators,
            "tokenizer": "SentencePiece Unigram",
            "target_vocab_size": TARGET_VOCAB_SIZE,
            "actual_vocab_size": sp_tokenizer.vocab_size(),
            "token_count": len(sp_ids),
            "token_ids": json.dumps(sp_ids),
            "pieces": json.dumps(sp_pieces, ensure_ascii=False),
            "unknown_count": sum(token_id == sp_tokenizer.unk_id() for token_id in sp_ids),
            "decoded": sp_tokenizer.decode(sp_ids),
            "roundtrip_equal": sp_tokenizer.decode(sp_ids) == text_value,
        },
    ])

comparison_df = pd.DataFrame(comparison_rows)
comparison_df["tokens_per_character"] = (
    comparison_df["token_count"] / comparison_df["character_count"]
)
comparison_df["tokens_per_whitespace_word"] = (
    comparison_df["token_count"] / comparison_df["word_count"].clip(lower=1)
)
comparison_path = OUTPUT_DIR / "tokenizer_comparison.csv"
comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
display(comparison_df[[
    "sample_id", "language", "tokenizer", "target_vocab_size",
    "actual_vocab_size", "token_count", "tokens_per_character",
    "tokens_per_whitespace_word", "unknown_count", "roundtrip_equal"
]])
print("saved:", comparison_path)
"""
        ),
        _code(
            """for sample_id in comparison_df["sample_id"].unique():
    print("=" * 90)
    print(sample_id)
    for row in comparison_df[comparison_df["sample_id"] == sample_id].to_dict("records"):
        print(row["tokenizer"], "tokens=", row["token_count"])
        print("pieces:", row["pieces"])
        print("decoded:", repr(row["decoded"]), "roundtrip=", row["roundtrip_equal"])
"""
        ),
        _markdown(
            """## 6. 结果解释与局限

- BPE 按频率反复合并相邻符号；本实验的 ByteLevel pre-tokenizer 先把 UTF-8 输入映射到字节可表示空间。
- SentencePiece 是框架，本实验采用 Unigram：从候选子词集合出发，依据概率模型和似然目标删减候选。
- 中文没有天然空格边界，token 数会受训练语料覆盖、normalizer、pre-tokenizer、byte fallback 和词表大小共同影响。
- `vocab_size=800` 是双方相同的训练目标；SentencePiece 使用 `hard_vocab_limit=True`，并由最终验证器确认两者实际词表也均为 800。
- 本实验只证明这份固定小语料和这些配置下的差异，不能推出某类算法在所有中文任务中普遍更优。
"""
        ),
        _markdown("## 7. 执行完成检查\n"),
        _code(
            """required_outputs = [
    OUTPUT_DIR / "tokenizer_extreme_cases.csv",
    OUTPUT_DIR / "special_token_results.json",
    OUTPUT_DIR / "truncation_results.json",
    OUTPUT_DIR / "tokenizer_comparison.csv",
    OUTPUT_DIR / "day4_environment.txt",
]
assert len(extreme_df) == 15
assert set(extreme_df["case_id"]) == {case["id"] for case in CASES}
assert all(path.is_file() and path.stat().st_size > 0 for path in required_outputs)
assert truncation_results["runs"]["right"]["after_tokens"] == 64
assert truncation_results["runs"]["left"]["after_tokens"] == 64
assert set(special_results["tokens"]) == {
    "<|endoftext|>", "<|im_start|>", "<|im_end|>"
}
assert set(comparison_df["tokenizer"]) == {
    "ByteLevel BPE", "SentencePiece Unigram"
}
completion_lines = [
    "DAY4_NOTEBOOK_EXECUTION=PASS",
    "fixed_cases=15",
    "special_tokens=3",
    "truncation_max_length=64",
    "comparison_target_vocab_size=800",
] + [f"output={path.name}" for path in required_outputs]
(OUTPUT_DIR / "day4_notebook_completion.txt").write_text(
    "\\n".join(completion_lines) + "\\n", encoding="utf-8"
)
print("\\n".join(completion_lines))
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"day4-cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (llm_exp)",
                "language": "python",
                "name": "llm_exp",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def write_corpus(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(build_corpus_lines()) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    write_notebook(args.notebook)
    write_corpus(args.corpus)
    print(f"notebook={args.notebook}")
    print(f"corpus={args.corpus}")
    print("fixed_cases=15")


if __name__ == "__main__":
    main()
