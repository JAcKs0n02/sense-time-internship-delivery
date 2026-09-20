"""Explicit text/vision tabs. Models run separately; this UI never starts one."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid
from contextlib import aclosing

WEEK7 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK7 / 'source'))
from week7_deployment.image_input import vision_payload
from week7_deployment.service_contract import validate_base_url

_spec = importlib.util.spec_from_file_location('week7_day37_text', WEEK7 / 'day37/app.py')
text_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(text_app)


class IncompleteResponse(RuntimeError):
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason
        super().__init__('empty, interrupted or truncated response')


async def stream_vision(prompt, image, temperature=0, top_p=1, max_tokens=256, *, client=None, stop_event=None):
    if type(max_tokens) is float and max_tokens.is_integer():
        max_tokens = int(max_tokens)
    payload = vision_payload(prompt, image, temperature, top_p, max_tokens)
    owned = client is None
    audit = None
    started = time.monotonic()
    status = 'interrupted'
    audit_dir = os.environ.get('WEEK7_VISION_UI_AUDIT_DIR')
    if audit_dir:
        directory = Path(audit_dir); directory.mkdir(parents=True, exist_ok=True)
        audit = (directory / (uuid.uuid4().hex + '.jsonl')).open('x', encoding='utf-8')
    def record(event, **values):
        if audit:
            audit.write(json.dumps({'event': event, 'elapsed_seconds': time.monotonic()-started, **values}, ensure_ascii=False)+'\n')
            audit.flush()
    try:
        record('request', model=payload['model'], prompt=prompt,
               source_sha256=hashlib.sha256(image).hexdigest(),
               temperature=temperature, top_p=top_p, max_tokens=max_tokens,
               image_url_sha256=hashlib.sha256(payload['messages'][0]['content'][1]['image_url']['url'].encode()).hexdigest())
        if owned:
            from openai import AsyncOpenAI
            try:
                url = validate_base_url(os.environ.get('WEEK7_VISION_BASE_URL', 'http://127.0.0.1:8001/v1'))
            except ValueError as exc:
                raise RuntimeError('invalid visual service configuration') from exc
            client = AsyncOpenAI(base_url=url, api_key='local-only', timeout=90, max_retries=0)
        text = ''; finish = None
        async with await text_app.until_stopped(client.chat.completions.create(**payload, stream=True), stop_event) as response:
            iterator = response.__aiter__()
            while True:
                try: chunk = await text_app.until_stopped(anext(iterator), stop_event)
                except StopAsyncIteration: break
                if chunk.model != 'week5-qwen2-vl-base':
                    raise RuntimeError('unexpected response model')
                if not chunk.choices: continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    text += choice.delta.content
                    record('delta', content=choice.delta.content)
                    yield text
                if choice.finish_reason: finish = choice.finish_reason
        if not text.strip() or finish != 'stop':
            raise IncompleteResponse(finish)
        status = 'complete'; record('complete', finish_reason=finish, text=text)
    except text_app.UserStopped:
        status = 'stopped'; record('stop_received'); raise
    except Exception as exc:
        status = 'error'
        record('error', error_type=type(exc).__name__,
               finish_reason=getattr(exc, 'finish_reason', None))
        raise
    finally:
        try:
            if owned and client is not None: await client.close()
        finally:
            record('closed', status=status)
            if audit: audit.close()


def vision_error(exc):
    if isinstance(exc, IncompleteResponse) and exc.finish_reason == 'length':
        return '视觉回答达到输出上限，内容未完整结束；请缩小问题范围或分步提问后重试。'
    if isinstance(exc, ValueError):
        return '图片或参数不合法：仅静态JPEG/PNG/WEBP、10MiB/2000万像素以内，问题1–1000字符；请检查后重试。'
    if getattr(exc, 'status_code', None) == 400:
        return '视觉请求超出上下文或图片限制，请缩短问题或降低输出token后重试。'
    return '视觉服务未就绪或回答未完整结束。请先停止文字后端并启动Week5视觉服务；不自动回退到文字模型。'


def build_app():
    import gradio as gr
    sessions = {}
    class ClearCancelsChatInterface(gr.ChatInterface):
        def _setup_stop_events(self, event_triggers, events_to_cancel):
            super()._setup_stop_events(event_triggers, events_to_cancel)
            self.chatbot.clear(None, cancels=events_to_cancel, queue=False, api_name=False)

    async def chat(message, history, image, temperature, top_p, max_tokens, request: gr.Request):
        key = request.session_hash
        if not key: raise gr.Error('缺少会话标识，请刷新后重试。')
        event = asyncio.Event(); sessions[key] = event
        last = None
        try:
            # Visual display history is intentionally not sent: only the current image.
            async with aclosing(stream_vision(message, image, temperature, top_p, max_tokens, stop_event=event)) as stream:
                async for value in stream:
                    last = value
                    yield value
        except text_app.UserStopped: return
        except Exception as exc:
            if last is not None:
                # Gradio 5.31.0 leaks its iterator-keyed diff baseline when a
                # generator raises after yielding. Exhaust normally, but keep
                # the partial answer visibly marked as incomplete.
                yield f'{last}\n\n⚠️ 回答未完整结束。{vision_error(exc)}'
                return
            raise gr.Error(vision_error(exc)) from None
        finally:
            if sessions.get(key) is event: sessions.pop(key, None)

    async def stop(request: gr.Request):
        event = sessions.get(request.session_hash)
        if event: event.set()

    with gr.Blocks(title='Week7 · 文字与图片', analytics_enabled=False,
                   theme=gr.themes.Soft(font=['sans-serif']), delete_cache=(3600, 3600)) as demo:
        gr.Markdown('# Week7 · 文字与图片\n单卡按需启动后端；切换标签不会自动启动模型。请勿上传敏感资料。')
        with gr.Tab('文字对话'):
            text_app.build_app(introduction='## Week4 DPO · AWQ 4-bit\n当前路由：week4-dpo-quantized · 127.0.0.1:8000\n\n文字历史只在本标签中传递。工程演示，非质量恢复认证；图片请使用“图片理解”标签。')
        with gr.Tab('图片理解'):
            gr.Markdown('## Week5 · Qwen2-VL-7B-Instruct 基座\n'
                        '当前路由：week5-qwen2-vl-base · 127.0.0.1:8001\n\n'
                        '**非质量达标认证。表格、公式、界面细节可能识别错误，请核对原图。**\n\n'
                        '每次仅处理当前图片和当前问题；下方历史仅供查看，不传回模型。换图请先清空对话。'
                        '图片会去除元数据、旋转校正、透明区域铺白底并缩小到约40万像素，细小文字可能丢失。')
            image = gr.File(label='上传一张图片（10MiB以内）', file_count='single',
                            file_types=['.png', '.jpg', '.jpeg', '.webp'], type='binary')
            with gr.Row():
                temperature = gr.Slider(0, 2, value=0, step=.1, label='视觉 Temperature')
                top_p = gr.Slider(.05, 1, value=1, step=.05, label='视觉 Top-p')
                max_tokens = gr.Slider(32, 512, value=256, step=32, label='视觉最大输出 token')
            interface = ClearCancelsChatInterface(chat, type='messages',
                chatbot=gr.Chatbot(type='messages', label='图片回答记录（不作为模型上下文）', height=380, show_copy_button=True),
                textbox=gr.Textbox(label='图片问题', placeholder='例如：逐列读出表格标题；请核对回答', submit_btn='发送', stop_btn='停止'),
                additional_inputs=[image, temperature, top_p, max_tokens],
                submit_btn='发送', stop_btn='停止', save_history=False, analytics_enabled=False,
                flagging_mode='never', concurrency_limit=1, api_name='vision_chat')
            interface.textbox.stop(stop, queue=False, api_name=False)
            interface.chatbot.clear(stop, queue=False, api_name=False)
            gr.Markdown('上传临时文件由Gradio定期清理（最长约2小时）；不启用永久历史。'
                        '审计日志仅在显式配置目录时保存问题/回答与图片哈希。'
                        '立即发送后停止的边界尚未实机验收；遇到异常请等待服务释放后重试。')
    return demo


if __name__ == '__main__':
    build_app().queue(max_size=8).launch(server_name='127.0.0.1', server_port=7860,
        share=False, show_error=False, max_file_size='10mb', show_api=False)
