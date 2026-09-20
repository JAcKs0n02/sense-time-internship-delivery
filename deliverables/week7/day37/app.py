"""Day37 text UI. No model loading, automatic instance startup or cloud calls."""
import json
import os
from pathlib import Path
import sys
import time
import uuid
import asyncio
from contextlib import aclosing

sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from week7_deployment.service_contract import request_payload,validate_base_url


def friendly_error(exc):
    status=getattr(exc,'status_code',None)
    if status==404:return '文字模型未就绪，请检查服务名称并启动对应服务。'
    if status in [401,403]:return '服务认证失败，请检查本地连接设置。'
    if status==400:return '请求超出模型上下文或参数限制，请清空历史、缩短输入后重试。'
    if isinstance(exc,ValueError):return '输入或参数不合法，请检查内容、历史和生成参数。'
    return '无法完成生成，请确认文字服务已启动且连接正常；未完成回答不应视为完整结果。'


class UserStopped(Exception):pass


async def until_stopped(awaitable,stop_event):
    if stop_event is None:return await awaitable
    operation=asyncio.ensure_future(awaitable)
    stopped=asyncio.create_task(stop_event.wait())
    try:
        await asyncio.wait([operation,stopped],return_when=asyncio.FIRST_COMPLETED)
        if stop_event.is_set():raise UserStopped()
        return await operation
    finally:
        for task in [operation,stopped]:
            if not task.done():task.cancel()
        await asyncio.gather(operation,stopped,return_exceptions=True)


async def stream_chat(message,history,temperature=.7,top_p=.9,max_tokens=256,*,client=None,stop_event=None):
    payload=request_payload(message,history,temperature,top_p,int(max_tokens) if type(max_tokens) is float and max_tokens.is_integer() else max_tokens)
    audit=None;owned=client is None;status='interrupted';started=time.monotonic()
    audit_dir=os.environ.get('WEEK7_UI_AUDIT_DIR')
    if audit_dir:
        directory=Path(audit_dir);directory.mkdir(parents=True,exist_ok=True)
        audit=(directory/(uuid.uuid4().hex+'.jsonl')).open('x',encoding='utf-8')
    def record(event,**values):
        if audit:
            audit.write(json.dumps({'event':event,'elapsed_seconds':time.monotonic()-started,**values},ensure_ascii=False)+'\n');audit.flush()
    try:
        record('request',payload=payload)
        if owned:
            from openai import AsyncOpenAI
            url=validate_base_url(os.environ.get('WEEK7_TEXT_BASE_URL','http://127.0.0.1:8000/v1'))
            client=AsyncOpenAI(base_url=url,api_key='local-only',timeout=90,max_retries=0)
        text='';finish=None
        async with await until_stopped(client.chat.completions.create(**payload,stream=True),stop_event) as response:
            iterator=response.__aiter__()
            while True:
                try:chunk=await until_stopped(anext(iterator),stop_event)
                except StopAsyncIteration:break
                if not chunk.choices:continue
                choice=chunk.choices[0]
                if choice.delta.content:
                    text+=choice.delta.content
                    record('delta',content=choice.delta.content)
                    yield text
                if choice.finish_reason:finish=choice.finish_reason
        if not text.strip() or finish!='stop':raise RuntimeError('empty, interrupted or truncated response')
        status='complete';record('complete',finish_reason=finish,text=text)
    except UserStopped:
        status='stopped';record('stop_received',requested_elapsed_seconds=getattr(stop_event,'requested_at',time.monotonic())-started);raise
    except Exception as exc:
        status='error';record('error',error_type=type(exc).__name__);raise
    finally:
        if owned and client is not None:await client.close()
        record('closed',status=status)
        if audit:audit.close()


async def ui_chat(message,history,temperature,top_p,max_tokens,*,stop_event=None):
    last=None
    try:
        # Gradio retains a lone user message when stopped before the first token.
        # Such turns can remain before later answered turns in Gradio's display.
        # Omit only users followed by another user (or EOF), preserving complete pairs.
        rows=list(history)
        def is_user(row):return isinstance(row,dict) and row.get('role')=='user'
        history=[row for i,row in enumerate(rows)
                 if not (is_user(row) and (i==len(rows)-1 or is_user(rows[i+1])))]
        async with aclosing(stream_chat(message,history,temperature,top_p,max_tokens,stop_event=stop_event)) as stream:
            async for text in stream:
                last=text
                yield text
    except UserStopped:return
    except Exception as exc:
        if last is not None:
            # Normal exhaustion lets Gradio clear its iterator-keyed diff state.
            # Keep the partial result visible, but never present it as complete.
            yield f'{last}\n\n⚠️ 回答未完整结束。{friendly_error(exc)}'
            return
        import gradio as gr
        raise gr.Error(friendly_error(exc)) from None


def build_app(*, introduction=None):
    import gradio as gr
    class ClearCancelsChatInterface(gr.ChatInterface):
        # Gradio 5.31.0's Clear resets state but does not cancel active events.
        # Use the same event handles as Stop so late output cannot restore history.
        def _setup_stop_events(self,event_triggers,events_to_cancel):
            super()._setup_stop_events(event_triggers,events_to_cancel)
            self.chatbot.clear(None,cancels=events_to_cancel,queue=False,api_name=False)
    sessions={}
    async def chat(message,history,temperature,top_p,max_tokens,request:gr.Request):
        key=request.session_hash
        if not key:raise gr.Error('缺少会话标识，请刷新页面后重试。')
        event=asyncio.Event();sessions[key]=event
        try:
            async with aclosing(ui_chat(message,history,temperature,top_p,max_tokens,stop_event=event)) as stream:
                async for value in stream:yield value
        finally:
            if sessions.get(key) is event:sessions.pop(key,None)
    async def stop(request:gr.Request):
        event=sessions.get(request.session_hash)
        if event:
            event.requested_at=time.monotonic()
            event.set()
    with gr.Blocks(title='Week7 · 量化模型对话',analytics_enabled=False,
                   theme=gr.themes.Soft(font=['sans-serif'])) as demo:
        gr.Markdown(introduction if introduction is not None else '# Week7 · 量化模型对话\nWeek4 DPO · AWQ 4-bit · 本机 vLLM\n\n工程演示，非质量恢复认证。图片输入将在 Day38 接入；请勿输入敏感资料。')
        with gr.Row():
            temperature=gr.Slider(0,2,value=.7,step=.1,label='Temperature')
            top_p=gr.Slider(.05,1,value=.9,step=.05,label='Top-p')
            max_tokens=gr.Slider(32,512,value=256,step=32,label='最大输出 token')
        interface=ClearCancelsChatInterface(chat,type='messages',
            chatbot=gr.Chatbot(type='messages',label='聊天记录',height=430,show_copy_button=True),
            textbox=gr.Textbox(placeholder='输入问题，按 Enter 发送',label='消息',lines=1,submit_btn='发送',stop_btn='停止'),
            additional_inputs=[temperature,top_p,max_tokens],
            submit_btn='发送',stop_btn='停止',save_history=False,analytics_enabled=False,
            flagging_mode='never',concurrency_limit=1,api_name='chat')
        interface.textbox.stop(stop,queue=False,api_name=False)
        interface.chatbot.clear(stop,queue=False,api_name=False)
        gr.Markdown('使用聊天区域的清空按钮开始新会话；生成中可停止。历史只保留在当前会话，未配置持久保存。')
    return demo


def main():
    demo=build_app()
    demo.queue(max_size=8).launch(server_name='127.0.0.1',server_port=7860,share=False,show_error=False)


if __name__=='__main__':main()
