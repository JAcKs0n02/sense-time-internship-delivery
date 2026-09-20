"""Dependency-light text service contracts; loopback only for this deployment."""
import math
import os
import signal
import subprocess
import time
from urllib.parse import urlsplit

MODEL_NAME='week4-dpo-quantized'


def server_arguments(python,model,port):
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('invalid port')
    return [str(python),'-m','vllm.entrypoints.openai.api_server','--model',str(model),
        '--host','127.0.0.1','--port',str(port),'--served-model-name',MODEL_NAME,
        '--quantization','awq','--dtype','half','--max-model-len','2048',
        '--gpu-memory-utilization','0.75','--max-num-seqs','4','--enforce-eager',
        '--disable-log-requests']


def validate_base_url(url):
    parsed=urlsplit(url)
    if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.path!='/v1' or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError('loopback /v1 endpoint required')
    if parsed.port is None or not 1024<=parsed.port<=65535:raise ValueError('explicit valid port required')
    return url


def request_payload(prompt,history,temperature=0,top_p=1,max_tokens=128):
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('nonempty prompt required')
    if type(temperature) not in [int,float] or not math.isfinite(temperature) or not 0<=temperature<=2:raise ValueError('invalid temperature')
    if type(top_p) not in [int,float] or not math.isfinite(top_p) or not 0<top_p<=1:raise ValueError('invalid top_p')
    if type(max_tokens) is not int or not 1<=max_tokens<=512:raise ValueError('invalid max_tokens')
    if not isinstance(history,list) or len(history)%2:raise ValueError('complete history turns required')
    messages=[]
    for i,item in enumerate(history):
        expected='user' if i%2==0 else 'assistant'
        if not isinstance(item,dict) or item.get('role')!=expected or not isinstance(item.get('content'),str) or not item['content'].strip():
            raise ValueError('invalid history role/content')
        messages.append({'role':expected,'content':item['content']})
    return {'model':MODEL_NAME,'messages':messages+[{'role':'user','content':prompt}],
            'temperature':temperature,'top_p':top_p,'max_tokens':max_tokens,'seed':20260911}


def consume_stream(chunks,started=None):
    started=time.monotonic() if started is None else started
    records=[];text='';finish=None
    for chunk in chunks:
        if not chunk.choices:continue
        choice=chunk.choices[0];content=choice.delta.content
        if content is not None:
            if not isinstance(content,str):raise ValueError('invalid stream content')
            text+=content
        if choice.finish_reason is not None:finish=choice.finish_reason
        records.append({'content':content,'finish_reason':choice.finish_reason,'elapsed_seconds':time.monotonic()-started})
    if not text.strip() or finish is None:raise ValueError('empty or incomplete stream')
    return records,text,finish


def validate_response(row):
    if row.get('model')!=MODEL_NAME:raise ValueError('wrong response model')
    choices=row.get('choices',[])
    if len(choices)!=1 or choices[0].get('finish_reason')!='stop':raise ValueError('unfinished response')
    content=choices[0].get('message',{}).get('content')
    usage=row.get('usage') or {}
    if not isinstance(content,str) or not content.strip() or usage.get('completion_tokens',0)<=0:
        raise ValueError('empty response or usage')
    return content


def stop_owned_process(proc,grace_seconds=20):
    # Use immediately for our start_new_session=True handle, never a loaded PID.
    if getattr(proc,'week7_group_drained',False):return
    if proc.poll() is None and os.getpgid(proc.pid)!=proc.pid:raise ValueError('not an owned session leader')
    def live_members():
        rows=subprocess.check_output(['ps','-eo','pid=,pgid=,stat='],text=True).splitlines()
        return [int(parts[0]) for row in rows if len(parts:=row.split())==3 and int(parts[1])==proc.pid and not parts[2].startswith('Z')]
    if live_members():
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+grace_seconds
    while live_members() and time.monotonic()<deadline:time.sleep(.1)
    if live_members():
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    proc.wait(timeout=10)
    deadline=time.monotonic()+5
    while live_members() and time.monotonic()<deadline:time.sleep(.1)
    if live_members():raise RuntimeError('owned process group still alive')
    proc.week7_group_drained=True
