"""Loopback OpenAI-compatible text client; prints streamed deltas as received."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[3]/'source'))
from week7_deployment.service_contract import request_payload,validate_base_url


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--base-url',default=os.environ.get('WEEK7_TEXT_BASE_URL','http://127.0.0.1:8000/v1'))
    parser.add_argument('--temperature',type=float,default=0)
    parser.add_argument('--top-p',type=float,default=1)
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--history',type=Path)
    parser.add_argument('--stream',action='store_true')
    args=parser.parse_args()
    url=validate_base_url(args.base_url)
    history=json.loads(args.history.read_text()) if args.history else []
    payload=request_payload(args.prompt,history,args.temperature,args.top_p,args.max_tokens)
    from openai import OpenAI
    with OpenAI(base_url=url,api_key='local-only',timeout=90,max_retries=0) as client:
        response=client.chat.completions.create(**payload,stream=args.stream)
        if args.stream:
            text='';finish=None
            with response:
                for chunk in response:
                    if not chunk.choices:continue
                    choice=chunk.choices[0]
                    if choice.delta.content:
                        text+=choice.delta.content;print(choice.delta.content,end='',flush=True)
                    if choice.finish_reason:finish=choice.finish_reason
            print()
            if not text.strip() or finish!='stop':raise RuntimeError('empty or incomplete stream')
        else:
            from week7_deployment.service_contract import validate_response
            print(validate_response(response.model_dump()))


if __name__=='__main__':main()
