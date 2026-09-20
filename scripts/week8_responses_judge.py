"""Strict offline Responses envelope validator; never repairs model output."""
from week8_deepseek_judge import validate_response as validate_chat_response


def validate_response(raw):
    if not isinstance(raw,dict) or raw.get('object')!='response':
        raise ValueError('expected response object')
    if raw.get('status')!='completed' or raw.get('error') is not None or raw.get('incomplete_details') is not None:
        raise ValueError('response failed or incomplete')
    items=raw.get('output')
    if not isinstance(items,list) or any(not isinstance(i,dict) or i.get('type') not in ('reasoning','message') or i.get('status')!='completed' for i in items):
        raise ValueError('unexpected output item')
    messages=[i for i in items if i['type']=='message']
    if len(messages)!=1 or messages[0].get('role')!='assistant':
        raise ValueError('expected exactly one assistant message')
    content=messages[0].get('content')
    if not isinstance(content,list) or len(content)!=1 or not isinstance(content[0],dict) or content[0].get('type')!='output_text':
        raise ValueError('expected single output_text')
    usage=raw.get('usage')
    if not isinstance(usage,dict):raise ValueError('missing usage')
    # Explicit protocol normalization reuses the existing score/duplicate-key/range
    # validator. The original Responses envelope is saved unchanged by callers.
    normalized={'model':raw.get('model'),'choices':[{'finish_reason':'stop','message':{'content':content[0].get('text')}}],
                'usage':{'prompt_tokens':usage.get('input_tokens'),'completion_tokens':usage.get('output_tokens'),'total_tokens':usage.get('total_tokens')}}
    return validate_chat_response(normalized,'deepseek-flash')
