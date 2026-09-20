"""Validate score data carried by a strict tool call; never execute tools."""
from week8_deepseek_judge import validate_response as validate_chat

def validate_response(raw):
    if not isinstance(raw,dict):raise ValueError('invalid envelope')
    choices=raw.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict) or choices[0].get('finish_reason')!='tool_calls':
        raise ValueError('expected completed tool call')
    message=choices[0].get('message')
    if not isinstance(message,dict) or message.get('role')!='assistant' or message.get('content') not in (None,''):
        raise ValueError('unexpected assistant message')
    calls=message.get('tool_calls')
    if not isinstance(calls,list) or len(calls)!=1 or not isinstance(calls[0],dict):raise ValueError('expected one tool call')
    call=calls[0];f=call.get('function')
    if call.get('type')!='function' or not isinstance(call.get('id'),str) or not call['id'] or not isinstance(f,dict) or f.get('name')!='submit_scores':
        raise ValueError('unexpected function')
    # Normalize only the envelope; arguments are parsed verbatim without repair.
    return validate_chat({'model':raw.get('model'),'usage':raw.get('usage'),'choices':[{'finish_reason':'stop','message':{'content':f.get('arguments')}}]},'deepseek-v4-pro')
