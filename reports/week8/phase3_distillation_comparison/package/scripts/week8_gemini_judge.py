"""Strict Gemini judge validation; no network calls on import."""
from week8_deepseek_judge import validate_response as validate_chat
MODEL='gemini-3.1-pro-preview'

def usage(raw):
    u=raw.get('usageMetadata',{})
    values={k:u.get(k,0) for k in ['promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount']}
    if any(type(v) is not int or v<0 for v in values.values()) or 'promptTokenCount' not in u or 'totalTokenCount' not in u:
        raise ValueError('missing or invalid usage')
    if values['totalTokenCount']!=sum(values[k] for k in ['promptTokenCount','candidatesTokenCount','thoughtsTokenCount']):raise ValueError('inconsistent usage')
    return values

def validate_response(raw):
    if not isinstance(raw,dict) or raw.get('modelVersion')!=MODEL:raise ValueError('unreviewed model identity')
    if raw.get('promptFeedback',{}).get('blockReason'):raise ValueError('blocked prompt')
    cs=raw.get('candidates')
    if not isinstance(cs,list) or len(cs)!=1 or cs[0].get('finishReason')!='STOP':raise ValueError('incomplete response')
    content=cs[0].get('content',{});parts=content.get('parts')
    if content.get('role')!='model' or not isinstance(parts,list) or not parts:raise ValueError('invalid content')
    if any(not isinstance(p,dict) or not isinstance(p.get('text'),str) or set(p)-{'text','thought','thoughtSignature'} for p in parts):raise ValueError('unexpected output part')
    texts=[p['text'] for p in parts if not p.get('thought',False)]
    if not texts:raise ValueError('missing final text')
    u=usage(raw)
    return validate_chat({'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':''.join(texts)}}],'usage':{'prompt_tokens':u['promptTokenCount'],'completion_tokens':u['candidatesTokenCount']+u['thoughtsTokenCount'],'total_tokens':u['totalTokenCount']}},MODEL)
