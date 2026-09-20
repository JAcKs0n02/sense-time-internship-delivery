#!/usr/bin/env python3
"""CPU verification worker, isolated so the coordinator can terminate it on timeout."""
import argparse
from pathlib import Path
from common import sha256,write_json
from week8_full_training import verify_formal_adapter
from week8_training_smoke import verify_merge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['adapter','merge']);p.add_argument('--path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int)
    p.add_argument('--base');p.add_argument('--stage',choices=['sft','dpo'])
    a=p.parse_args()
    if a.output.exists():raise ValueError('verification output already exists')
    if a.kind=='adapter':
        if a.steps is None or a.base is None or a.stage is None:raise ValueError('adapter verification parameters missing')
        result=verify_formal_adapter(a.path,a.steps,a.base,a.stage)
    else:
        result=verify_merge(a.path)
        result['model_path']=str(a.path.resolve())
        result['identity']=[{'path':str(x.relative_to(a.path)),'sha256':sha256(x),'bytes':x.stat().st_size}
                            for x in sorted(a.path.rglob('*')) if x.is_file()]
    write_json(a.output,result)

if __name__=='__main__':main()
