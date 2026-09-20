"""Tiny CPU schedule probe only; never trains the student or teacher model."""
import json, math, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WANDB_DISABLED']='true'
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
torch.set_num_threads(1)
OUT=Path('/root/autodl-tmp/week8-distillation-preflight-20260918')
class Rows(Dataset):
    def __len__(self):return 180
    def __getitem__(self,i):return {'input_ids':torch.tensor([float(i)/180]),'labels':torch.tensor([0.0])}
class Tiny(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([0.1]));self.seen=0
    def forward(self,input_ids,labels=None):
        if self.training:self.seen+=len(input_ids)
        logits=input_ids*self.weight
        return {'loss':(logits-labels).square().mean(),'logits':logits}
results=[]
for name,max_steps in [('epoch_default',-1),('explicit_full_epoch_steps',46)]:
    args=TrainingArguments(output_dir=str(OUT/name),num_train_epochs=2,max_steps=max_steps,
        per_device_train_batch_size=1,per_device_eval_batch_size=8,gradient_accumulation_steps=8,
        eval_strategy='epoch',save_strategy='no',logging_steps=1,report_to=[],use_cpu=True,
        disable_tqdm=True,dataloader_num_workers=0,seed=42)
    model=Tiny();trainer=Trainer(model=model,args=args,train_dataset=Rows(),eval_dataset=Rows())
    trainer.train();state=trainer.state
    result={'name':name,'global_step':state.global_step,'max_steps':state.max_steps,
            'epoch':state.epoch,'num_train_epochs':state.num_train_epochs,'training_examples_seen':model.seen,
            'eval_epochs':[r['epoch'] for r in state.log_history if 'eval_loss' in r]}
    results.append(result)
    print('PROBE_RESULT',json.dumps(result),flush=True)
(OUT/'epoch_probe.json').write_text(json.dumps({'scope':'tiny CPU model, trainer schedule only','results':results},indent=2))
