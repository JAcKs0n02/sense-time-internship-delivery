import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import distill
from common import sha256,write_json

class EvidenceTests(unittest.TestCase):
    def test_training_state_rejects_early_stop_and_bad_metrics(self):
        self.assertTrue(hasattr(distill,'validate_student_state'),'student completion validator missing')
        state={'epoch':2.0,'global_step':2,'max_steps':2,'num_train_epochs':2,
               'log_history':[{'step':1,'loss':1.0,'grad_norm':0.5,'learning_rate':5e-5},
                              {'step':1,'epoch':1.0,'eval_loss':1.2},
                              {'step':2,'loss':0.8,'grad_norm':0.3,'learning_rate':0.0},
                              {'step':2,'epoch':2.0,'eval_loss':1.1}]}
        self.assertEqual(distill.validate_student_state(state)['global_step'],2)
        for change in ('epoch','step','nan','no_update','missing_eval'):
            bad=copy.deepcopy(state)
            if change=='epoch':bad['epoch']=1.8
            if change=='step':bad['global_step']=1
            if change=='nan':bad['log_history'][0]['loss']=float('nan')
            if change=='no_update':
                for row in bad['log_history']:
                    if 'grad_norm' in row:row['grad_norm']=0
            if change=='missing_eval':bad['log_history'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):
                distill.validate_student_state(bad)

    def test_review_binds_targets_receipt_and_selected_answers(self):
        self.assertTrue(hasattr(distill,'verify_target_review'),'teacher target review gate missing')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            rows=[{'sample_id':str(i),'instruction':str(i),'output':'answer'} for i in range(10)]
            targets=out/'teacher_targets.json';write_json(targets,rows)
            source=out/'selected_prompts.json';write_json(source,[{'sample_id':str(i),'instruction':str(i)} for i in range(10)])
            teacher=out/'teacher_identity.json'
            teacher.write_bytes((ROOT/'deliverables/week4/day20/source/results/corrective_final/merged_model_manifest.json').read_bytes())
            raw=[]
            for row in rows:
                path=out/'teacher_raw'/f"{int(row['sample_id']):04d}.json"
                write_json(path,{'sample_id':row['sample_id'],'messages':distill.teacher_messages(row),'answer':'answer','truncated':False})
                raw.append(path)
            receipt=out/'generation_receipt.json'
            write_json(receipt,{'status':'GENERATED_PENDING_QUALITY_REVIEW','accepted':10,
                               'config':{'epochs':2},'generated':10,
                               'teacher_manifest_sha256':sha256(teacher),
                               'files':[{'path':str(p.relative_to(out)),'sha256':sha256(p)} for p in [targets,source,teacher,*raw]]})
            review=out/'review.json'
            value={'status':'APPROVED_FOR_STUDENT_TRAINING','generation_receipt_sha256':sha256(receipt),
                   'targets_sha256':sha256(targets),'reviewed_sample_ids':[str(i) for i in range(10)],
                   'notes':'Checked completeness and relevance; no unresolved issues.','unresolved_issues':[]}
            write_json(review,value)
            self.assertEqual(distill.verify_target_review(targets,receipt,review,{'epochs':2})['accepted'],10)
            for change in ('pending','stale','unreviewed','issue','missing_binding','changed_source'):
                bad=copy.deepcopy(value)
                if change=='pending':bad['status']='PENDING'
                if change=='stale':bad['targets_sha256']='wrong'
                if change=='unreviewed':bad['reviewed_sample_ids']=['unknown']
                if change=='issue':bad['unresolved_issues']=['wrong answer']
                if change=='missing_binding':
                    original=receipt.read_text()
                    broken=json.loads(original);broken['files']=[e for e in broken['files'] if e['path']!='selected_prompts.json']
                    write_json(receipt,broken);bad['generation_receipt_sha256']=sha256(receipt)
                if change=='changed_source':source.write_text('changed')
                write_json(review,bad)
                with self.subTest(change=change),self.assertRaises(ValueError):
                    distill.verify_target_review(targets,receipt,review,{'epochs':2})
                if change=='missing_binding':receipt.write_text(original)

    def test_generation_requires_manifest_before_loading_model(self):
        from argparse import Namespace
        self.assertTrue(hasattr(distill,'verify_teacher_identity'),'teacher provenance gate missing')
        with self.assertRaises(ValueError):
            distill.verify_teacher_identity(Namespace(teacher='/unverified',teacher_manifest=None))

    def test_training_cannot_bypass_teacher_review(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            args=Namespace(student='/unverified',input=Path(tmp)/'targets.json',
                           output_dir=Path(tmp)/'training',generation_receipt=None,quality_review=None)
            with self.assertRaisesRegex(ValueError,'generation-receipt'):
                distill.train(args,{'epochs':2})
            self.assertFalse(args.output_dir.exists())

    def test_two_epoch_schedule_covers_partial_accumulation_batch(self):
        self.assertTrue(hasattr(distill,'student_training_schedule'),'full epoch step schedule missing')
        plan=distill.student_training_schedule(180)
        self.assertEqual(plan['max_steps'],46)
        self.assertEqual(plan['trainer_epoch_bound'],3)
        self.assertEqual(plan['sample_presentations'],360)
        state={'epoch':2.0,'global_step':46,'max_steps':46,'num_train_epochs':3,
               'log_history':[{'step':i,'loss':1.0,'grad_norm':0.1,'learning_rate':1e-5} for i in range(1,47)]
                  +[{'step':23,'epoch':1.0,'eval_loss':1.1},{'step':46,'epoch':2.0,'eval_loss':1.0}]}
        self.assertEqual(distill.validate_student_state(state,plan)['epochs'],2)
        bad=copy.deepcopy(state);bad.update(global_step=44,max_steps=44,epoch=1.9333333333333333,num_train_epochs=2)
        with self.assertRaises(ValueError):distill.validate_student_state(bad,plan)
        self.assertEqual(distill.student_training_schedule(176)['max_steps'],44)
        with self.assertRaises(ValueError):distill.student_training_schedule(0)

if __name__=='__main__':unittest.main()
