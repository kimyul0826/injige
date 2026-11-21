"""Offline checks for Batch cache reuse, out-of-order results and resume without resubmission."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import zipfile
from scripts import run_batch_table1 as runner
from llm_astar.artifacts import write_json


class BatchTests(unittest.TestCase):
    def setUp(self):
        (runner.ROOT/'.tmp').mkdir(exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=runner.ROOT/'.tmp')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.args = argparse.Namespace(config=str(runner.ROOT/'configs/gpt4omini_table1_upstream.json'),
            input=str(runner.ROOT/'data/environment_50_30.json'), output=str(self.root/'out'),
            reuse_cache=str(self.root/'old'), prompts=['standard','cot','repe'], limit=1,
            dry_run=False, submit_only=True, retry_failed=False, zip=True)
        config = json.loads(Path(self.args.config).read_text())
        self.jobs = runner.jobs_for(config, list(runner.load_queries(self.args.input))[:1], self.args.prompts)
        key,job = next(iter(self.jobs.items()))
        self.oldkey = key
        write_json(self.root/'old'/(key+'.json'), dict(config=job['config'], prompt=job['prompt'],
            cache_key=key, text='[[5,5],[20,20]]', usage=None, finish_reason='stop',
            cache_hit=False, generation_seconds=1, synthetic=False))
        self.client = Mock()
        self.client.files.create.return_value = SimpleNamespace(id='file-input')
        self.client.batches.create.return_value = SimpleNamespace(id='batch-test')

    def batch(self, status):
        data = dict(id='batch-test', status=status, output_file_id='file-output' if status=='completed' else None,
                    error_file_id=None)
        return SimpleNamespace(**data, model_dump=lambda:data)

    def responses(self, keys):
        return '\n'.join(json.dumps(dict(custom_id=k, error=None, response=dict(status_code=200,
            request_id='req-'+k, body=dict(model='gpt-4o-mini-2024-07-18',
            choices=[dict(message=dict(content='[[5,5],[20,20]]'), finish_reason='stop')],
            usage=dict(prompt_tokens=10, completion_tokens=8, total_tokens=18))))) for k in reversed(keys))

    def test_submit_resume_collect_and_reuse(self):
        self.client.batches.retrieve.return_value = self.batch('in_progress')
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(self.args, self.client)
        path = self.root/'out/logs/batch/requests-1.jsonl'
        requests = [json.loads(l) for l in path.read_text().splitlines()]
        self.assertEqual(len(requests), 2)
        self.assertNotIn(self.oldkey, [r['custom_id'] for r in requests])
        for r in requests:
            job = self.jobs[r['custom_id']]
            self.assertEqual(r['body']['messages'], [dict(role='user',content=job['prompt'])])
            self.assertEqual(r['body']['max_tokens'], 8000)
        self.client.batches.retrieve.return_value = self.batch('completed')
        self.client.files.content.return_value = SimpleNamespace(text=self.responses([r['custom_id'] for r in requests]))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = runner.run(self.args, self.client)
            runner.run(self.args, self.client)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(self.client.batches.create.call_count, 1)
        self.assertEqual(self.client.files.create.call_count, 1)
        rows = (self.root/'out/results.jsonl').read_text().splitlines()
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(list((self.root/'out/responses').glob('*.json'))),3)
        with zipfile.ZipFile(str(self.root/'out')+'.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertIn('out/logs/batch/state.json', z.namelist())

    def test_failed_requests_are_not_regular_api_fallback(self):
        self.client.batches.retrieve.return_value = self.batch('completed')
        missing = [k for k in self.jobs if k!=self.oldkey]
        self.client.files.content.return_value = SimpleNamespace(text=self.responses(missing[:1]))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, '1 requests remain'):
                runner.run(self.args,self.client)
            with self.assertRaisesRegex(ValueError, 'Some requests failed'):
                runner.run(self.args,self.client)
        self.assertFalse((self.root/'out/results.jsonl').exists())
        self.assertEqual(self.client.batches.create.call_count,1)

    def test_dry_run_has_no_submission_or_writes(self):
        self.args.dry_run=True
        with contextlib.redirect_stdout(io.StringIO()):
            result=runner.run(self.args,self.client)
        self.assertEqual(result['remaining_requests'],2)
        self.assertFalse((self.root/'out').exists())
        self.client.files.create.assert_not_called()
