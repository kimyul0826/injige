"""API orchestration checks use offline fixtures; no credentials or paid calls."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from scripts import run_api_table1 as runner


class APITableTests(unittest.TestCase):
    def setUp(self):
        (runner.ROOT/'.tmp').mkdir(exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=runner.ROOT/'.tmp')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        response = self.root/'response.txt'
        response.write_text('Generated Path: [[5,5],[20,20]]')
        config = self.root/'config.json'
        config.write_text(json.dumps(dict(name='offline-api-test', provider='fixture',
            prompt_family='gpt', response_file=str(response), max_tokens=8000, temperature=0)))
        self.args = argparse.Namespace(config=str(config), input=str(runner.ROOT/'data/environment_50_30.json'),
            output=str(self.root/'output'), prompts=['standard','cot','repe'], limit=2, dry_run=False, zip=True)

    def test_resume_and_portable_zip(self):
        first = runner.run(self.args)
        output = Path(self.args.output)
        times = {p.name:p.stat().st_mtime_ns for p in (output/'responses').glob('*.json')}
        self.assertEqual(first['rows'], 14)
        self.assertEqual(len(times), 6)
        second = runner.run(self.args)
        self.assertEqual(second['state'], 'completed')
        self.assertEqual(times, {p.name:p.stat().st_mtime_ns for p in (output/'responses').glob('*.json')})
        with zipfile.ZipFile(str(output)+'.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertIn('output/input.json', z.namelist())
            self.assertIn('output/status.json', z.namelist())

    def test_plan_needs_no_api_key_and_writes_nothing(self):
        self.args.config = str(runner.ROOT/'configs/gpt4omini_table1_upstream.json')
        self.args.dry_run = True
        with patch.dict('os.environ', {}, clear=True):
            plan = runner.run(self.args)
        self.assertEqual(plan['generation_jobs'], 6)
        self.assertFalse(Path(self.args.output).exists())
