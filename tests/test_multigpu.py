"""Persistent CPU-only orchestration regressions; temporary artifacts are auto-removed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from llm_astar.models import Model
from llm_astar.artifacts import bundle_results
from scripts import run_multigpu as runner


def failing_worker(device, config, tasks, events, output):
    events.put(('error', 'intentional worker failure', device))
    raise RuntimeError('Intentional orchestration test failure')


class DistributedTests(unittest.TestCase):
    def setUp(self):
        (runner.ROOT/'.tmp').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runner.ROOT/'.tmp')
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.response = self.root/'fixture.txt'
        self.response.write_text('[[5,5],[20,20]]')
        self.config = self.root/'config.json'
        self.config.write_text(json.dumps(dict(name='fixture-test', provider='fixture',
            response_file=str(self.response), prompt_family='llama', max_tokens=8000, temperature=0)))
        self.args = argparse.Namespace(config=str(self.config), model_path=None,
            input=str(runner.ROOT/'data/environment_50_30.json'), output=str(self.root/'output'),
            limit=12, prompts=['standard', 'cot', 'repe'], gpus='0,1', zip=True)

    def test_distribution_resume_merge_and_archive(self):
        first = runner.run(self.args)
        self.assertEqual(first['state'], 'completed')
        output = Path(self.args.output)
        self.assertEqual(len(list((output/'jobs').glob('*.json'))), 36)
        self.assertEqual(len(list((output/'logs').glob('*.log'))), 2)
        rows = [json.loads(x) for x in (output/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 84)
        self.assertEqual(len({(r['sample'], r['prompt'], r['method']) for r in rows}), 84)
        self.assertTrue(all(r.get('synthetic') for r in rows if r['method'] != 'astar'))
        mtimes = {p.name: p.stat().st_mtime_ns for p in (output/'responses').glob('*.json')}
        # Simulate interruption after saving the raw response but before committing its job.
        next((output/'jobs').glob('*.json')).unlink()
        self.args.gpus = '2'
        resumed = runner.run(self.args)
        self.assertEqual(resumed['completed_jobs'], 36)
        self.assertEqual(mtimes, {p.name: p.stat().st_mtime_ns for p in (output/'responses').glob('*.json')})
        runtime = json.loads((output/'runtime.json').read_text())
        self.assertEqual(runtime['resumed_jobs'], 35)
        self.assertEqual(runtime['devices'], ['2'])
        (output/'model.safetensors').write_text('not included')
        (output/'.env').write_text('not included')
        (output/'responses'/'linked.json').symlink_to(self.config)
        archive = bundle_results(output)
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            names = z.namelist()
            self.assertIn('output/table1.csv', names)
            self.assertIn('output/status.json', names)
            self.assertFalse(any('linked.json' in p or 'safetensors' in p or '.env' in p for p in names))
        self.args.prompts = ['standard']
        with self.assertRaisesRegex(ValueError, 'changed'):
            runner.run(self.args)

    def test_worker_failure_retains_partial_status_and_zip(self):
        with patch.object(runner, 'worker', failing_worker):
            with self.assertRaisesRegex(RuntimeError, 'failed|exited'):
                runner.run(self.args)
        output = Path(self.args.output)
        self.assertEqual(json.loads((output/'status.json').read_text())['state'], 'failed')
        self.assertTrue(Path(str(output)+'.zip').is_file())
        self.assertEqual(len((output/'results.jsonl').read_text().splitlines()), 12)

    def test_concurrent_identical_cache_writes(self):
        config = json.loads(self.config.read_text())
        query = dict(start=[5,5],goal=[20,20],horizontal_barriers=[],vertical_barriers=[])
        def generate(_):
            return Model(config, str(self.root/'cache')).generate(query, 'standard')
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(generate, range(32)))
        self.assertEqual(len({r['cache_key'] for r in results}), 1)
        saved = list((self.root/'cache').glob('*.json'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text())['text'], '[[5,5],[20,20]]')
        self.assertFalse(list((self.root/'cache').glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
