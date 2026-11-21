"""RePE2: every demonstration round and Check verdict matches the exact collision test, rounds are minimal."""
import ast
import re
import unittest

from llm_astar.__main__ import load_queries
from llm_astar.models import build_prompt, parse_path
from llm_astar.prompt_variants import REPE2_DEMOS, REPE2_INSTRUCTION, REPE2_ORIG_DEMOS
from llm_astar.search import Environment

Q0 = dict(range_x=[0, 51], range_y=[0, 31], start=[5, 5], goal=[20, 20],
          horizontal_barriers=[[10, 0, 25], [15, 30, 50]], vertical_barriers=[[25, 10, 22]])


def demos(text=REPE2_DEMOS):
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        val = lambda key: ast.literal_eval(next(l for l in lines if l.startswith(key)).split(': ', 1)[1])
        q = dict(range_x=[0, 51], range_y=[0, 31], start=val('Start Point'), goal=val('Goal Point'),
                 horizontal_barriers=val('Horizontal Barriers'), vertical_barriers=val('Vertical Barriers'))
        rounds = [(int(m[1]), ast.literal_eval(m[2])) for m in re.finditer(r'Round (\d+): (\[\[.*\]\])', block)]
        verdicts = re.findall(r'(Not valid|Valid)\.$', block, re.M)
        yield q, rounds, verdicts, val('Generated Path')


class Repe2Tests(unittest.TestCase):
    def test_demo_checks_match_collision_test(self):
        for q, rounds, verdicts, answer in list(demos()) + list(demos(REPE2_ORIG_DEMOS)):
            env = Environment(q, collision_backend='shapely')
            self.assertEqual(len(rounds), len(verdicts))
            for (k, path), verdict in zip(rounds, verdicts):
                with self.subTest(start=q['start'], round=k):
                    self.assertEqual(len(path) - 2, k)
                    valid = not any(env.collision(tuple(a), tuple(b)) for a, b in zip(path, path[1:]))
                    self.assertEqual(valid, verdict == 'Valid')
            self.assertEqual(verdicts[-1], 'Valid')
            self.assertEqual(rounds[-1][1], answer)
            self.assertEqual([k for k, _ in rounds], list(range(len(rounds))))

    def test_demo_rounds_are_minimal(self):
        # exact minimal waypoint count over every cell the official filter keeps (exp/claude/build_repe2_demos.solve)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp' / 'claude'))
        import build_repe2_demos as b
        ks = []
        for q, rounds, _, _ in demos():
            k, _ = b.solve(q)
            self.assertEqual(rounds[-1][0], k)
            ks.append(k)
        self.assertEqual(sorted(ks), [0, 1, 1, 2, 3])

    def test_demos_not_in_evaluation_set(self):
        import json
        from pathlib import Path
        data = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'environment_50_30.json').read_text())
        maps = {(json.dumps(m['horizontal_barriers']), json.dumps(m['vertical_barriers'])) for m in data}
        for q, _, _, _ in demos():
            self.assertNotIn((json.dumps(q['horizontal_barriers']), json.dumps(q['vertical_barriers'])), maps)

    def test_prompt_structure(self):
        t = build_prompt(Q0, 'repe2', 'gpt')
        self.assertEqual(t.count(REPE2_INSTRUCTION), 1)
        self.assertTrue(t.rstrip().endswith('Vertical Barriers: [[25, 10, 22]]'))
        self.assertEqual(t.count('Start Point:'), 6)
        self.assertNotIn('Barrier Endpoints', t)
        with self.assertRaises(ValueError):
            build_prompt(Q0, 'repe2', 'unknown-family')

    def test_ablation_e1a_noround(self):
        t = build_prompt(Q0, 'repe2_noround', 'gpt')
        self.assertNotIn('Round', t)
        self.assertNotIn('Check:', t)
        self.assertNotIn(REPE2_INSTRUCTION, t)
        header = build_prompt(Q0, 'standard', 'gpt').split('Start Point: [5, 5]')[0]
        self.assertTrue(t.startswith(header))   # released standard instruction, unchanged
        for q, rounds, _, answer in demos():   # same problems and answers as RePE2, answer only
            self.assertIn(f"Start Point: {q['start']}\nGoal Point: {q['goal']}", t)
            self.assertIn(f'Generated Path: {answer}', t)
        self.assertTrue(t.rstrip().endswith('Generated Path:'))
        self.assertEqual(t.count('Start Point:'), 6)

    def test_ablation_e1b_origdemos(self):
        t = build_prompt(Q0, 'repe2_origdemos', 'gpt')
        self.assertEqual(t.count(REPE2_INSTRUCTION), 1)
        self.assertIn(REPE2_ORIG_DEMOS, t)
        self.assertTrue(t.rstrip().endswith('Vertical Barriers: [[25, 10, 22]]'))
        released = [q['start'] for q, _, _, _ in demos(REPE2_ORIG_DEMOS)]
        self.assertEqual(released, [[5, 5], [2, 3], [40, 20], [4, 4], [25, 25]])
        # differs from full RePE2 only in the demonstrations
        full = build_prompt(Q0, 'repe2', 'gpt')
        self.assertEqual(t.replace(REPE2_ORIG_DEMOS, 'X'), full.replace(REPE2_DEMOS, 'X'))

    def test_llama_layout_matches_released(self):
        from llm_astar.vendor.prompts import llama_prompt
        released = build_prompt(Q0, 'standard', 'llama')
        # E1-a in Llama layout differs from the released Llama prompt only in the five demonstrations
        noround = build_prompt(Q0, 'repe2_noround', 'llama')
        head = released[:released.index('<|start_header_id|>user')]
        tail = released[released.rindex('<|start_header_id|>user'):]
        self.assertTrue(noround.startswith(head) and noround.endswith(tail))
        for style, demos in (('repe2', REPE2_DEMOS), ('repe2_origdemos', REPE2_ORIG_DEMOS)):
            t = build_prompt(Q0, style, 'llama')
            self.assertEqual(t.count(REPE2_INSTRUCTION), 1)
            self.assertTrue(t.endswith(tail))
            self.assertEqual(t.count('<|start_header_id|>user<|end_header_id|>'), 6)
            self.assertEqual(t.count('<|start_header_id|>assistant<|end_header_id|>'), 6)
            # every demonstration answer appears inside an assistant turn, rounds included
            for block in demos.strip().split('\n\n'):
                lines = block.split('\n')
                self.assertIn('<|start_header_id|>assistant<|end_header_id|>\n' + '\n'.join(lines[4:]) + '\n<|eot_id|>', t)
        # released demonstrations re-solved in rounds keep the released Llama answers as final paths
        orig = build_prompt(Q0, 'repe2_origdemos', 'llama')
        for answer in ('[[5, 5], [26, 9], [25, 23], [20, 20]]', '[[40, 20], [20, 3], [10, 10]]'):
            self.assertIn('Generated Path: ' + answer, orig)

    def test_parser_takes_final_generated_path(self):
        reply = 'Round 0: [[5, 5], [20, 20]]\nCheck: Not valid.\nRound 1: [[5, 5], [26, 9], [20, 20]]\nCheck: Valid.\nGenerated Path: [[5, 5], [26, 9], [20, 20]]'
        self.assertEqual(parse_path(reply, 'upstream'), [(5, 5), (26, 9), (20, 20)])

    def test_dataset_prompts_build(self):
        for _, q in load_queries('data/environment_50_30.json'):
            build_prompt(q, 'repe2', 'gpt')


if __name__ == '__main__':
    unittest.main()
