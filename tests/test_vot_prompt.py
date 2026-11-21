"""Zero-shot VoT keeps the released CoT instructions/query; the ASCII map only adds input."""
import json
from pathlib import Path
import unittest
from llm_astar.models import build_prompt, parse_path
from llm_astar.prompt_variants import COARSE_REASONING, MAP_LEGEND, VOT_TRIGGER, ascii_map

ROOT = Path(__file__).resolve().parents[1]
QUERY = dict(start=[1, 6], goal=[16, 23], range_x=[0, 51], range_y=[0, 31],
             horizontal_barriers=[[11, 18, 34]],
             vertical_barriers=[[7, 20, 29], [9, 3, 15], [13, 19, 21], [15, 5, 24]])


class VotPromptTests(unittest.TestCase):
    def test_zero_shot_has_no_demonstrations(self):
        for family in ('gpt', 'llama'):
            with self.subTest(family=family):
                prompt = build_prompt(QUERY, 'vot', family)
                cot = build_prompt(QUERY, 'cot', family)
                self.assertEqual(prompt.count('Start Point:'), 1)
                self.assertEqual(prompt.count(VOT_TRIGGER), 1)
                self.assertNotIn('Thought:', prompt)
                # Instruction lines and the final query are copied from the released CoT prompt.
                self.assertTrue(cot.rstrip().endswith(prompt.split('Start Point:')[1].rstrip()))
                for line in cot.split('Start Point:')[0].splitlines():
                    if line.strip() and not line.startswith('<|start_header_id|>user'):
                        self.assertIn(line, prompt)

    def test_map_variant_only_adds_legend_and_grid(self):
        for style in ('vot', 'standard', 'cot'):
            with self.subTest(style=style):
                plain = build_prompt(QUERY, style, 'gpt')
                mapped = build_prompt(QUERY, style, 'gpt', 'ascii_map')
                grid = '\nMap:\n' + ascii_map(QUERY)
                self.assertEqual(mapped.count(grid), 1)
                self.assertGreater(mapped.index(grid), mapped.rfind('Start Point:'))
                self.assertEqual(mapped.replace(MAP_LEGEND + '\n', '', 1).replace(grid, '', 1), plain)

    def test_grid_coordinates(self):
        rows = ascii_map(QUERY).splitlines()
        self.assertEqual(len(rows), 2 + 31 + 1)
        cells = {int(r[:2]): r[3:] for r in rows[2:-1]}
        self.assertEqual(len(cells[0]), 51)
        self.assertEqual(cells[6][1], 'S')
        self.assertEqual(cells[23][16], 'G')
        self.assertEqual(cells[11][18:35], '#' * 17)
        self.assertEqual(cells[11][35], '.')
        self.assertTrue(all(cells[y][9] == '#' for y in range(3, 16)))
        self.assertEqual(cells[0], '#' * 51)
        self.assertTrue(all(cells[y][0] == '#' == cells[y][50] for y in range(31)))

    def test_every_dataset_query_builds(self):
        from llm_astar.__main__ import load_queries
        for sample, query in load_queries(ROOT / 'data/environment_50_30.json'):
            prompt = build_prompt(query, 'vot', 'gpt', 'ascii_map')
            grid = prompt.split('Map:\n')[1].splitlines()
            self.assertEqual(sum(r.count('S') for r in grid), 1, sample)
            self.assertEqual(sum(r.count('G') for r in grid), 1, sample)

    def test_output_format_unchanged(self):
        self.assertIn('Generated Path: [[x1, y1], [x2, y2], ...]', build_prompt(QUERY, 'vot', 'gpt', 'ascii_map'))
        self.assertEqual(parse_path('...\nGenerated Path: [[1, 6], [16, 23]]', mode='upstream'), [(1, 6), (16, 23)])

    def test_coarse_reasoning_and_combination(self):
        plain = build_prompt(QUERY, 'vot', 'gpt')
        coarse = build_prompt(QUERY, 'vot', 'gpt', 'coarse_reasoning')
        self.assertEqual(coarse.replace(COARSE_REASONING + '\n', '', 1), plain)
        self.assertLess(coarse.index(COARSE_REASONING), coarse.index('Start Point:'))
        both = build_prompt(QUERY, 'vot', 'gpt', 'coarse_reasoning+waypoint_limit_by_obstacles')
        limit = build_prompt(QUERY, 'vot', 'gpt', 'waypoint_limit_by_obstacles')
        self.assertEqual(both.replace(COARSE_REASONING + '\n', '', 1), limit)
        mapped = build_prompt(QUERY, 'vot', 'gpt', 'waypoint_limit_by_obstacles+ascii_map')
        self.assertLess(mapped.index('Map:'), mapped.index('For this problem, use at most'))


if __name__ == '__main__':
    unittest.main()
