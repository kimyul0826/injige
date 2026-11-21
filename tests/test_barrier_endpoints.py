"""Barrier endpoints: prompt lists the cell just past each barrier end; collisions are handled after the LLM."""
import sys
import unittest
from pathlib import Path

from llm_astar.__main__ import load_queries
from llm_astar.models import build_prompt
from llm_astar.prompt_variants import barrier_endpoints
from llm_astar.search import Environment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp' / 'claude'))
import endpoint_snap as es  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'data' / 'environment_50_30.json'
Q0 = dict(range_x=[0, 51], range_y=[0, 31], start=[5, 5], goal=[20, 20],
          horizontal_barriers=[[10, 0, 25], [15, 30, 50]], vertical_barriers=[[25, 10, 22]])


class BarrierEndpointTests(unittest.TestCase):
    def test_cells_just_past_each_end(self):
        # [0,10] and [50,15] would be on the map border; [26,10] is past the east end of y=10
        self.assertEqual(barrier_endpoints(Q0), [[26, 10], [29, 15], [25, 9], [25, 23]])

    def test_dataset_endpoints_free_and_interior(self):
        for _, q in load_queries(DATA):
            (x0, x1), (y0, y1) = q['range_x'], q['range_y']
            for x, y in barrier_endpoints(q):
                self.assertTrue(x0 < x < x1 - 1 and y0 < y < y1 - 1)
                for fy, a, b in q['horizontal_barriers']:
                    self.assertFalse(y == fy and min(a, b) <= x <= max(a, b))
                for fx, a, b in q['vertical_barriers']:
                    self.assertFalse(x == fx and min(a, b) <= y <= max(a, b))

    def test_only_adds_after_final_query(self):
        added = ('\nBarrier Endpoints: ' + str(barrier_endpoints(Q0)) +
                 '\nFor this problem, choose every intermediate waypoint of the final generated path '
                 'from the barrier endpoints, excluding the start and goal points. '
                 'You do not need to use every barrier endpoint; use only the ones the path needs.')
        for family in ('llama', 'gpt'):
            for style in ('standard', 'cot', 'repe'):
                with self.subTest(family=family, style=style):
                    original = build_prompt(Q0, style, family)
                    modified = build_prompt(Q0, style, family, 'barrier_endpoints')
                    self.assertEqual(modified.count(added), 1)
                    self.assertEqual(modified.replace(added, '', 1), original)
                    self.assertGreater(modified.index(added), modified.rfind('Start Point: [5, 5]'))

    def test_official_filter_keeps_every_endpoint(self):
        for _, q in load_queries(DATA):
            env = Environment(q)
            ends = [tuple(e) for e in barrier_endpoints(q)]
            kept = env.targets([tuple(q['start'])] + ends + [tuple(q['goal'])], 'upstream')
            self.assertEqual(kept[1:-1], ends)

    def test_endpoint_demos_only_add_lines_and_move_two_waypoints(self):
        import re
        for family in ('llama', 'gpt'):
            for style in ('standard', 'cot', 'repe'):
                with self.subTest(family=family, style=style):
                    base = build_prompt(Q0, style, family, 'barrier_endpoints')
                    demo = build_prompt(Q0, style, family, 'endpoint_demos+barrier_endpoints')
                    final = base.rfind('Start Point:')
                    self.assertEqual(demo[demo.rfind('Start Point:'):], base[final:])
                    stripped = re.sub(r'\nBarrier Endpoints: \[[^\n]*\]', '', demo[:demo.rfind('Start Point:')])
                    expected = base[:final].replace('[26, 9]', '[26, 10]').replace('[11, 6]', '[11, 7]')
                    self.assertEqual(stripped, expected)
                    self.assertEqual(demo.count('Barrier Endpoints:'), base.count('Start Point:'))
                    self.assertIn('Barrier Endpoints: [[26, 10], [29, 15], [25, 9], [25, 23]]', demo)

    def test_dataset_prompts_build(self):
        for _, q in load_queries(DATA):
            build_prompt(q, 'standard', 'gpt', 'barrier_endpoints')
            build_prompt(q, 'repe', 'gpt', 'endpoint_demos+barrier_endpoints')

    def test_snap_keeps_collision_free_chain(self):
        env = Environment(Q0, collision_backend='shapely')
        T = es.snap_chain(env, Q0, [(5, 5), (25, 9), (25, 23), (20, 20)])
        self.assertEqual(T, [(5, 5), (26, 10), (25, 23), (20, 20)])
        self.assertFalse(any(env.collision(a, b) for a, b in zip(T, T[1:])))

    def test_snap_without_valid_chain_drops_waypoints(self):
        env = Environment(Q0, collision_backend='shapely')
        self.assertEqual(es.snap_chain(env, Q0, [(29, 15)]), [(5, 5), (20, 20)])


if __name__ == '__main__':
    unittest.main()
