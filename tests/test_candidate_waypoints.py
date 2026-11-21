"""Candidate waypoints: only the final query gets the candidate list; candidates are free interior cells."""
import unittest
from pathlib import Path

from llm_astar.__main__ import load_queries
from llm_astar.models import build_prompt
from llm_astar.prompt_variants import candidate_waypoints

DATA = Path(__file__).resolve().parents[1] / 'data' / 'environment_50_30.json'
Q0 = dict(range_x=[0, 51], range_y=[0, 31], start=[5, 5], goal=[20, 20],
          horizontal_barriers=[[10, 0, 25], [15, 30, 50]], vertical_barriers=[[25, 10, 22]])


def queries():
    return (q for _, q in load_queries(DATA))


class CandidateWaypointTests(unittest.TestCase):
    def test_example_map(self):
        self.assertEqual(candidate_waypoints(Q0), [[24, 9], [24, 11], [24, 21], [24, 23], [26, 9], [26, 11],
                                                   [26, 21], [26, 23], [29, 14], [29, 16], [31, 14], [31, 16]])

    def test_only_adds_after_final_query(self):
        for family in ('llama', 'gpt'):
            for style in ('standard', 'cot', 'repe'):
                with self.subTest(family=family, style=style):
                    original = build_prompt(Q0, style, family)
                    modified = build_prompt(Q0, style, family, 'candidate_waypoints')
                    added = ('\nCandidate Waypoints: ' + str(candidate_waypoints(Q0)) +
                             '\nFor this problem, choose every intermediate waypoint of the final generated path '
                             'from the candidate waypoints, excluding the start and goal points.')
                    self.assertEqual(modified.count(added), 1)
                    self.assertEqual(modified.replace(added, '', 1), original)
                    self.assertGreater(modified.index(added), modified.rfind('Start Point: [5, 5]'))

    def test_dataset_candidates_free_and_interior(self):
        for q in queries():
            (x0, x1), (y0, y1) = q['range_x'], q['range_y']
            for x, y in candidate_waypoints(q):
                self.assertTrue(x0 < x < x1 - 1 and y0 < y < y1 - 1)
                for fy, a, b in q['horizontal_barriers']:
                    self.assertFalse(y == fy and min(a, b) <= x <= max(a, b))
                for fx, a, b in q['vertical_barriers']:
                    self.assertFalse(x == fx and min(a, b) <= y <= max(a, b))
            build_prompt(q, 'standard', 'gpt', 'candidate_waypoints')


if __name__ == '__main__':
    unittest.main()
