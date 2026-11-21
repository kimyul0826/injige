"""The cap is a prompt instruction, not path truncation or a search restriction."""
import unittest
from llm_astar.models import build_prompt


class WaypointLimitTests(unittest.TestCase):
    def test_only_final_query_gets_numeric_cap(self):
        for count in (0, 1, 4):
            q = dict(start=[5,5], goal=[20,20], horizontal_barriers=[[10,0,25]]*count,
                     vertical_barriers=[])
            sentence = (f'\nFor this problem, use at most {count} intermediate waypoints '
                        'in the final generated path, excluding the start and goal points.')
            for family in ('llama', 'gpt'):
                for style in ('standard', 'cot', 'repe', 'vot'):
                    with self.subTest(count=count, family=family, style=style):
                        original = build_prompt(q, style, family)
                        modified = build_prompt(q, style, family, 'waypoint_limit_by_obstacles')
                        self.assertEqual(modified.count(sentence), 1)
                        self.assertEqual(modified.replace(sentence, '', 1), original)
                        final_query = modified.rfind('Start Point: [5, 5]')
                        self.assertGreater(modified.index(sentence), final_query)
