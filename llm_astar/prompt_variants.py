"""Opt-in input clarification; keep the released demonstrations and search unchanged."""
import ast
import re

MOVEMENT = 'You can move horizontally, vertically, and diagonally.\nBarriers cannot be crossed.'


def clarify_prompt(prompt, family, variant, query=None):
    if variant == 'upstream':
        return prompt
    if variant == 'waypoint_limit_by_obstacles':
        if query is None:
            raise ValueError('Waypoint limit requires the final query')
        limit = len(query.get('horizontal_barriers', [])) + len(query.get('vertical_barriers', []))
        marker = 'Vertical Barriers: ' + str(query.get('vertical_barriers', []))
        position = prompt.rfind(marker)
        if position < 0:
            raise ValueError('Final query barrier list missing from prompt')
        position += len(marker)
        condition = (f'\nFor this problem, use at most {limit} intermediate waypoints '
                     'in the final generated path, excluding the start and goal points.')
        return prompt[:position] + condition + prompt[position:]
    if variant == 'eight_way_only':
        description = 'Horizontal barriers are represented as [y, x_start, x_end], and vertical barriers are represented as [x, y_start, y_end].'
        if description not in prompt:
            raise ValueError('Original barrier description missing from prompt')
        return prompt.replace(description, description + '\nYou can move horizontally, vertically, and diagonally.', 1)
    if variant != 'eight_way_explicit_walls':
        raise ValueError(f'Unknown prompt_variant: {variant}')

    def barriers(match):
        lines = ['Barriers:']
        for orientation, values in [('horizontal', ast.literal_eval(match[1])),
                                    ('vertical', ast.literal_eval(match[2]))]:
            for fixed, lo, hi in values:
                a, b = ((lo, fixed), (hi, fixed)) if orientation == 'horizontal' else ((fixed, lo), (fixed, hi))
                axis = 'y' if orientation == 'horizontal' else 'x'
                lines.append(f'- B{len(lines)}: {orientation} barrier at {axis} = {fixed}, from {a} to {b}')
        return '\n'.join(lines) if len(lines) > 1 else 'Barriers: none'

    prompt = re.sub(r'Horizontal Barriers: (\[[^\n]*\])[ \t]*\nVertical Barriers: (\[[^\n]*\])', barriers, prompt)
    return prompt.replace(
        'Horizontal barriers are represented as [y, x_start, x_end], and vertical barriers are represented as [x, y_start, y_end].',
        MOVEMENT, 1)
