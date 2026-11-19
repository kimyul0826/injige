"""Opt-in input clarification; keep the released demonstrations and search unchanged."""
import ast
import re

MOVEMENT = '''Movement rules:
Coordinates are (x, y). From the current integer grid point (x, y), ALL EIGHT neighboring moves are allowed:
(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1),
(x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1).
Horizontal/vertical moves cost 1; diagonal moves cost sqrt(2). You are NOT restricted to horizontal and vertical movement.
Every move must avoid walls: the entire movement segment, including its endpoints, must not intersect or touch a wall. Diagonal moves cannot cross walls.
Wall coordinates below describe the endpoints of line segments, not their centers. Horizontal walls have constant y; vertical walls have constant x.
'''


def clarify_prompt(prompt, family, variant):
    if variant == 'upstream':
        return prompt
    if variant != 'eight_way_explicit_walls':
        raise ValueError(f'Unknown prompt_variant: {variant}')

    def walls(match):
        lines = ['Walls:']
        for orientation, values in [('horizontal', ast.literal_eval(match[1])),
                                    ('vertical', ast.literal_eval(match[2]))]:
            for fixed, lo, hi in values:
                a, b = ((lo, fixed), (hi, fixed)) if orientation == 'horizontal' else ((fixed, lo), (fixed, hi))
                axis = 'y' if orientation == 'horizontal' else 'x'
                lines.append(f'- W{len(lines)}: {orientation} wall at {axis} = {fixed}, from {a} to {b}')
        return '\n'.join(lines) if len(lines) > 1 else 'Walls: none'

    prompt = re.sub(r'Horizontal Barriers: (\[[^\n]*\])[ \t]*\nVertical Barriers: (\[[^\n]*\])', walls, prompt)
    prompt = prompt.replace('Horizontal barriers are represented as [y, x_start, x_end], and vertical barriers are represented as [x, y_start, y_end].', '')
    if family == 'llama':
        marker = '<|start_header_id|>system<|end_header_id|>'
        if marker not in prompt:
            raise ValueError('Llama prompt is missing the system header')
        return prompt.replace(marker, marker+'\n'+MOVEMENT, 1)
    return MOVEMENT+'\n'+prompt
