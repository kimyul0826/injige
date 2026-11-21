"""Opt-in input clarification; keep the released demonstrations and search unchanged."""
import ast
import math
import re

MOVEMENT = 'You can move horizontally, vertically, and diagonally.\nBarriers cannot be crossed.'
VOT_TRIGGER = 'Visualize the state after each reasoning step.'
FINAL_INSTRUCTION = 'Conclude your response with the generated path'
COARSE_REASONING = ('Reason at the level of barriers and turning points, not individual grid cells.\n'
                    'Each waypoint must be a turning point of the path; do not list consecutive grid cells.')
MAP_LEGEND = ('A map of the final problem is also drawn as a character grid: "#" is a barrier or the map boundary, '
              '"S" is the start, "G" is the goal, "." is free space. Rows are printed from the largest y at the top '
              'to the smallest y at the bottom; the row label is y and the column label is x.')


REPE2_INSTRUCTION = ('Find the path in rounds. Round k uses exactly k intermediate waypoints, starting from Round 0 '
                     '(a straight line from the start to the goal). After each round, check whether every segment '
                     'avoids all barriers, and stop at the first valid round.')
# Demonstrations: five problems on FRESH random 51x31 maps (not in the evaluation set) whose minimal waypoint
# counts are 0, 2, 1, 3, 1, each solved with the shortest chain of that size. Built by exp/claude/build_repe2_demos.py
# (picked for low LLM-A* operation ratio among dataset-like problems) and rendered by render_repe2_demos.py;
# every round and Check verdict is verified against the exact segment-intersection test (tests/test_repe2_prompt.py).
REPE2_DEMOS = """Start Point: [5, 20]
Goal Point: [45, 3]
Horizontal Barriers: [[10, 16, 23], [24, 6, 27]]
Vertical Barriers: [[3, 18, 21], [15, 23, 30], [35, 1, 4]]
Round 0: [[5, 20], [45, 3]]
Check: Every segment avoids all barriers. Valid.
Generated Path: [[5, 20], [45, 3]]

Start Point: [2, 21]
Goal Point: [47, 12]
Horizontal Barriers: [[5, 22, 30], [27, 45, 48], [22, 0, 3]]
Vertical Barriers: [[32, 0, 18], [24, 15, 30]]
Round 0: [[2, 21], [47, 12]]
Check: The segment from [2, 21] to [47, 12] crosses the vertical barrier at x=32 spanning y=0 to y=18 and the vertical barrier at x=24 spanning y=15 to y=30. Not valid.
Round 1: [[2, 21], [22, 14], [47, 12]]
Check: The segment from [22, 14] to [47, 12] crosses the vertical barrier at x=32 spanning y=0 to y=18. Not valid.
Round 2: [[2, 21], [22, 14], [33, 19], [47, 12]]
Check: Every segment avoids all barriers. Valid.
Generated Path: [[2, 21], [22, 14], [33, 19], [47, 12]]

Start Point: [2, 5]
Goal Point: [39, 19]
Horizontal Barriers: [[23, 0, 12], [13, 11, 22], [25, 42, 50], [26, 0, 14]]
Vertical Barriers: [[32, 6, 21], [16, 20, 30]]
Round 0: [[2, 5], [39, 19]]
Check: The segment from [2, 5] to [39, 19] crosses the vertical barrier at x=32 spanning y=6 to y=21. Not valid.
Round 1: [[2, 5], [23, 24], [39, 19]]
Check: Every segment avoids all barriers. Valid.
Generated Path: [[2, 5], [23, 24], [39, 19]]

Start Point: [13, 27]
Goal Point: [43, 7]
Horizontal Barriers: [[12, 39, 50]]
Vertical Barriers: [[19, 23, 30], [45, 0, 15], [7, 0, 12], [37, 0, 16]]
Round 0: [[13, 27], [43, 7]]
Check: The segment from [13, 27] to [43, 7] crosses the vertical barrier at x=19 spanning y=23 to y=30 and the vertical barrier at x=37 spanning y=0 to y=16. Not valid.
Round 1: [[13, 27], [20, 22], [43, 7]]
Check: The segment from [20, 22] to [43, 7] crosses the vertical barrier at x=37 spanning y=0 to y=16. Not valid.
Round 2: [[13, 27], [20, 22], [37, 17], [43, 7]]
Check: The segment from [37, 17] to [43, 7] crosses the horizontal barrier at y=12 spanning x=39 to x=50. Not valid.
Round 3: [[13, 27], [20, 22], [37, 17], [38, 13], [43, 7]]
Check: Every segment avoids all barriers. Valid.
Generated Path: [[13, 27], [20, 22], [37, 17], [38, 13], [43, 7]]

Start Point: [6, 7]
Goal Point: [39, 27]
Horizontal Barriers: [[22, 20, 37], [20, 0, 24], [25, 47, 50], [4, 5, 8]]
Vertical Barriers: [[28, 0, 10]]
Round 0: [[6, 7], [39, 27]]
Check: The segment from [6, 7] to [39, 27] crosses the horizontal barrier at y=22 spanning x=20 to x=37. Not valid.
Round 1: [[6, 7], [37, 21], [39, 27]]
Check: Every segment avoids all barriers. Valid.
Generated Path: [[6, 7], [37, 21], [39, 27]]
"""


# Ablation E1-b: the five RELEASED standard demonstrations (same maps, start, goal), re-solved round by round.
# Waypoints are taken from the released answers; every verdict is checked in tests/test_repe2_prompt.py.
REPE2_ORIG_DEMOS = """Start Point: [5, 5]
Goal Point: [20, 20]
Horizontal Barriers: [[10, 0, 25], [15, 30, 50]]
Vertical Barriers: [[25, 10, 22]]
Round 0: [[5, 5], [20, 20]]
Check: The segment from [5, 5] to [20, 20] crosses the horizontal barrier at y=10 spanning x=0 to x=25. Not valid.
Round 1: [[5, 5], [26, 9], [20, 20]]
Check: [5, 5] to [26, 9] is clear, but [26, 9] to [20, 20] crosses the vertical barrier at x=25 spanning y=10 to y=22. Not valid.
Round 2: [[5, 5], [26, 9], [25, 23], [20, 20]]
Check: [5, 5] to [26, 9] passes below the end of the horizontal barrier at y=10, [26, 9] to [25, 23] stays right of the vertical barrier at x=25, and [25, 23] to [20, 20] passes above its end. Valid.
Generated Path: [[5, 5], [26, 9], [25, 23], [20, 20]]

Start Point: [2, 3]
Goal Point: [18, 8]
Horizontal Barriers: [[7, 0, 10], [5, 30, 40]]
Vertical Barriers: [[14, 12, 30], [40, 0, 11]]
Round 0: [[2, 3], [18, 8]]
Check: The segment from [2, 3] to [18, 8] reaches y=7 at x=14.8, beyond the end of the horizontal barrier at y=7 spanning x=0 to x=10, and touches no other barrier. Valid.
Generated Path: [[2, 3], [18, 8]]

Start Point: [40, 20]
Goal Point: [10, 10]
Horizontal Barriers: [[15, 20, 30], [15, 35, 50]]
Vertical Barriers: [[20, 4, 30]]
Round 0: [[40, 20], [10, 10]]
Check: The segment from [40, 20] to [10, 10] crosses the vertical barrier at x=20 spanning y=4 to y=30. Not valid.
Round 1: [[40, 20], [20, 3], [10, 10]]
Check: [40, 20] to [20, 3] passes y=15 at x=34.1, through the gap between the horizontal barriers at y=15, and passes below the vertical barrier at x=20; [20, 3] to [10, 10] is clear. Valid.
Generated Path: [[40, 20], [20, 3], [10, 10]]

Start Point: [4, 4]
Goal Point: [16, 16]
Horizontal Barriers: [[20, 0, 20], [5, 30, 40]]
Vertical Barriers: [[40, 0, 15], [20, 10, 21]]
Round 0: [[4, 4], [16, 16]]
Check: The segment from [4, 4] to [16, 16] stays below y=20 and left of x=20, so it touches no barrier. Valid.
Generated Path: [[4, 4], [16, 16]]

Start Point: [25, 25]
Goal Point: [4, 4]
Horizontal Barriers: [[7, 0, 20], [15, 10, 15]]
Vertical Barriers: [[10, 15, 30], [30, 0, 18], [10, 0, 5]]
Round 0: [[25, 25], [4, 4]]
Check: The segment from [25, 25] to [4, 4] crosses the horizontal barrier at y=15 spanning x=10 to x=15 and the horizontal barrier at y=7 spanning x=0 to x=20. Not valid.
Round 1: [[25, 25], [21, 7], [4, 4]]
Check: [25, 25] to [21, 7] passes y=15 at x=22.8, right of the barrier at y=15; [21, 7] to [4, 4] passes just beyond the end of the barrier at y=7 and above the vertical barrier at x=10 spanning y=0 to y=5. Valid.
Generated Path: [[25, 25], [21, 7], [4, 4]]
"""


def strip_rounds(demos):
    """Standard (answer-only) form of round demonstrations: drop the Round/Check lines."""
    return '\n'.join(l for l in demos.split('\n') if not l.startswith(('Round ', 'Check:')))


def llama_turns(demos):
    """Demonstrations in the released Llama-3 chat layout: problem as a user turn, answer as an assistant turn."""
    out = []
    for block in demos.strip().split('\n\n'):
        lines = block.split('\n')
        out.append('<|start_header_id|>user<|end_header_id|>\n' + '\n'.join(lines[:4]) + '\n<|eot_id|>\n'
                   '<|start_header_id|>assistant<|end_header_id|>\n' + '\n'.join(lines[4:]) + '\n<|eot_id|>\n')
    return ''.join(out)


def repe2_template(family, variant='full'):
    """RePE2 and its E1 ablations.
    full      : round instruction + new demonstrations solved in rounds + final query without a path cue
    noround   : released standard instruction + the same new demonstrations, answer only (E1-a)
    origdemos : round instruction + the released demonstrations re-solved in rounds (E1-b)
    The Llama-3 version uses the released Llama chat layout (system / user / assistant turns) with the same
    instruction sentence and demonstrations; the final assistant turn is left open, as in the released prompt."""
    from .vendor.prompts import gpt_prompt, llama_prompt
    demos = {'full': REPE2_DEMOS, 'noround': strip_rounds(REPE2_DEMOS), 'origdemos': REPE2_ORIG_DEMOS}[variant]
    if family == 'llama':
        standard = llama_prompt['standard']
        head = standard[:standard.index('<|start_header_id|>user')]
        if variant != 'noround':
            head = head.replace(FINAL_INSTRUCTION, REPE2_INSTRUCTION + '\n' + FINAL_INSTRUCTION, 1)
        tail = standard[standard.rindex('<|start_header_id|>user'):]
        return head + llama_turns(demos) + tail
    if family != 'gpt':
        raise ValueError(f'Unknown prompt family: {family}')
    standard = gpt_prompt['standard']
    head = standard[:standard.index('Start Point: [5, 5]')]
    if variant == 'noround':
        return head + demos + '\n' + standard[standard.rindex('Start Point: {start}'):]
    head = head.replace(FINAL_INSTRUCTION, REPE2_INSTRUCTION + '\n' + FINAL_INSTRUCTION, 1)
    cot = gpt_prompt['cot']                      # final query without a trailing "Generated Path:" cue,
    tail = cot[cot.rindex('Start Point: {start}'):]   # so the model starts with "Round 0"
    return head + demos + '\n' + tail


def vot_template(cot_template, family):
    """Zero-shot VoT (Wu et al. 2024): the released CoT instructions and final query, no demonstrations."""
    if family == 'llama':
        head = cot_template[:cot_template.index('<|eot_id|>')]
        tail = cot_template[cot_template.rindex('<|start_header_id|>user<|end_header_id|>'):]
        template = head + '<|eot_id|>\n' + tail
    else:
        head = cot_template[:cot_template.index('\n\n') + 2]
        tail = cot_template[cot_template.rindex('Start Point: {start}'):]
        template = head + tail
    if template.count(FINAL_INSTRUCTION) != 1:
        raise ValueError('Expected one final-path instruction in the CoT prompt')
    return template.replace(FINAL_INSTRUCTION, VOT_TRIGGER + '\n' + FINAL_INSTRUCTION, 1)


def ascii_map(query):
    """Text grid of the integer cells in [min, exclusive_max); barriers include both endpoints."""
    (x0, x1), (y0, y1) = query['range_x'], query['range_y']
    width, height = x1 - x0, y1 - y0
    cell = [['.'] * width for _ in range(height)]

    def mark(x, y, char):
        if x0 <= x < x1 and y0 <= y < y1:
            cell[y - y0][x - x0] = char

    for x in range(x0, x1):
        mark(x, y0, '#'); mark(x, y1 - 1, '#')
    for y in range(y0, y1):
        mark(x0, y, '#'); mark(x1 - 1, y, '#')
    for key in ('horizontal_barriers', 'vertical_barriers'):
        for fixed, lo, hi in query.get(key, []):
            lo, hi = sorted((lo, hi))
            for t in range(math.ceil(lo), math.floor(hi) + 1):
                if float(fixed).is_integer():
                    mark(*((t, int(fixed)) if key.startswith('horizontal') else (int(fixed), t)), '#')
    mark(*query['start'], 'S')
    mark(*query['goal'], 'G')
    label = max(len(str(y0)), len(str(y1 - 1)))
    pad = ' ' * (label + 1)
    tens = pad + ''.join(str(x // 10 % 10) if x % 10 == 0 else ' ' for x in range(x0, x1))
    ones = pad + ''.join(str(x % 10) for x in range(x0, x1))
    rows = [f'{y:>{label}} ' + ''.join(cell[y - y0]) for y in range(y1 - 1, y0 - 1, -1)]
    return '\n'.join([tens, ones, *rows, ones])


def add_ascii_map(prompt, query):
    if prompt.count(FINAL_INSTRUCTION) < 1:
        raise ValueError('Final-path instruction missing from prompt')
    first = prompt.index(FINAL_INSTRUCTION)
    prompt = prompt[:first] + MAP_LEGEND + '\n' + prompt[first:]
    marker = 'Vertical Barriers: ' + str(query.get('vertical_barriers', []))
    position = prompt.rfind(marker)
    if position < 0:
        raise ValueError('Final query barrier list missing from prompt')
    position += len(marker)
    return prompt[:position] + '\nMap:\n' + ascii_map(query) + prompt[position:]


def candidate_waypoints(query):
    """Free cells diagonally next to every barrier end that is not on the map border, sorted and deduplicated.

    Barriers are closed axis-aligned segments; a cell is free if it lies strictly inside the map boundary and
    on no barrier. Pure geometry from the query, so it runs unchanged on unseen maps.
    """
    (x0, x1), (y0, y1) = query['range_x'], query['range_y']
    segments = []
    for fixed, lo, hi in query.get('horizontal_barriers', []):
        segments.append(((min(lo, hi), fixed), (max(lo, hi), fixed)))
    for fixed, lo, hi in query.get('vertical_barriers', []):
        segments.append(((fixed, min(lo, hi)), (fixed, max(lo, hi))))

    def on_barrier(c):
        return any(a[0] <= c[0] <= b[0] and a[1] <= c[1] <= b[1] for a, b in segments)

    def on_border(p):
        return p[0] <= x0 or p[0] >= x1 - 1 or p[1] <= y0 or p[1] >= y1 - 1

    out = set()
    for a, b in segments:
        for end in (a, b):
            if on_border(end):
                continue
            for dx in (-1, 1):
                for dy in (-1, 1):
                    c = (end[0] + dx, end[1] + dy)
                    if x0 + 1 <= c[0] <= x1 - 2 and y0 + 1 <= c[1] <= y1 - 2 and not on_barrier(c):
                        out.add(c)
    return [list(c) for c in sorted(out)]


def barrier_endpoints(query):
    """Cell just past each end of every barrier, along the barrier direction, in query order.

    Horizontal [y, x1, x2] gives [x1 - 1, y] and [x2 + 1, y]; vertical [x, y1, y2] gives [x, y1 - 1] and
    [x, y2 + 1]. Cells the official LLM-A* waypoint filter would drop (on a barrier, or outside
    range_x[0]+1 < x < range_x[1]-2, range_y[0]+1 < y < range_y[1]-2) are left out; duplicates removed.
    """
    (x0, x1), (y0, y1) = query['range_x'], query['range_y']
    segments, out = [], []
    for fixed, lo, hi in query.get('horizontal_barriers', []):
        lo, hi = sorted((lo, hi))
        segments.append(((lo, fixed), (hi, fixed)))
        out += [[lo - 1, fixed], [hi + 1, fixed]]
    for fixed, lo, hi in query.get('vertical_barriers', []):
        lo, hi = sorted((lo, hi))
        segments.append(((fixed, lo), (fixed, hi)))
        out += [[fixed, lo - 1], [fixed, hi + 1]]

    def keep(c):
        return (x0 + 1 < c[0] < x1 - 2 and y0 + 1 < c[1] < y1 - 2 and
                not any(a[0] <= c[0] <= b[0] and a[1] <= c[1] <= b[1] for a, b in segments))
    return [c for i, c in enumerate(out) if keep(c) and c not in out[:i]]


DEMO_RANGE = dict(range_x=[0, 51], range_y=[0, 31])   # demonstrations use the 50x30 dataset grid
# Two released demonstration waypoints sit one cell off a barrier-end cell; moved onto it (both stay collision-free).
DEMO_ENDPOINT_FIXES = (('[26, 9]', '[26, 10]'), ('[11, 6]', '[11, 7]'))


def endpoint_demos(prompt):
    """Add a 'Barrier Endpoints' line to every demonstration and move the two off-candidate demo waypoints."""
    final = prompt.rfind('Start Point:')
    demos, tail = prompt[:final], prompt[final:]
    for old, new in DEMO_ENDPOINT_FIXES:
        if old not in demos:
            raise ValueError(f'Demonstration waypoint {old} missing from prompt')
        demos = demos.replace(old, new)

    def add(match):
        query = dict(DEMO_RANGE, horizontal_barriers=ast.literal_eval(match[1]),
                     vertical_barriers=ast.literal_eval(match[2]))
        return match[0] + '\nBarrier Endpoints: ' + str(barrier_endpoints(query))

    demos, count = re.subn(r'Horizontal Barriers: (\[[^\n]*\])[ \t]*\nVertical Barriers: (\[[^\n]*?\])(?=[ \t]*\n)',
                           add, demos)
    if count == 0:
        raise ValueError('No demonstrations found')
    return demos + tail


def clarify_prompt(prompt, family, variant, query=None):
    if '+' in variant:
        # Combined variants are applied left to right, e.g. "coarse_reasoning+waypoint_limit_by_obstacles".
        for part in variant.split('+'):
            prompt = clarify_prompt(prompt, family, part, query=query)
        return prompt
    if variant == 'upstream':
        return prompt
    if variant == 'coarse_reasoning':
        if prompt.count(FINAL_INSTRUCTION) < 1:
            raise ValueError('Final-path instruction missing from prompt')
        first = prompt.index(FINAL_INSTRUCTION)
        return prompt[:first] + COARSE_REASONING + '\n' + prompt[first:]
    if variant == 'ascii_map':
        if query is None:
            raise ValueError('ASCII map requires the final query')
        return add_ascii_map(prompt, query)
    if variant == 'vot':
        marker = 'Conclude your response with the generated path'
        if prompt.count(marker) != 1:
            raise ValueError('Expected one final-path instruction in the original prompt')
        return prompt.replace(marker, 'Visualize the state after each reasoning step.\n' + marker, 1)
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
    if variant == 'candidate_waypoints':
        if query is None:
            raise ValueError('Candidate waypoints require the final query')
        marker = 'Vertical Barriers: ' + str(query.get('vertical_barriers', []))
        position = prompt.rfind(marker)
        if position < 0:
            raise ValueError('Final query barrier list missing from prompt')
        position += len(marker)
        condition = ('\nCandidate Waypoints: ' + str(candidate_waypoints(query)) +
                     '\nFor this problem, choose every intermediate waypoint of the final generated path '
                     'from the candidate waypoints, excluding the start and goal points.')
        return prompt[:position] + condition + prompt[position:]
    if variant == 'endpoint_demos':
        return endpoint_demos(prompt)
    if variant == 'barrier_endpoints':
        if query is None:
            raise ValueError('Barrier endpoints require the final query')
        marker = 'Vertical Barriers: ' + str(query.get('vertical_barriers', []))
        position = prompt.rfind(marker)
        if position < 0:
            raise ValueError('Final query barrier list missing from prompt')
        position += len(marker)
        condition = ('\nBarrier Endpoints: ' + str(barrier_endpoints(query)) +
                     '\nFor this problem, choose every intermediate waypoint of the final generated path '
                     'from the barrier endpoints, excluding the start and goal points. '
                     'You do not need to use every barrier endpoint; use only the ones the path needs.')
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
