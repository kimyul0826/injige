# Sources and reproduction boundaries

Paper: Meng et al. (2024), LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning.
https://arxiv.org/abs/2407.02511

Official repository: https://github.com/SilinMeng0510/llm-astar
Pinned commit: 9414ed894d774f71c36e8c0afd0e20e6033b95b1 (MIT).
`upstream/` is an optional local reference checkout, excluded from the source package. Runtime code does not require it.
`llm_astar/vendor/prompts.py` is copied verbatim from `llmastar/pather/llm_a_star/prompt.py`.
`data/environment_50_30.json` is copied verbatim from the official dataset.
The applicable MIT license is preserved in `llm_astar/vendor/LICENSE` and `upstream/LICENSE`.
Search is reimplemented from Algorithm 1 and the released Python code, with attribution here.

Important differences:
- Paper §4.2 says GPT-3.5-TURBO; released path-generation code calls GPT-3.5-TURBO-INSTRUCT.
  Requested gpt35.json uses chat GPT-3.5 Turbo. Optional gpt35_upstream.json preserves the
  completion endpoint and original newline stop setting. These are distinct experiments.
- Released Llama is Meta-Llama-3-8B-Instruct, bfloat16, greedy decoding, max_new_tokens=8000.
  Local config preserves those settings but removes hard-coded cuda:3 and uses the configured context policy (`upstream`: 8000 new tokens regardless of prompt length; `cap`: remaining context). Server mode uses chat messages extracted from the raw template.
- Default `upstream` search preserves duplicate heap pops, blocked-neighbor storage accounting,
  target switching before CLOSED filtering, extra waypoint margin and half-open obstacle filtering.
  Current segment intersection uses Shapely 2.2.0, including released degenerate-line behavior.
- Both search modes avoid input mutation and report unreachable paths instead of raising KeyError.
  Start/goal must be valid interior integer points. Upper range bounds are exclusive, as in the
  released dataset: [0,51] has boundary walls at 0 and 50.
- `paper` search skips duplicate CLOSED entries, tests CLOSED before target updates,
  and filters waypoints by actual point collision. Its counts differ from released implementation.
- Prompts retain original text and examples, including their limitations (no map bounds in
  final path-generation question). Changing prompts is an experimental change.
- Model output parsing uses literal_eval with finite numeric validation, never eval.
  Parse failure is recorded. LLM-only fails; LLM-A* uses [start,goal] targets (not plain A*:
  the target-to-goal distance term still exists). No fabricated LLM output or hidden A* oracle.
- LLM-only is evaluated as the original raw piecewise-linear path: endpoints, bounds, and
  every segment are checked. Endpoint repair/filtering applies only to LLM-A*.
- Ratios are geometric means over valid method/baseline pairs; validity includes all selected
  samples. Invalid/zero-length pairs are excluded from path-length ratios and the paired count
  is reported. Empty means are null. Operation/storage are search counts, not LLM latency,
  GPU memory, wall-clock time, or token cost. Timing/usage are recorded separately.
- This implementation does not claim exact reproduction of reported Table 1 numbers.
  Model versions, service behavior and these explicit implementation choices affect results.
