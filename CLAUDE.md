### 1. Plan Mode Default
When the user starts a new optimization session or asks "what's next", enter Plan Mode first:
- Read `tasks/todo.md` and `tasks/lessons.md` for current state
- Read `kernel.py` (current best) and `reference.py` (target)
- Propose the next iteration strategy before writing any code

### 2. Subagent Strategy
Use subagents for parallel research when needed:
- **Explore agent**: Search KernelBench patterns, find similar solved problems
- **Plan agent**: Design kernel optimization strategy before coding
- Keep main context clean — delegate deep research to subagents

### 3. Iteration Loop (Stage 3+)
For each kernel optimization iteration:
1. Read current `kernel.py` and understand what it does
2. Plan the specific change (document in todo.md)
3. Edit `kernel.py` with the new approach
4. Run `uv run python prepare.py` to evaluate
5. Parse output for `status:` and `speedup:`
6. If correct AND faster → KEEP (git commit + push)
7. If incorrect OR slower → DISCARD (git restore kernel.py)
8. Update `tasks/todo.md` with result
9. Update `tasks/lessons.md` if new insight learned

### 4. Verification Before Commit
Before any git commit:
- Ensure `status:correct` from prepare.py
- Ensure speedup >= current best (monotonic improvement)
- Write clear commit message with speedup number

### 5. Key Constraints
- `prepare.py` and `reference.py` are READ-ONLY
- Must use CUDA C++ via `load_inline` (not Triton — see lessons.md)
- KernelBench times our model FIRST → warmup penalty is real
- Override `.to()` for GPU warmup, not `__init__`
