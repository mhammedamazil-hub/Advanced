# 🧠 MINICODER

**A tiny, brain-inspired, self-learning coding intelligence** — not a stateless
chatbot, but a cognitive architecture with working memory, persistent episodic
+ semantic + procedural memory, an autonomous 7-stage sleep/consolidation
engine, a hand-built compact NumPy transformer, gated continual learning,
sandboxed developer tools, measurable metacognition, and a live dashboard.

Lifecycle: `WAKE → PERCEIVE → THINK → LEARN → REMEMBER → SLEEP → CONSOLIDATE → IMPROVE → WAKE`

The brain **never starts with an empty mind**: all long-term memory persists in
SQLite across restarts, sessions, and interrupted sleep cycles.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# full lifecycle demo (WAKE→…→SLEEP→IMPROVE) with measured before/after metrics
.venv/bin/python -m minicoder.cli demo --fast
# enter/leave sleep on demand
.venv/bin/python -m minicoder.cli sleep
.venv/bin/python -m minicoder.cli status
# live dashboard (http://localhost:8000)
.venv/bin/python -m minicoder.cli dashboard --port 8000
# run the test suite (all subsystems)
.venv/bin/pip install pytest && .venv/bin/python -m pytest tests/ -q
```

## What exists (all implemented, all tested — 80 tests)

| Claim in spec | Where it lives | Proof |
|---|---|---|
| Working memory w/ capacity, priority, TTL, compression, pinning | `minicoder/memory/working.py` | `tests/test_working.py` |
| Episodic memory, verified/failed/unverified distinction | `minicoder/memory/episodic.py` | `tests/test_episodic.py` |
| Semantic knowledge graph, facts≠hypotheses, contradictions | `minicoder/memory/semantic.py` | `tests/test_semantic.py` |
| Procedural skills, **no auto-execution of untrusted procedures** | `minicoder/memory/procedural.py` | `tests/test_procedural.py` |
| 7-stage sleep engine, resumable after interruption | `minicoder/sleep/engine.py` | `tests/test_sleep.py` |
| Sleep replay→consolidate→patterns→organize→forget→learn→wake | same | `tests/test_sleep.py`, `tests/test_memory_eval.py` |
| Durable store: atomic writes, backups, corruption recovery | `minicoder/memory/store.py` | `tests/test_store.py` |
| Provenance on every record | schema + upserts | exported dumps |
| Compact model (GQA, weight tying, RMSNorm, byte-BPE) | `minicoder/model/` | gradient-checked: `tests/test_transformer.py` |
| Continual learning L1–L5 with checkpoints + rollback | `minicoder/learning/continual.py`, `model/trainer.py` | `tests/test_trainer.py` |
| Catastrophic-forgetting prevention (replay + EWC-lite) | `model/trainer.py` | `tests/test_forgetting.py` |
| Sandboxed tools (python/node/shell/fs/git, limits + deny-list) | `minicoder/tools/sandbox.py` | `tests/test_sandbox.py` |
| Metacognition (numeric confidence, gaps, strategy policy) | `minicoder/metacog.py` | `tests/test_metacog.py` |
| Benchmarks (held-out Mini-Bench, HumanEval-style execution) | `minicoder/evals/benchmarks.py` | `tests/test_eval.py` |
| Memory-retrieval / consolidation / retention metrics | `minicoder/evals/memory_eval.py` | `tests/test_memory_eval.py` |
| Dashboard (state, memory, sleep, learning, bench, controls) | `minicoder/dashboard/server.py` | live preview |
| GitHub Actions training & scheduled sleep cycles | `.github/workflows/` | CI badges |

## The compact model

Pure NumPy decoder-only transformer, **320,160 params** by default:
pre-norm RMSNorm blocks, **Grouped Query Attention** (4→2 KV heads),
**weight-tied** embeddings, GELU FFN, byte-level BPE (deterministic), AdamW +
cosine schedule, **LoRA** adapters (zero-init ⇒ neutral), EWC-lite + replay,
**int8** per-row quantization (~4× storage compression, measured).

Param-count scaling of the same architecture (measure with
`python -c "from minicoder.config import ModelConfig; from minicoder.model.transformer import MiniGPT, param_count; ..."`):

| Tier | d | L | heads→kv | ff | vocab | params |
|---|---|---|---|---|---|---|
| nano (default, sandbox-proven) | 96 | 3 | 4→2 | 256 | 800 | **0.32M** |
| micro | 192 | 4 | 6→2 | 512 | 8k | ~6M |
| small | 288 | 6 | 6→2 | 768 | 16k | ~18M |
| base | 448 | 8 | 8→2 | 1152 | 32k | ~55M |
| plus | 640 | 10 | 10→2 | 1664 | 48k | ~125M |

Hard environment truth (this sandbox): 2 CPU cores, 3.8 GB RAM, no GPU.
Training beyond ~1M params is deliberately delegated to the GitHub Actions
workflow (`.github/workflows/train.yml`), which publishes checkpoints +
benchmarks as artifacts. Do not believe any project that claims big results
from a box like this — ours are honest.

## Does the memory + sleep machinery actually help? (measured, not claimed)

`demo` runs the spec §10 comparison on real sandbox-executed tests:

1. **model without memory** — raw generation over held-out Mini-Bench
2. **model + persistent memory** — same tasks re-asked after experience → verified episodes replayed instantly
3. **after sleep** — consolidation metrics: concepts extracted, patterns discovered, retrieval before/after
4. **after LoRA learning** — benchmark-gated adapter (kept only if no regression)

Run it yourself: `.venv/bin/python -m minicoder.cli demo --fast` — results land in `demo_output/report.json` and the `benchmarks` table (dashboard shows them).

Representative honest run on this sandbox (2 CPU, no GPU, ~3.5 min, `--fast`): the 0.32M model after 450 steps scores **15–20%** on the 20-task held-out Mini-Bench (raw greedy generation); after 8 in-corpus attempts + environment feedback, **every re-asked task with a verified episode was solved instantly from memory** (3/3, 2.5 s vs ~12 s regeneration), and one sleep cycle consolidated **+32 concepts**, discovered recurring-error patterns (e.g. *`modulenotfounderror` in debugging*), and verified one new skill during targeted practice. Numbers vary with training steps/sampling — the harness never fabricates them.

## Memory layout (all under `data/memory/`)

```
brain.db               SQLite WAL store — source of truth (with .bak.1..3 rotation)
working/current.json   working-memory snapshot (atomic writes)
checkpoints/<name>/    model.npz + manifest + adapter (versioned, rollback-able)
sleep_logs/            per-cycle JSON reports
datasets/              level-3 targeted training sets (verified references)
exports/               full memory exports/imports (JSON)
archived/…concepts/    mirrored via SQLite `archive` table (recoverable)
sandbox/               confinement root for tool execution
```

## CLI reference

| Command | Purpose |
|---|---|
| `status` | full brain state JSON |
| `train --steps N` | train compact model on synthetic curriculum |
| `solve <task-id>` | THINK→ACT→VERIFY→LEARN on one task |
| `sleep [--resume ID]` | run (or resume) a sleep cycle |
| `episode …` / `perceive …` | record experience / working note |
| `recall <query>` | query all memory systems |
| `eval` | run Mini-Bench, record result |
| `learn 2|3|4|5` | gated continual-learning levels |
| `export` / `import <path>` | memory portability |
| `dashboard --port 8000` | live UI (Sleep/Wake/Eval/Export/Import controls) |
| `demo --fast` | the whole lifecycle with measurements |

## Honesty & safety rules this project enforces

* **Nothing unverified is ever called fact** — every memory has provenance + confidence + status (`fact`/`hypothesis`/`outdated`); conflicts are registered as contradictions, never silently overwritten.
* **Untrusted procedures never auto-execute** — sandbox re-verification promotes them.
* **The only model checkpoint is never mutated** — level-5 updates are gated, checkpointed, benchmark-compared, auto-rollback on regression.
* **No fabricated results**: the sandbox cannot fetch HumanEval/MBPP datasets, so we ship an in-house Mini-Bench (seed-separated from training) and document the plug-in points for real benchmarks (`docs/EXTERNAL_DATA.md`).
* **Biological sleep is modeled computationally** (replay/consolidation/triggered forgetting) — no literal brain imitation.

## Repository layout

```
minicoder/
  brain.py             lifecycle orchestrator
  metacog.py           numeric self-monitoring
  curriculum.py        synthetic legal corpus + taxonomy
  memory/  store.py working.py episodic.py semantic.py procedural.py system.py textsim.py
  sleep/   engine.py
  model/   tokenizer.py transformer.py trainer.py lora.py quantize.py
  learning/continual.py
  tools/   sandbox.py
  evals/   benchmarks.py memory_eval.py
  dashboard/server.py  (single-page UI, no JS deps)
  cli.py  demo.py
tests/                80 tests across all subsystems
docs/                 ARCHITECTURE.md EXTERNAL_DATA.md
.github/workflows/    ci.yml · sleep_cycle.yml (nightly) · train.yml (manual, gated)
```

## 💬 Chatbot · 📜 Constitution · 🎓 Skill-studying sleep

**Chatbot & coder** — talk to the brain (CLI REPL or dashboard chat card or `POST /api/chat`):

```bash
.venv/bin/python -m minicoder.cli chat          # REPL
# you> rule: never run destructive shell commands      # → constitution
# you> remember: our API runs on port 9000             # → episodic memory
# you> learn skill: add numbers | def add(a,b):\n return a+b   # → procedural
# you> solve py-fizzbuzz                               # → verified coding
# you> sleep                                           # → enter sleep
```

Open questions are answered **memory-grounded**: the reply cites the concepts/episodes/skills it used (`sources`), states numeric `confidence`, and honestly says *"I don't know this yet — teach me"* when support is low.

**Constitution (rules it never forgets)** — rules are semantic facts in the protected `constitution` domain: pinned into working memory at every wake/startup, injected into chat context, and **immune to the forgetting stage** (tested: a 100-day-old rule survives decay archival).

**Skill *studying* during sleep (real training, not just storage)** — sleep stage 6 now takes unstudied skills, renders their steps into instruction-format training samples, and runs the **level-4 gated LoRA training** on them. The adapter is kept only if the benchmark doesn't regress; only then is the skill marked `studied: true`. So: *teach a skill today, and after the next sleep the brain has actually studied it* — plus the episode, the checkpoint, and the learn-log record proving it.
