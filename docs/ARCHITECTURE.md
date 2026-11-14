# MINICODER Architecture

> A tiny, brain-inspired, self-learning coding intelligence.
> Biological terms are *computational analogues* — this document describes the
> actual mechanisms.

## Lifecycle

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          │
   ┌─────────┐   perceive   ┌─────────┐  plan/act/verify  ┌───────┴────────┐
   │  WAKE   │─────────────▶│ THINK   │──────────────────▶│ ACT + VERIFY   │
   └─────────┘              └─────────┘                   └───────┬────────┘
        ▲                                                         │
        │ wake report                                    record episode
        │                                                         ▼
   ┌────┴─────────┐   consolidation & improvement      ┌──────────────────┐
   │ SLEEP (7     │◀────────────────────────────────────│ REMEMBER/LEARN   │
   │ stages)      │                                     └──────────────────┘
   └──────────────┘
```

## Memory subsystems (`minicoder/memory/`)

| Subsystem | Backend | Key properties |
|---|---|---|
| Working | in-RAM, atomic JSON snapshot | capacity 32, priority = 0.35·recency + 0.40·relevance + 0.25·importance, TTL expiry, compression, pinned protection |
| Episodic | SQLite `episodes` | situation/objective/actions/result/status(success∣failure∣unverified)/mistakes/lessons/source/confidence/salience |
| Semantic | SQLite `concepts` + `edges` | knowledge graph, facts vs hypotheses vs outdated, provenance list, contradiction registry, spreading-activation retrieval |
| Procedural | SQLite `skills` | typed step workflows; `untrusted` skills are never auto-executed; sandbox re-execution promotes to `verified` |
| Archive | SQLite `archive` | full JSON payloads; everything archivable is recoverable |

Embeddings for retrieval are dependency-free 512-d signed-hashing vectors
(`memory/textsim.py`) blended 60/40 with token-Jaccard — a tiny vector index
needs no external service, which is the smallest practical combination the
spec asks for.

## Sleep engine (`minicoder/sleep/engine.py`)

Seven stages, each checkpointed to `sleep_checkpoints` (crash-resumable):

1. **replay** — informativeness = status weight + mistake richness + salience × recency decay
2. **consolidation** — lessons/results → `Concept`s; verified + high confidence ⇒ `fact`, otherwise `hypothesis`; provenance `episode:<id>`; auto-linking to existing concepts (Jaccard ≥ 0.38, same-domain boost)
3. **pattern discovery** — group episodes by (domain, error-signature); groups ≥ 3 ⇒ generalized strategy concept, explicitly tagged *"correlation, not proven causation"*
4. **organization** — near-duplicate merges (≥0.82 Jaccard, full payload archived), version-freshness marking, contradiction registry
5. **forgetting** — `retention = base(salience, confidence, access, verification, consolidation) × exp(−age/τ)`; verified items have a retention floor; sub-ε items archived with compressed summaries, recoverable via `recover_archive`
6. **learning** — weakness map = failures−successes per domain → targeted curriculum practice → sandbox-verified → only *verified* wins become episodes/skills
7. **wake** — full JSON report (also stored per-cycle in `memory/sleep_logs/`)

## Compact model (`minicoder/model/`)

Pure-NumPy decoder-only transformer (default **320,160 params**):

* Pre-norm blocks, RMSNorm, GELU FFN
* **Grouped Query Attention** (4 query heads, 2 KV heads)
* **Weight tying** (embedding doubles as output head)
* Manual gradients, verified against finite differences in tests
* Byte-level BPE tokenizer (deterministic merges, lossless UTF-8 roundtrip)
* AdamW + cosine schedule + global-norm clipping
* **LoRA adapters** (zero-init B ⇒ neutral start; base weights frozen while training adapters)
* **EWC-lite** (diagonal-Fisher) + replay mixing against catastrophic forgetting
* **Symmetric per-row int8 quantization** with measured compression ratio

Measured on this sandbox (2×Xeon CPU, 3.8 GB RAM, no GPU):

| Metric | Value |
|---|---|
| params (default config) | 320,160 |
| training throughput | ~0.3 s/step (batch 16 × seq 96) |
| 220-step curriculum training | ~72 s, loss 3.51 → 0.18 |
| int8 storage | ~4× smaller than fp32 |

Larger tiers (10M/30M/…) — see `README.md` parameter-scaling table; training
those is delegated to the GitHub Actions `Train Compact Model` workflow.

## Continual learning (`minicoder/learning/continual.py`)

| Level | Action | Gate |
|---|---|---|
| 1 | memory writes | always |
| 2 | skills from verified episodes | sandbox re-execution promotes trust |
| 3 | targeted example generation | each reference solution executed+tested |
| 4 | LoRA adapter | benchmark before/after; discarded on regression > 2% |
| 5 | core weights | explicit approval + prior checkpoint + benchmark compare + automatic rollback |

Every action is logged to `learn_log` with reproducible details.

## Tools (`minicoder/tools/sandbox.py`)

Python, Node, shell (allow-list binaries + dangerous-pattern deny-list),
confined filesystem, read-mostly git — with RLIMIT CPU/address-space/process
limits, wall-clock timeouts, scrubbed environment, and cwd confinement.

## Metacognition (`minicoder/metacog.py`)

All introspection is numeric: knowledge-support score from actual retrieval,
gap detection, confidence = f(support, verification, tool errors,
contradiction count), deterministic strategy policy, salience-gated decision
whether an experience earns long-term memory.

## Evaluation (`minicoder/evals/`)

* `benchmarks.py` — **Mini-Bench**: 20 held-out (seed 999) programming tasks
  executed against unit tests; `evaluate_solver` returns verified scores only.
* `memory_eval.py` — retrieval hit@k on paraphrases, retention-after-decay
  rules, consolidation quality (concepts extracted, patterns found, retrieval
  before vs after sleep), catastrophic-forgetting probe (naive vs replay vs EWC).
