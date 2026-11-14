# External training-data plan (for connected machines)

The sandbox used to build this project has no access to dataset hosts, so the
shipped corpus is 100% synthetic (`minicoder/curriculum.py`, provenance
`synthetic-curriculum`) — and we make **no claims** of having trained on
external data.

On a machine with internet access, the following sources are the recommended
legal corpus, with the license/provenance handling the project expects:

| Source | What it gives | License notes |
|---|---|---|
| **The Stack v2** (BigCode) | 900B+ tokens of code in 600+ languages | SWH archive; filter to permissive licenses via `licenses` metadata; requires ToS acceptance |
| **StarCoderData** (BigCode) | Cleaned training set of StarCoder2 | same governance; use the `jupyter`/`github` splits as needed |
| **CodeSearchNet** | 6 languages, function-level, docstring pairs | per-repo licenses; mostly permissive subsets are clearly tagged |
| **CommitPackFT** | filtered commits, instruction-like diffs | permissive-license filtered |
| **Official docs** (Python, MDN, K8s…) | documentation QA | check each site's license; prefer CC-BY or similar |
| **Synthetic/self-generated** tasks + reference solutions verified by execution | debugging, testing, refactoring drills | unencumbered; verification provides ground truth |

## Pipeline (to implement where network is available)

```
fetch (manifest of URLs + snapshot IDs)
  → license filter (allow-list: MIT/BSD/Apache/CC-BY/Unlicense…)
  → PII/secrets scan (gitleaks-style rules)
  → near-dedup (MinHash/Suffix-array)
  → quality heuristics (avg line length, % comments, test coverage presence)
  → provenance record per sample (source url, license, snapshot, hash)
  → jsonl shards → tokenizer streams
```

Rules of engagement encoded in `curriculum.py` practice flows still apply:

* Every external sample stores provenance; nothing provenance-free enters
  semantic memory as a `fact`.
* Benchmarks (HumanEval/MBPP/SWE-bench) are **evaluation-only** — training on
  them would invalidate the comparison; when executing them on a connected
  machine, import them in `minicoder/evals/benchmarks.py` behind the same
  `evaluate_solver` API.
