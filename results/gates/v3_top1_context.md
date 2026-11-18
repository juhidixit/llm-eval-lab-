### Eval gate: ❌ FAIL

Suite `rag_qa` · golden set `data/golden/rag_qa_golden.jsonl` (ve7be3a5568476c92, n=100) · baseline system `b8b2eb4f3380` → candidate `8642b406c5a8`

| metric | rule | baseline | candidate | result | detail |
|---|---|---:|---:|:---:|---|
| grounded | drop <= 0.05 | 0.870 | 0.810 | ✅ | Δ -0.060 [-0.150, +0.030], p_worse=0.105, 13 regressed / 7 fixed |
| refusal_correct | >= 0.8 | 0.820 | 0.770 | ❌ |  |
| refusal_correct | drop <= 0.03 | 0.820 | 0.770 | ✅ | Δ -0.050 [-0.130, +0.030], p_worse=0.149, 12 regressed / 7 fixed |
| refused_unanswerable | >= 0.9 | 1.000 | 1.000 | ✅ |  |
| refused_unanswerable | drop <= 0.1 | 1.000 | 1.000 | ✅ | Δ +0.000 [+0.000, +0.000], p_worse=1.000, 0 regressed / 0 fixed |
| answered | drop <= 0.05 | 0.775 | 0.713 | ✅ | Δ -0.062 [-0.175, +0.037], p_worse=0.142, 12 regressed / 7 fixed |
| token_f1 | drop <= 0.05 | 0.464 | 0.374 | ❌ | Δ -0.089 [-0.145, -0.033], p_worse=0.001 |
| latency_p95_s | +50% max | 15.656 | 13.992 | ✅ | -10.6% |
