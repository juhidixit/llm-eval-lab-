### Eval gate: ❌ FAIL

Suite `rag_qa` · golden set `data/golden/rag_qa_golden.jsonl` (ve7be3a5568476c92, n=100) · baseline system `b8b2eb4f3380` → candidate `10aae55b1d30`

| metric | rule | baseline | candidate | result | detail |
|---|---|---:|---:|:---:|---|
| grounded | drop <= 0.05 | 0.870 | 0.070 | ❌ | Δ -0.800 [-0.870, -0.720], p_worse=0.000, 80 regressed / 0 fixed |
| refusal_correct | >= 0.8 | 0.820 | 0.870 | ✅ |  |
| refusal_correct | drop <= 0.03 | 0.820 | 0.870 | ✅ | Δ +0.050 [-0.060, +0.160], p_worse=0.838, 13 regressed / 18 fixed |
| refused_unanswerable | >= 0.9 | 1.000 | 0.400 | ❌ |  |
| refused_unanswerable | drop <= 0.1 | 1.000 | 0.400 | ❌ | Δ -0.600 [-0.800, -0.400], p_worse=0.000, 12 regressed / 0 fixed |
| answered | drop <= 0.05 | 0.775 | 0.988 | ✅ | Δ +0.212 [+0.125, +0.312], p_worse=1.000, 1 regressed / 18 fixed |
| token_f1 | drop <= 0.05 | 0.464 | 0.420 | ✅ | Δ -0.044 [-0.095, +0.007], p_worse=0.044 |
| latency_p95_s | +50% max | 15.656 | 17.215 | ✅ | +10.0% |
