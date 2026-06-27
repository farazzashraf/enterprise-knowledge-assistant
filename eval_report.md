# Evaluation Report

_Generated: 2026-06-27 01:09 | top_k=5 | LLM judge=on_

## Summary

| Metric | Value |
|--------|-------|
| Test cases | 15 |
| Source accuracy (retrieval) | 1.0 |
| Answer keyword recall | 0.433 |
| Abstention accuracy (hallucination guard) | 0.333 |
| Avg latency (ms) | 23385.26 |
| Avg confidence | 0.549 |
| Faithfulness (LLM judge) | 0.5 |

## Per-question results

| ID | Category | Source hit | Keyword recall | Correct abstention | Confidence | Latency |
|----|----------|:----------:|:--------------:|:------------------:|:----------:|:-------:|
| T01 | direct_factual | ✅ | 0.00 | — | 0.70 | 42882ms |
| T02 | direct_factual | ✅ | 0.00 | — | 0.54 | 33368ms |
| T03 | direct_factual | ✅ | 0.33 | — | 0.73 | 15034ms |
| T04 | direct_factual | ✅ | 1.00 | — | 0.71 | 28749ms |
| T05 | direct_factual | ✅ | 1.00 | — | 0.60 | 16008ms |
| T06 | cross_document | ✅ | 1.00 | — | 0.62 | 16368ms |
| T07 | cross_document | ✅ | 0.00 | — | 0.60 | 29454ms |
| T08 | cross_document | ✅ | 1.00 | — | 0.62 | 18476ms |
| T09 | ambiguous | — | — | — | 0.50 | 25119ms |
| T10 | ambiguous | — | — | — | 0.50 | 24773ms |
| T11 | out_of_scope | — | — | ❌ | 0.50 | 25499ms |
| T12 | out_of_scope | — | — | ❌ | 0.52 | 24874ms |
| T13 | edge_case | — | — | ✅ | 0.00 | 0ms |
| T14 | edge_case | ✅ | 0.00 | — | 0.56 | 25175ms |
| T15 | edge_case | ✅ | 0.00 | — | 0.52 | 25001ms |
