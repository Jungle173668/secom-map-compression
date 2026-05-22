# Initial Results — Configuration & Notes

Date: 2026-05-22
Dataset: LoCoMo (locomo10.json), 45 QA samples, 10 conversations (conv0–conv9)
LLM: gpt-4o-mini (via OpenAI API)
Judge: binary 0/1 (LLM-as-judge, strict exact correctness)

---

## Methods

### sliding_window
- Parameter: n_turns = 150 (keep last 150 turns)
- Expected token reduction: ~73%
- No LLM call for compression, pure truncation

### summarization
- Parameter: max_summary_tokens = 5000
- Design: one-shot — all old turns (history[:-10]) summarized in 1 LLM call, last 10 turns kept verbatim
- Model: gpt-4o-mini (SUMMARIZE_MODEL)
- Known issue: LLM generates ~1000 tokens regardless of the 5000 token budget → actual reduction ~93%
- Root cause: one-shot summarization of 500 turns causes LLM to over-compress; RecurSum (session-by-session) would be more faithful

### secom_map
- Entity Protection (EP): True
- Query Rewriting (QR): True
- Compress rate: 0.65 (LLMLingua-2)
- Segmentation: LLM-based topic segmentation (segment_history)
- context_map.update: called once with full history (known issue — should be incremental per session)
- Retrieval: BM25 (rank_bm25), top_k=3

---

## Results Summary

| Method         | Accuracy | Avg Token Reduction | Notes                        |
|----------------|----------|---------------------|------------------------------|
| sliding_window | 0.044    | 0.735               | 2/45 correct                 |
| summarization  | 0.089    | 0.932               | 4/45 correct; over-compressed|
| secom_map      | 0.311    | 0.656               | 14/45 correct; 1 error       |

By category:

| Category    | sliding_window | summarization | secom_map |
|-------------|---------------|---------------|-----------|
| multi_hop   | 1/15          | 1/15          | 9/15      |
| open_domain | 1/5           | 0/5           | 1/5       |
| single_hop  | 0/10          | 2/10          | 1/10      |
| temporal    | 0/15          | 1/15          | 3/15      |

---

## Known Issues / Planned Improvements

1. **summarization**: switch to RecurSum (session-by-session incremental update) so max_summary_tokens is respected
2. **secom_map context_map**: should call update() per session (D1→D25) not once with full history
3. **secom_map segmentation**: replace LLM-based segment_history() with session boundaries (dia_id) — saves 1 LLM call and improves segment quality
4. **1 error**: conv3_q30 threw `'int' object has no attribute 'strip'` — to be investigated

---

## Why Accuracy Is Low vs Paper

- Paper uses GPT4Score (0–100, partial credit); we use binary 0/1
- Paper uses GPT-4; we use gpt-4o-mini
- LoCoMo questions intentionally test early-conversation facts (~500 turns back)
- Relative ordering matches paper: secom_map >> summarization > sliding_window
