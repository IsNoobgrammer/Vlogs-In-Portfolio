# the NaN that almost ended us

2026-04-25

the token was `चालीसा`.

Sanskrit word. the embedding model came back with `tensor([nan, nan, nan, ..., nan, nan, nan])` and i'd been celebrating for thirty minutes before i noticed. the model we'd pushed to HuggingFace — the one that was supposed to prove our whole heuristic worked — was just noise.

sed.

---

## okay so here's what actually happened

sounds simple when you say it out loud. model trained on one tokenizer. we want it to speak Hindi , Hinglish , mixed Indic stuff — without retraining from scratch. so we build a new tokenizer with the right vocab and transplant it in.

naive version: copy weights for matching tokens. sounds reasonable. tried it early 2024.

catastrophic hallucination.

not "made up a fact" hallucination. the outputs weren't even in the same reality. tokenizer A splits `"bank"` as one token. tokenizer B splits it as `["ban", "k"]`. completely different context. semantic geometry collapsed. model's internal coordinate system pointing at nothing.

not a code bug. conceptual misunderstanding that had to fail publicly before we could fix it.

## the solution (which also broke)

insight: don't copy. approximate.

new Hindi token `" से उन्हें "` doesn't exist in old vocab? decompose into sub-tokens the old tokenizer knows. weighted average of those embeddings. new token starts life as the average of its ancestors.

worked. not perfectly. but well enough to not hallucinate.

baseline to beat: Retok — Ostendorff and Rehm paper. PPL **5,357** for Qwen-3B. our local decomposition: **3,977**. we're winning.

then we added global signal. FAISS nearest neighbor search in old vocabulary embedding space. for new token , find k closest neighbors , weight by similarity. theory was solid: some tokens don't decompose cleanly. need global geometric signal.

first run after adding global:

`tensor([nan, nan, nan, ..., nan, nan, nan])`

it was `चालीसा`. Sanskrit word. every single 0.65-threshold neighbor similarity came back below cutoff. weight sum was zero. we divided by it.

fk.

fix was three lines:

```python
if weights_sum == 0:
    renormalized_weights = masked_weights * 0
else:
    renormalized_weights = masked_weights / weights_sum
```

aadi spotted the edge case in a VC earlier. nobody caught it until token #847 out of 33,000+. kind of bug that runs fine for hundreds of iterations then silently destroys everything.

-10 to Ravenclaw.

## 48 hours , 50+ PPL numbers

after the NaN fix , marathon started. push model to HuggingFace. run ppl eval on `tinycompany/ppl`. wait. get number. adjust hyperparameter. repeat. chat logs from April 3–7 have more PPL numbers than sentences.

trajectory:

```
Random resize baseline:              14,883 PPL  ← this is the cliff
Retok (the bar to beat):              5,357 PPL
Local only , no global:               3,977 PPL  ← we're ahead
Hybrid (Nomic , k=3 , w=0.42):       2,961 PPL  ← best Qwen result
Hybrid (Jina v3 , Llama 3.2 3B):     1,197 PPL  ← everyone stopped talking
```

1,197. against 14,883 baseline. everyone went quiet for a minute.

tested eight embedding models. Jina v3 won. all-MiniLM-L6-v2 worst (expected — smaller embedding space). IndicBERT threw `OverflowError: int too big to convert` — tokenizer couldn't handle our vocab size. left as known limitation.

unexpected finding: for Hindi tokens that break into very short sub-pieces , global heuristic actually *hurts*. sub-tokens too small to carry meaning. English-centric embedding model can't find good neighbors. worse initialization than local alone. fusion weight matters more than the method.

## the number that went in the paper

final table , April 25–28 , 2025:

| Method | Mean PPL | Median PPL |
|:---|:---:|:---:|
| **Ours (QTK-ReTok , Llama 3.2 3B)** | **1,772** | **7.58** |
| TransTokenizer (Llama 3.2 3B) | 3,639 | 306 |
| AdiBun-ReTok (Llama 3B) | 29,282 | 5,437 |
| TransTokenizer (SmolLM2 1.7B) | 49,464 | 13,252 |

median PPL **7.58**. against TransTokenizer's **306**. model recovered English fluency after transplantation. just from heuristic initialization. no fine-tuning.

median vs mean: mean is 1,772 because Hindi still rough (mean 8,759 for Hindi subset). model hasn't been retrained in Hindi. high mean expected. median 7.58 tells you what's actually happening — English geometry survived almost intact.

sent the table to aloobun. he said `fk yes`.

that was the review.

---

open question: can we break PPL 1,000 mean without fine-tuning? 1,197 was closest. felt both near and far. Hindi gap (median 7.58 English vs 8,759 Hindi) suggests transplantation is working — remaining gap is just model not seeing enough Hindi. different problem. solvable.

13 months from *"copy the embedding weights"* to *"FAISS-based global heuristic with thresholded fusion."* not because we were slow. because the problem kept teaching us what it actually was.

karm kare.
