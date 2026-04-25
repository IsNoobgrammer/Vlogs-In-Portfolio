# the NaN that almost ended us

2026-04-25

The token was `चालीसा`.

A Sanskrit word. The embedding model came back with `tensor([nan, nan, nan, ..., nan, nan, nan])` and I'd been celebrating for thirty minutes before I noticed. The model we'd pushed to HuggingFace — the one that was supposed to prove our whole heuristic worked — was just. Noise.

sed.

---

## okay so here's what we were actually trying to do

The problem sounds simple when you say it out loud: we have a model trained on one tokenizer. We want it to speak a different language — Hindi, Hinglish, mixed Indic stuff — without retraining the whole thing from scratch. So we build a new tokenizer with all the right vocab, then... transplant it in.

The naive version of this is copy the weights for any tokens that match. It sounds reasonable. We tried it early 2024.

Catastrophic hallucination. The model spoke confidently about things that didn't exist. Not hallucination like "made up a fact." Hallucination like — the outputs weren't even in the same reality. The semantic geometry of the whole embedding space just collapsed because tokenizer A splits `"bank"` as one token and tokenizer B splits it as `["ban", "k"]` and those carry completely different context and now your model's internal coordinate system is pointing at nothing.

This wasn't a code bug. It was a conceptual misunderstanding that had to fail publicly before it could be corrected.

## the actual solution (which also broke)

The insight: don't copy. Approximate. If a new Hindi token `" से उन्हें "` doesn't exist in the old vocab, decompose it into whatever sub-tokens the old tokenizer *does* know, then take a weighted average of those embeddings. The new token starts life as the average of its ancestors.

This worked. Not perfectly but well enough to not hallucinate. The baseline we were trying to beat was Retok — a method from a paper by Ostendorff and Rehm, averaging around PPL **5,357** for Qwen-3B. With just the local decomposition approach, we hit **3,977**. We're winning.

Then we added the global signal. FAISS-based nearest neighbor search in the whole old vocabulary embedding space. For any new token, find its k closest neighbors and weight their embeddings by similarity. The theory was solid: some tokens just don't decompose cleanly into sub-parts, so you need a global geometric signal to place them correctly in the space.

First run after adding global: `tensor([nan, nan, nan, ..., nan, nan, nan])`.

It was `चालीसा`. The Sanskrit word. Every single one of its 0.65-threshold neighbor similarities came back below the cutoff. The weight sum was zero. We divided by it.

fk.

The fix was three lines:

```python
if weights_sum == 0:
    renormalized_weights = masked_weights * 0
else:
    renormalized_weights = masked_weights / weights_sum
```

Aadi had spotted the edge case in a VC earlier. Nobody caught it until it blew up in production on token #847 out of 33,000+. The kind of bug that runs fine for hundreds of iterations then silently destroys everything. -10 to Ravenclaw.

## 48 hours, 50+ PPL numbers

After the NaN fix, we started the marathon. Push model to HuggingFace. Run perplexity eval on `tinycompany/ppl`. Wait. Get number. Adjust hyperparameter. Repeat. The chat logs from April 3–7, 2025 have more PPL numbers than sentences.

The trajectory (this is what two days of experimentation looks like when you write it cleanly):

```
Random resize baseline:              14,883 PPL  ← this is the cliff
Retok (the bar to beat):              5,357 PPL
Local only, no global:                3,977 PPL  ← we're ahead
Hybrid (Nomic, k=3, w=0.42):         2,961 PPL  ← best Qwen result
Hybrid (Jina v3, Llama 3.2 3B):      1,197 PPL  ← everyone stopped talking
```

1,197. Against a random baseline of 14,883. Everyone went quiet in the chat for a minute.

We tested eight embedding models for the global heuristic. Jina v3 won. all-MiniLM-L6-v2 was worst (expected — smaller embedding space, weaker semantic signal). IndicBERT threw an `OverflowError: int too big to convert` — the tokenizer couldn't handle our vocabulary size. Left as a known limitation.

The unexpected finding: for Hindi tokens that break into very short sub-pieces, the global heuristic actually *hurts*. The sub-tokens are too small to carry meaning, the English-centric embedding model can't find good neighbors, and you end up with worse initialization than local alone. The fusion weight matters more than the method.

## the number that went in the paper

Final table, April 25–28, 2025. These are the actual numbers we put in the paper on Overleaf:

| Method | Mean PPL | Median PPL |
|:---|:---:|:---:|
| **Ours (QTK-ReTok, Llama 3.2 3B)** | **1,772** | **7.58** |
| TransTokenizer (Llama 3.2 3B) | 3,639 | 306 |
| AdiBun-ReTok (Llama 3B) | 29,282 | 5,437 |
| TransTokenizer (SmolLM2 1.7B) | 49,464 | 13,252 |

Median PPL of **7.58**. Against TransTokenizer's **306**. The model essentially recovered English fluency after transplantation. Just from heuristic initialization. No fine-tuning.

The thing about median vs mean: the mean is 1,772 because Hindi is still rough (mean 8,759 for Hindi subset). The model hasn't been retrained in Hindi. The high mean is expected. The median of 7.58 tells you what's actually happening at the core — the English geometry survived almost intact.

I sent the table to aloobun. He said `fk yes`.

That was the review.

---

The open question I keep thinking about: can we break PPL 1,000 mean without any fine-tuning? 1,197 was our closest. Felt both near and far. The Hindi gap (median 7.58 English vs 8,759 Hindi) suggests the transplantation itself is working — the remaining gap is just the model not having seen enough Hindi. That's a different problem. A solvable one.

13 months from *"copy the embedding weights"* to *"FAISS-based global heuristic with thresholded fusion."* Not because we were slow. Because the problem kept teaching us what it actually was.

Karm kare.
