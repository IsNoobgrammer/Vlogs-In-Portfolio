# we designed the perfect solution. then we didn't use it.

2026-04-25

We tried Z-loss. It's what everyone does.

You have a Mixture-of-Experts model. Routing logits can explode — high-magnitude logits collapse diversity, every token routes to the same expert, your MoE becomes a very expensive MLP. So you add a Z-loss term: penalize large logit magnitudes. Problem (kind of) solved. Paper ships.

Nah.

---

## why Z-loss felt wrong

Z-loss is a blunt instrument. It adds gradient noise. Fixed penalty coefficient for a dynamic problem — your logit distribution at step 100 is not the same as at step 10,000, but the penalty doesn't know that. And the penalty coefficient is *another* hyperparameter you have to tune on top of the router temperature and the load-balancing aux loss weight and everything else.

When aadi shared a Google AI Studio config with Z-loss baked in, the response was direct: *"We don't want any Z-loss. It's better if we dynamically cap router logits before bias."*

So we designed dynamic clamping.

The idea: instead of a fixed penalty, compute the clamp bounds *from the logit distribution itself* — batch by batch, step by step. No tunable penalty. The bounds adapt to what the logits are actually doing.

```python
if self.use_dynamic_clamp:
    with torch.no_grad():
        logit_min = torch.min(router_logits)
        logit_max = torch.max(router_logits)
        logit_mean = torch.mean(router_logits)
    
    half_range = (logit_max - logit_min) / 2.0 * self.dynamic_clamp_scale
    clamp_min_val = logit_mean - half_range
    clamp_max_val = logit_mean + half_range
    
    router_logits = torch.clamp(router_logits, min=clamp_min_val.item(), max=clamp_max_val.item())
```

Clean. Elegant. Self-adapting. No hyperparameter for the bounds.

Then aloobun went further — quantile-based clamping, like a box plot. Clamp to the 25th percentile on the bottom, 85th percentile on the top. Outliers removed. Distribution stabilized. Batch-by-batch.

```python
if self.training and self.config.use_dynamic_logit_clamping:
    with torch.no_grad():
        lower_bound = torch.quantile(router_logits.detach(), 0.25)
        upper_bound = torch.quantile(router_logits.detach(), 0.85)
    router_logits = torch.clamp(router_logits, min=lower_bound, max=upper_bound)
```

This is the good version. Box plot logic applied to neural routing. Mathematically principled. Practically sound.

## and then we didn't implement it either

April 27, 2025. After all that design work:

> *"No clamping. At least not for now."*

lol.

The full arc: "Z-loss is bad" → "dynamic clamping is better" → "quantile clamping is even better" → "no clamping at all, let's just see what happens architecturally."

This sounds like giving up. It wasn't.

The honest answer was that we needed *actual training runs* to know if logit explosion was even our current problem. You don't add stability interventions before you observe instability. That's premature optimization with extra steps. The architecture hadn't been stressed yet. We didn't know if the logits would even diverge without clamping.

So we reverted to aux load-balancing loss — the boring standard approach — and kept clamping as a fallback waiting in the code, commented out, ready for the moment we actually see the problem.

```python
""" No Clamping for Now
TODO: @aloobun make clamp range dynamic based on mean/median/mode/std of current logits"""
# if self.logit_clamp_val > 0:
#     router_logits = torch.clamp(router_logits, min=-self.logit_clamp_val, max=self.logit_clamp_val)
```

That TODO is still there. The solution is still there. Just waiting.

## the thing about MoE routing nobody talks about

If you clamp all logits to the same value — which a very aggressive clamp might do — `topk` has no way to differentiate. All experts look equally good. Routing becomes random. You've solved logit explosion by creating routing collapse.

The fix: after clamping, add a small learnable bias before softmax. Break the symmetry. Let the model still express preference even in a compressed logit range.

*"Clamp ke baad small bias then softmax — else suppose it all clamped to same value, how to find most suitable top-k?"*

That's the kind of edge case that bites you at 2am when the routing distribution collapses and you can't tell if it's the clamp or the aux loss or just a bad batch.

## what we're actually doing

Aux loss. The boring answer that's been in every MoE paper since 2017. We're not using Z-loss (gradient noise, fixed coefficient). We're not using dynamic clamping (good idea, unvalidated on our specific setup). We're using a soft load-balancing penalty that we *know* works, on hardware we *know* the characteristics of, for an architecture we're still learning the failure modes of.

The philosophy: trust the architecture first. Intervene only with evidence.

Dynamic clamping is still the right idea. Just for a later version of us, with more training runs and a clearer picture of what's actually failing.

Sometimes the best engineering decision is the TODO comment in the code.
