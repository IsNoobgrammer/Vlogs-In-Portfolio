# we designed the perfect solution. then we didn't use it.

2026-04-25

First, the architecture. Because none of the routing decisions make sense without knowing what we were routing *for*.

BiBo's MoE config, finalized April 24, 2025:

```
1 shared conv expert  (always active, kernel=3 on gate_proj)
9 routable MLP experts  (top-k=2)
1 identity/residual expert  (the cheapest possible: near pass-through)
~4B total params, ~1.5–2B active
```

The shared expert fires for every token regardless of routing. The 9 MLP experts specialize. The identity expert gives the router a "no-op" option for tokens that don't need any transformation. And the whole thing is designed as a drop-in replacement for the FFN block of any transformer decoder — *"just plug into any decoder layer and it should work."*

With that context: routing stability matters. A lot. If your router logits explode, every token routes to the same expert, your MoE becomes a very expensive dense MLP, and you've wasted all the parameter efficiency you designed for.

So. Z-loss.

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

There was also a question about whether the *router itself* should be a convolution. The argument:

> *"Suppose 'book then book lover and book hater'. Standard MLP router looks only at the current token. A conv router looks at 'I love books' and routes to the appropriate expert based on local context."*

A conv router sees a 3-token window. So it doesn't just ask "what is this token?" — it asks "what is this token, given what just came before?" That's a genuinely different routing signal. The memory overhead is `kernel_size × params`, but since the router is small (hidden_dim → num_experts), the cost is acceptable.

We didn't implement the conv router yet either. *"linear vs conv routing — interesting study."* — aloobun.

That's two ideas sitting in TODO comments now.

## what we're actually doing

Aux loss. Switch-transformer formulation. The boring answer that's been in every MoE paper since 2017:

```python
aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
if self.training and self.aux_loss_coef > 0:
    with torch.no_grad():
        mask = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        load = mask.sum(dim=0) / num_tokens
        prob_mean = routing_weights.sum(dim=0) / num_tokens
        aux_loss = (load * prob_mean).sum() * self.num_experts * self.aux_loss_coef
```

`load` is actual expert usage. `prob_mean` is ideal usage. Minimize their correlation → uniform load. The `num_experts` multiplier keeps the aux loss magnitude consistent regardless of expert count. Aloobun implemented this. It's clean. It works.

We're not using Z-loss (gradient noise, fixed coefficient). We're not using dynamic clamping (good idea, unvalidated on our current setup). We're using a penalty we *know* works, on hardware we *know* the characteristics of, for an architecture we're still discovering the failure modes of.

One more thing we added: Dynamic Tanh (DyT) replacing RMSNorm everywhere — post-attention, Q-norm, K-norm, post-FFN. RMSNorm operates over the whole sequence; DyT operates per token individually. Faster. Streaming-compatible. Less communication overhead in distributed settings. Another quiet decision that the architecture will never announce but will show up in throughput.

The philosophy: trust the architecture first. Intervene only with evidence.

Dynamic clamping is still the right idea. Just for a later version of us, with more training runs and a clearer picture of what's actually failing.

Sometimes the best engineering decision is the TODO comment in the code.
