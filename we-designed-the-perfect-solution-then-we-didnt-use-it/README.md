# we designed the perfect solution. then we didn't use it.

2026-04-25

first , the architecture. because none of the routing decisions make sense without knowing what we were routing *for*.

BiBo's MoE config , finalized April 24 , 2025:

```
1 shared conv expert  (always active , kernel=3 on gate_proj)
9 routable MLP experts  (top-k=2)
1 identity/residual expert  (near pass-through)
~4B total params , ~1.5–2B active
```

shared expert fires for every token. 9 MLP experts specialize. identity expert gives router a "no-op" option. whole thing is drop-in replacement for FFN block — *"just plug into any decoder layer and it should work."*

routing stability matters. a lot. router logits explode → every token routes to same expert → MoE becomes expensive dense MLP → wasted all parameter efficiency.

so. Z-loss.

nah.

---

## why Z-loss felt wrong

blunt instrument. adds gradient noise. fixed penalty for dynamic problem — logit distribution at step 100 is not same as step 10,000. penalty doesn't know that. and it's *another* hyperparameter on top of router temperature and aux loss weight.

when aadi shared a Google AI Studio config with Z-loss baked in , the response was: *"We don't want any Z-loss. better to dynamically cap router logits before bias."*

so we designed dynamic clamping.

idea: compute clamp bounds *from logit distribution itself* — batch by batch. no tunable penalty. bounds adapt.

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

clean. self-adapting. no hyperparameter.

then aloobun went further — quantile-based clamping. like a box plot. 25th percentile bottom , 85th top. outliers removed.

```python
if self.training and self.config.use_dynamic_logit_clamping:
    with torch.no_grad():
        lower_bound = torch.quantile(router_logits.detach(), 0.25)
        upper_bound = torch.quantile(router_logits.detach(), 0.85)
    router_logits = torch.clamp(router_logits, min=lower_bound, max=upper_bound)
```

box plot logic on neural routing. this is the good version.

## and then we didn't implement it

April 27. after all that design work:

> *"No clamping. At least not for now."*

lol.

full arc: "Z-loss is bad" → "dynamic clamping is better" → "quantile clamping is even better" → "no clamping at all , let's see what happens."

sounds like giving up. wasn't.

honest answer: needed *actual training runs* to know if logit explosion was even our problem. don't add stability interventions before you observe instability. premature optimization with extra steps.

reverted to aux load-balancing loss — boring standard approach. clamping sitting in code , commented out , waiting.

```python
""" No Clamping for Now
TODO: @aloobun make clamp range dynamic based on mean/median/mode/std of current logits"""
# if self.logit_clamp_val > 0:
#     router_logits = torch.clamp(router_logits, min=-self.logit_clamp_val, max=self.logit_clamp_val)
```

that TODO is still there. solution still there. just waiting.

## the edge case nobody talks about

clamp all logits to same value → `topk` can't differentiate → all experts look equally good → routing becomes random → solved logit explosion by creating routing collapse.

fix: after clamping , add small learnable bias before softmax. break symmetry.

*"Clamp ke baad small bias then softmax — else suppose it all clamped to same value , how to find most suitable top-k?"*

kind of edge case that bites you at 2am.

also: should the *router itself* be a convolution?

> *"Suppose 'book then book lover and book hater'. Standard MLP router looks only at current token. conv router looks at 'I love books' and routes to appropriate expert based on local context."*

conv router sees 3-token window. "what is this token , given what just came before?" genuinely different routing signal.

didn't implement that either. *"linear vs conv routing — interesting study."* — aloobun.

two ideas in TODO comments now.

## what we're actually doing

aux loss. switch-transformer formulation. boring answer from every MoE paper since 2017:

```python
aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
if self.training and self.aux_loss_coef > 0:
    with torch.no_grad():
        mask = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        load = mask.sum(dim=0) / num_tokens
        prob_mean = routing_weights.sum(dim=0) / num_tokens
        aux_loss = (load * prob_mean).sum() * self.num_experts * self.aux_loss_coef
```

`load` is actual usage. `prob_mean` is ideal. minimize correlation → uniform load. aloobun implemented this. clean. works.

not using Z-loss. not using dynamic clamping. using penalty we *know* works , on hardware we *know* , for architecture we're still discovering.

one more thing: Dynamic Tanh (DyT) replacing RMSNorm everywhere — post-attention , Q-norm , K-norm , post-FFN. RMSNorm operates over whole sequence. DyT per token. faster. streaming-compatible. quiet decision that shows up in throughput.

philosophy: trust architecture first. intervene only with evidence.

dynamic clamping is still the right idea. just for later version of us.

sometimes the best engineering decision is the TODO comment in the code.
