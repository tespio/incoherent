# incoherent

An independent replication & falsification study of [Cohere's "concerning behaviour from a popular
open-weight AI model"](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model)
(Qwen3.8-Flash-Next, Oct 2026) and of the evidence released with it.

**Read the study: https://tespio.github.io/incoherent/**

Five tests, 138+ forced completions, zero recurring "call-home" endpoints.
Trace authentic (token-verified); repro shown to be generic base-model autocomplete
(IBM granite: 90% URL completion on the identical cut point); the Aliyun "flavor" shown
to be lineage data, absent in 32/32 local Qwen3.8-27B completions; the target model itself
blocked by Alibaba's upstream moderation when replayed through a commercial provider.

All raw outputs are published unmodified in `data/`. Every test in `scripts/` is runnable —
see the site's *Reproduce* section. Sampling parameters match Cohere's published repro
exactly (temp 1.0, top_p 1.0, top_k 20, 48 tokens, cut at token 2,323).

The rendered `index.html` is generated from the files in `data/`; the generator is kept outside the repo.
