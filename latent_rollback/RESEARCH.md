# Latent Rollback: Reversing LLM Logical States via Residual Stream Vector Subtraction

*Internal technical report — March 2026*

---

## Abstract

We demonstrate that a transformer's logical state — its implicit "belief" about a factual value such as a port number — can be reversed by subtracting a latent delta vector from the residual stream at a mid-network layer during generation. The delta is computed as the difference between layer activations under two prompt conditions (pre- and post-state-change), extracted at the last token position. Applied to Llama 3 8B base (BF16) running via MLX on Apple Silicon, the technique achieves 60/60 successful rollbacks under same-context conditions: the model reads text asserting a new value but generates the prior value. This report documents the full experimental protocol, failure mode taxonomy, and architectural conclusions for an MVP commit/rollback engine.

---

## Core Experiment: Phase 2 — State Reversibility

**Model:** `mlx-community/Meta-Llama-3-8B` (BF16 weights, base model, not instruct)

The base model was chosen deliberately. Instruction-tuned variants have RLHF-baked answer-format priors strong enough to resist residual stream interventions — the model's trained behavior overrides the steered representation before it reaches the generation head. The base model is more "plastic" to steering.

**Protocol.** Two prompts share identical framing around a database configuration. Prompt A ends with the model outputting `5432` (the original port). Prompt B appends a migration notice — "MIGRATION COMPLETE: port updated to 8080" — and the model outputs `8080`. The delta vector is:

```
delta = v_B(layer=14, last_token) - v_A(layer=14, last_token)
```

Rollback is applied by subtracting `delta` from all prompt-position residual stream activations at layer 14 during generation of Prompt B. The intervention happens before the forward pass completes — the model processes the modified representation through the remaining 17 layers.

**Result:** 60/60 trials with greedy decoding (temperature=0) returned `5432` despite the prompt text explicitly stating the port is `8080`. This is MASSIVE_SUCCESS by the experiment's classification criteria.

---

## Test 1: Universal Patch — Cross-Prompt Generalization

The critical question after same-context success: does the delta transfer to novel prompts with different surface form? Five novel prompt formats were tested — a webapp URL, a Kubernetes YAML snippet, a bash environment variable export, narrative prose describing a monitoring dashboard, and a JIRA ticket.

Results: 2/5 succeeded (webapp URL and JIRA ticket). The three failures (k8s YAML, bash env var, narrative prose) represent distinct failure modes analyzed below.

---

## Test 2: Multi-Hop Ledger

Three sequential states were constructed: A (port 5432), B (port 8080), C (port 9000), with deltas `delta_1 = v_B - v_A` and `delta_2 = v_C - v_B`.

The cosine similarity between `delta_1` and `delta_2` was **-0.04** — near-orthogonal. This is an important positive result: independent state changes occupy geometrically independent subspaces. The ledger does not accumulate interference.

Single-step rollback by subtracting `delta_2` achieved PARTIAL_SUCCESS at scale 3.0. Full two-hop rollback by subtracting `delta_1 + delta_2` failed — the combined vector undershoots, indicating that summing deltas across hops does not linearly compose in activation space at the magnitudes tested.

---

## Delta Norm Profile and Layer Selection

Computing `||v_B(L) - v_A(L)||` across all 32 layers reveals a monotonically increasing profile peaking at layer 31. This is the first structural finding: the largest representational difference is at the final layer, not the middle of the network.

This makes layer 31 a poor intervention site despite its maximum norm. By layer 31, there is one layer remaining before the language model head — insufficient depth for the model to re-process a steered representation into a coherent output distribution. The intervention fires too late.

The empirically optimal layer was **layer 14**, leaving 18 layers of re-processing depth. The implicit selection criterion is not maximum projection onto the delta, but something closer to `max(projection * remaining_depth)` — a trade-off between signal strength and re-processing capacity. Calibrating this criterion across other model families is left for future work.

---

## Probe-Then-Steer: Projection Analysis

Before steering, novel prompts were projected onto the delta unit vector at layer 14. The results:

| Prompt Format | Projection | Outcome |
|---------------|------------|---------|
| webapp URL    | -0.039     | SUCCESS |
| k8s YAML      | -0.013     | FAILURE |
| bash env var  | -0.051     | FAILURE |
| narrative     | +0.590     | FAILURE (outputs 3306) |
| JIRA ticket   | +0.534     | SUCCESS |

Low projection (webapp, k8s, env) indicates geometry mismatch: the prompt encodes the port concept in a different subspace than where the delta points, so the delta has near-zero leverage regardless of scale. The narrative case is the most instructive: high projection, but the model outputs `3306` (MySQL's default port) rather than `5432`. The model never encoded `8080` to begin with — its pretrained association between "monitoring dashboard" and MySQL fired before the port appeared in the residual stream. Subtracting a delta cannot recover a state that was never encoded.

---

## Failure Mode Taxonomy

Three distinct mechanisms account for all rollback failures.

**Geometry mismatch.** Novel prompts with different surface structure (YAML syntax, shell variable syntax) encode port information in subspaces that the single-pair delta does not span. The delta is essentially blind to these formats. The fix is concept vector extraction from many diverse examples rather than a single contrast pair — but this requires significantly more data (see Concept Vector Attempt below).

**Prior override.** In the narrative prompt, the model's pretrained associations dominate. "Monitoring dashboard" reliably co-occurs with MySQL in pretraining data, so the model instantiates a MySQL-world state where `8080` is never encoded as the port. Rollback by subtraction is inapplicable here; positive steering toward the target state is required instead.

**Format trap failure.** Some prompt structures don't position the target token as the first generated token. In multi-format averaging experiments, 7 of 8 prompts failed a sanity gate because the prompt did not terminate at a position that forced the port as the immediate continuation. This is a prompt-engineering constraint, not a fundamental limitation of the steering mechanism.

---

## Concept Vector Attempt

To address geometry mismatch, 8 examples per group were collected (5432-context prompts and 8080-context prompts across diverse formats), and a concept direction was computed as `mean(Group B) - mean(Group A)`.

The result was 1/5 success versus the single-pair delta's 2/5 — a regression. The cause is insufficient diversity at 8 examples per group: the collected prompts share too much vocabulary ("production database" context appears in both groups), so the mean difference does not isolate the port-value concept. The representation learning literature on RepE recommends 32-64+ contrast pairs minimum to extract a reliable concept direction. At 8 pairs, the averaged vector degrades rather than improves.

---

## Commit Engine Architecture

The prototype commit engine built around these findings has the following structure.

**DeltaLedger** stores the full 32-layer delta profile per commit (all layer norms, plus the layer-14 activation vector). Each commit costs two forward passes — roughly one second on Apple Silicon M-series — and approximately 512KB of storage per commit.

**Rollback** requires one probe pass (to compute projection) plus modified generation. Layer selection uses the `projection * remaining_depth` heuristic. Multi-hop rollback sums triggered deltas at the modal intervention layer before generation begins.

**Persistence** uses JSON serialization. The ledger is queryable by commit hash, with full delta profiles retained for potential future use at non-default layers.

---

## Limitations

| Limitation | Detail |
|---|---|
| Generalization scope | Rollback is reliable only for prompts syntactically similar to the original commit pair |
| Prior override | Prompts where the model's pretrained association dominates are not addressable with subtraction-only rollback |
| Layer calibration | Layer 14 is empirically optimal for Llama 3 8B; other architectures require independent calibration |
| Distributed encoding | Monotonically increasing norm profile suggests concept encoding is distributed across all layers, not localized to a semantic "peak" |
| Single-position extraction | Delta computed at last token position only; cases where the concept is spread across multiple token positions are not captured |
| Concept vector sample size | 8 examples per group insufficient; RepE literature suggests 32-64+ contrast pairs for reliable direction extraction |
| Multi-hop composition | Summing multi-hop deltas does not linearly compose; full two-hop rollback currently fails |

---

## What Works for MVP

Same-context rollback is reliable at 60/60. Commit and rollback latency are both under one second. The orthogonality result (cosine = -0.04 for independent state changes) validates the ledger's non-interference model mathematically. The architecture — DeltaLedger, probe-then-steer, depth-weighted layer selection, JSON persistence — is complete and functional. The primary gap between MVP and general use is concept vector quality, which is a data collection problem with a known solution path.
