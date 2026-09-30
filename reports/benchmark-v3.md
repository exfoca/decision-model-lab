# Decision Model Lab — Benchmark v3 Technical Report

## Scaling Structured Decision Models from 0.8B to 2B

### Executive Summary

Benchmark v3 provides the strongest evidence obtained so far in the Decision Model Lab that compact decision-oriented models can serve as practical components in larger AI systems.

The evaluation compared the `Jev-Style-0.8B-Decision-v3` model against the `Jev-Style-2B-Decision-v3-GGUF` family using Q4 and Q8 quantization. Each configuration was evaluated on aligned English and Brazilian Portuguese datasets containing 99 cases, under multiple execution profiles:

- closed-book;
- rule-conditioned baseline;
- `native-criteria-v1`;
- `optimized-v1`.

The main result is not merely that the 2B model achieved higher accuracy. The more relevant finding is that increasing model capacity changed the behavior of the system in several dimensions simultaneously:

- accuracy increased substantially;
- calibration improved;
- structured semantic profiles became more effective;
- Portuguese performance approached English performance;
- Q4 preserved most of the decision quality observed with Q8;
- selective prediction became operationally meaningful;
- latency improved in the tested GGUF runtime despite the larger model size.

The best observed configurations reached **86.87% accuracy**, corresponding to **86 correct decisions out of 99 cases**.

The results suggest that the 2B model crosses an important practical capacity threshold for structured decision tasks.

---

## 1. Experimental Scope

Benchmark v3 evaluates decision models over a common set of aligned cases rather than free-form text generation.

The benchmark contains:

- 99 aligned evaluation cases;
- English and Brazilian Portuguese dataset variants;
- multiple decision spaces;
- baseline, boundary, contrastive, adversarial, counterfactual, temporal, causal, evidence, security, scope, and other semantic categories;
- short, medium, and long contexts.

Three model configurations were compared:

| Model | Runtime representation |
|---|---|
| Jev-Style-0.8B-Decision-v3 | original 0.8B candidate |
| Jev-Style-2B-Decision-v3-GGUF | Q4 |
| Jev-Style-2B-Decision-v3-GGUF | Q8 |

Each model was evaluated under four semantic/execution profiles:

| Profile | Description |
|---|---|
| `closed-book` | decision without explicit rule conditioning |
| baseline rule-conditioned | rules supplied without an additional semantic optimization profile |
| `native-criteria-v1` | rule-conditioned execution using criteria close to the model's native decision representation |
| `optimized-v1` | refined semantic representation intended to improve rule interpretation |

Each profile was executed independently in English and Brazilian Portuguese.

This produced a total of 24 benchmark runs.

---

## 2. Aggregate Accuracy

The clearest result is the scaling effect between the 0.8B and 2B models.

### 0.8B

The best 0.8B configuration was:

`native-criteria-v1`, English

with:

- 73 correct cases;
- 73.74% accuracy.

The optimized profile did not improve accuracy at this model size:

| Configuration | Accuracy |
|---|---:|
| closed-book EN | 65.66% |
| closed-book PT-BR | 59.60% |
| native criteria EN | **73.74%** |
| native criteria PT-BR | 66.67% |
| optimized EN | 65.66% |
| optimized PT-BR | 60.61% |
| rule-conditioned baseline EN | 60.61% |
| rule-conditioned baseline PT-BR | 60.61% |

### 2B Q4

The 2B Q4 model produced a substantially different result:

| Configuration | Accuracy |
|---|---:|
| closed-book EN | 75.76% |
| closed-book PT-BR | 73.74% |
| native criteria EN | 83.84% |
| native criteria PT-BR | 81.82% |
| optimized EN | **86.87%** |
| optimized PT-BR | 83.84% |
| rule-conditioned baseline EN | 81.82% |
| rule-conditioned baseline PT-BR | 79.80% |

### 2B Q8

The Q8 model produced the strongest overall result:

| Configuration | Accuracy |
|---|---:|
| closed-book EN | 77.78% |
| closed-book PT-BR | 70.71% |
| native criteria EN | 83.84% |
| native criteria PT-BR | 82.83% |
| optimized EN | **86.87%** |
| optimized PT-BR | **86.87%** |
| rule-conditioned baseline EN | 83.84% |
| rule-conditioned baseline PT-BR | 82.83% |

The result is particularly notable because the optimized profile, which was ineffective on the 0.8B model, became the strongest profile on the 2B model.

---

## 3. Evidence of a Capacity Threshold

The transition from 0.8B to 2B does not resemble a uniform incremental improvement.

Instead, the benchmark suggests a change in the model's ability to integrate structured information.

Under `optimized-v1`:

| Language | 0.8B | 2B Q4 | 2B Q8 |
|---|---:|---:|---:|
| English | 65.66% | **86.87%** | **86.87%** |
| PT-BR | 60.61% | 83.84% | **86.87%** |

For English, this corresponds to an improvement from 65 to 86 correct decisions.

For Brazilian Portuguese, the strongest result improves from 60 to 86 correct decisions.

This is unlikely to be explained by a small change in classifier behavior alone.

A plausible interpretation is that the larger model can more reliably combine:

```text
context
+ available choices
+ semantic criteria
+ explicit rules
+ boundary conditions
```

The 0.8B model appears capable of using simpler semantic scaffolding, particularly `native-criteria-v1`, but frequently fails to benefit from the denser optimized representation.

The 2B model behaves differently.

Once sufficient capacity is available, `optimized-v1` becomes beneficial rather than detrimental.

This is one of the most important findings of Benchmark v3.

---

## 4. Calibration

Model quality cannot be evaluated through accuracy alone.

For a decision engine, the relationship between confidence and correctness is especially important because confidence may later be used for routing, abstention, escalation, or selective execution.

The 2B models improved substantially on Brier score and log loss.

Representative best values:

| Model family | Best Brier score | Best log loss |
|---|---:|---:|
| 0.8B | ~0.437 | ~0.761 |
| 2B Q4 | ~0.234 | ~0.433 |
| 2B Q8 | **~0.219** | **~0.408** |

Lower values are better.

The reduction is substantial enough to indicate that scaling improved not only top-1 decisions but also the underlying probability distributions.

This is particularly important for future uses where the model may not directly execute a decision, but instead estimate whether a decision is reliable enough to proceed automatically.

Expected Calibration Error also improved in several configurations, although ECE should be interpreted cautiously because Benchmark v3 contains only 99 cases and uses a relatively small calibration sample.

Brier score and log loss currently provide more robust evidence.

---

## 5. Selective Prediction

One of the most operationally significant differences between the two model sizes appears in selective prediction.

Selective prediction asks:

> How many cases can the model answer while remaining below a specified empirical error budget?

The 0.8B model performs poorly under this criterion.

Most 0.8B runs cover only approximately 0–4% of cases under strict error budgets, with the strongest configuration reaching only low double-digit coverage.

The 2B model behaves very differently.

Some configurations reach approximately:

- 18% coverage under a 1% empirical error budget;
- 70–82% coverage under a 5% empirical error budget;
- approximately 90% coverage under a 10% empirical error budget.

This represents a qualitative change in usefulness.

A model with moderate global accuracy may still be highly valuable if it can reliably distinguish between cases it understands and cases that should be escalated.

A future decision system could therefore use a policy such as:

```text
decision request
      |
      v
small decision model
      |
      +-- confidence >= threshold --> accept structured result
      |
      +-- confidence < threshold --> escalate
                                    |
                                    +-- larger LLM
                                    +-- retrieval
                                    +-- additional reasoning
                                    +-- human review
```

This architecture may be more important than maximizing global benchmark accuracy.

### Important limitation

The thresholds in Benchmark v3 are currently derived and evaluated on the same benchmark population.

Therefore, the reported selective coverage must be considered **in-sample risk coverage**.

Production-grade evaluation should instead use:

```text
calibration set
      |
      v
select confidence threshold
      |
      v
freeze threshold
      |
      v
independent holdout test
      |
      v
measure real risk / coverage
```

Until this separation exists, selective prediction results should be treated as strong experimental evidence rather than production guarantees.

---

## 6. Q4 vs Q8 Quantization

The Q4 results are particularly significant.

Quantization generally introduces a concern that reduced numerical precision may alter the model's decision surface.

Benchmark v3 shows surprisingly little degradation.

The strongest English configuration produced:

- Q4 optimized: 86.87%;
- Q8 optimized: 86.87%.

For Brazilian Portuguese:

- Q4 optimized: 83.84%;
- Q8 optimized: 86.87%.

Other profiles also remain close.

This suggests that the structured decision behavior of the 2B model is comparatively robust to Q4 quantization.

That has important deployment implications.

If confirmed on larger benchmarks, Q4 could become a primary deployment format rather than merely a degraded low-resource alternative.

Potential targets include:

- local desktop agents;
- CPU-oriented inference;
- edge systems;
- mobile devices;
- multiple concurrent decision workers;
- always-on background decision services.

Q8 remains useful as a reference configuration for quality evaluation, but Benchmark v3 provides no evidence that Q4 should be dismissed for high-quality decision workloads.

---

## 7. English vs Brazilian Portuguese

Language behavior changes substantially between the 0.8B and 2B models.

With the 0.8B model, PT-BR generally produced a 5–7 percentage-point accuracy penalty relative to equivalent English configurations.

This effect becomes much smaller in the 2B family.

Most notably:

`optimized-v1`, Q8

produced:

- English: 86.87%;
- PT-BR: 86.87%.

However, equal aggregate accuracy does not imply identical decisions.

The two language variants can reach the same accuracy while making errors on different cases.

This suggests a more precise interpretation:

The 2B model appears to reduce the aggregate language penalty, but some degree of **language-conditioned decision drift** remains.

Formally:

```text
Accuracy(EN) ≈ Accuracy(PT-BR)
```

does not necessarily imply:

```text
ErrorSet(EN) = ErrorSet(PT-BR)
```

Future benchmarks should therefore treat cross-language agreement as an explicit metric rather than comparing only aggregate accuracy.

This may also enable bilingual consistency checks where semantically equivalent representations are independently evaluated and disagreements are escalated.

---

## 8. Decision-Type Performance

The benchmark contains primarily `choice` decisions, with a smaller number of `noul` cases.

The 2B models improve substantially on the dominant `choice` category.

The strongest runs reach approximately:

- 86% accuracy on `choice`;
- 100% accuracy on the current six `noul` cases.

The small `noul` sample prevents strong conclusions about abstention behavior, but the result is encouraging.

Future datasets should substantially expand explicit abstention and insufficient-evidence cases.

---

## 9. Semantic Categories

The improvement from 0.8B to 2B is not uniform across all semantic categories.

Several categories become especially strong under the 2B optimized configurations.

Examples include:

- counterfactual reasoning;
- threshold decisions;
- scope classification;
- multiclass decisions;
- temporal reasoning;
- causal judgments;
- contrastive examples;
- adversarial cases;
- invariance tests.

Some optimized 2B configurations reach 88.89–100% accuracy on several of these nine-case subsets.

This suggests that explicit semantic conditioning becomes increasingly useful as model capacity grows.

However, these per-tag subsets remain small.

A result of 88.89% on a nine-case category means eight correct cases, while 100% represents nine correct cases.

These values should therefore be treated primarily as diagnostic signals rather than stable estimates of category-level accuracy.

---

## 10. Persistent Failure Modes

Despite strong aggregate performance, Benchmark v3 exposes a small number of cases that remain difficult across the strongest 2B configurations.

The most interesting failures are not ordinary low-confidence errors.

Some are **systematic high-confidence errors**.

Two particularly important patterns appear.

### 10.1 Insufficient evidence incorrectly converted into a definite answer

In several cases, the expected result is conceptually equivalent to:

```text
insufficient_evidence
```

yet the model produces a definite positive or negative decision.

This behavior is more concerning than an ordinary classification error because it indicates failure at the epistemic boundary.

The model is not simply selecting the wrong class.

It may be failing to recognize that the available information is insufficient to justify a definite judgment.

### 10.2 Confident systematic errors

Some persistent failures occur with high output probability.

These cases demonstrate that confidence alone cannot eliminate all errors.

A future production architecture must assume:

```text
high confidence != guaranteed correctness
```

Selective prediction can reduce risk substantially, but systematic confident errors require additional defenses such as:

- adversarial evaluation;
- rule verification;
- cross-model disagreement detection;
- semantic invariance tests;
- explicit uncertainty classes;
- external validators.

---

## 11. Latency and Throughput

An unexpected result is that the 2B GGUF configurations are faster than the previously tested 0.8B system in the current laboratory environment.

Representative mean latency:

| Configuration | Mean latency |
|---|---:|
| 0.8B closed-book EN | ~101 ms |
| 2B Q4 closed-book EN | ~70 ms |
| 2B Q8 closed-book EN | ~70 ms |

The 2B models also achieve higher measured throughput.

This result must not be interpreted as evidence that a 2B neural network is intrinsically cheaper than a 0.8B network.

The execution stacks are different.

The correct conclusion is:

> The tested 2B GGUF system provides both higher decision quality and better observed operational latency than the previously tested 0.8B system.

For an engineering system, model architecture and runtime must therefore be evaluated together.

The relevant unit is not only the neural model.

It is:

```text
model
+ quantization
+ inference runtime
+ execution protocol
+ semantic representation
```

---

## 12. Macro F1 Interpretation

Benchmark v3 reports Macro F1, with some 2B configurations reaching values above 90%.

This metric requires careful interpretation.

The benchmark combines multiple independent decision spaces with different label vocabularies.

Labels may include, for example:

```text
yes
no
insufficient_evidence

approve
review
reject

normal
degraded
unavailable

account
regional
endpoint
global
```

These labels do not belong to one conventional mutually exclusive classification task.

A global Macro F1 therefore assigns equal weight to labels originating from different decision families.

It remains useful as a diagnostic aggregate, but should not be interpreted in exactly the same way as Macro F1 from a standard single-task classifier.

Future reports should place greater emphasis on:

- overall accuracy;
- F1 per decision family;
- per-tag accuracy;
- calibration;
- high-confidence error rate;
- risk-coverage behavior;
- cross-language agreement.

---

## 13. System-Level Interpretation

The results suggest that compact decision models should not necessarily be evaluated as replacements for large language models.

A more useful role may be as a **decision layer** inside a larger AI architecture.

Potential responsibilities include:

```text
classify
prioritize
rank
accept
reject
abstain
route
escalate
retain
discard
```

The model can therefore operate as a fast background decision engine supporting more expensive components.

Example architecture:

```text
                    +----------------+
                    | incoming event |
                    +-------+--------+
                            |
                            v
                  +-------------------+
                  | decision model    |
                  | small / local     |
                  +---------+---------+
                            |
                +-----------+------------+
                |                        |
          high confidence           uncertain
                |                        |
                v                        v
        structured action        expensive reasoning
                                     |
                         +-----------+-----------+
                         |                       |
                        LLM                    RAG
                         |                       |
                         +-----------+-----------+
                                     |
                                     v
                               final decision
```

This changes the optimization objective.

The model does not necessarily need near-perfect global accuracy.

It needs to provide:

1. inexpensive inference;
2. useful probability distributions;
3. strong accuracy over common decisions;
4. reliable identification of uncertain cases;
5. predictable escalation behavior.

Benchmark v3 provides encouraging evidence for all five, although selective prediction still requires independent calibration validation.

---

## 14. Implications for Agent Architectures

Modern AI agents frequently invoke expensive generative models for operations that are fundamentally classification or routing problems.

Examples include:

- whether retrieval is necessary;
- whether a memory should be retained;
- whether evidence is sufficient;
- which tool should be called;
- whether a result requires additional verification;
- whether an event is relevant;
- whether context should be promoted into working memory;
- whether another reasoning pass is justified.

A compact decision model could perform these operations continuously without invoking the primary language model.

The architecture becomes closer to:

```text
fast decision layer
        |
        +--> obvious case: act immediately
        |
        +--> uncertain case: invoke deeper cognition
```

This resembles a computational distinction between fast and slow reasoning.

The small model becomes the high-frequency judgment layer, while the larger generative model is reserved for tasks requiring synthesis, explanation, or complex reasoning.

---

## 15. Proposed Benchmark v4

Benchmark v3 has reached the point where simply increasing the number of random cases would provide limited additional information.

Benchmark v4 should instead be **error-driven**.

### 15.1 Expand systematic failures

Each persistent failure discovered in v3 should generate a family of minimal pairs.

For example:

```text
original failure
      |
      +-- wording variation
      +-- threshold +epsilon
      +-- threshold -epsilon
      +-- evidence added
      +-- evidence removed
      +-- irrelevant distractor added
      +-- ordering changed
      +-- language changed
```

Ten to twenty variants per failure would reveal whether the error represents:

- dataset ambiguity;
- lexical sensitivity;
- threshold confusion;
- semantic failure;
- true model limitation.

### 15.2 Separate calibration and evaluation

Benchmark v4 should introduce:

```text
calibration dataset
test dataset
```

Confidence thresholds must be selected exclusively on the calibration set and frozen before test evaluation.

This will allow legitimate claims about selective risk.

### 15.3 Add risk-coverage curves

Rather than reporting only three fixed budgets, future reports should include:

- full risk-coverage curve;
- Area Under the Risk-Coverage Curve;
- threshold stability;
- high-confidence error rate.

### 15.4 Add decision-family metrics

Metrics should be computed independently for each decision vocabulary.

For example:

```text
binary evidence decisions
approval decisions
impact decisions
scope decisions
abstention decisions
```

This would make Macro F1 easier to interpret.

### 15.5 Measure quantization disagreement

Q4 and Q8 should be compared using explicit metrics:

- top-1 agreement;
- probability divergence;
- exclusive-correct cases;
- disagreement rate.

### 15.6 Measure language invariance

Equivalent EN and PT-BR cases should also report:

- decision agreement;
- confidence drift;
- probability divergence;
- language-exclusive errors.

### 15.7 Expand epistemic-boundary evaluation

Cases where the correct response is equivalent to:

```text
unknown
unavailable
insufficient_evidence
abstain
```

should become a dedicated benchmark family.

Current results suggest that identifying the boundary between "a decision can be made" and "the evidence is insufficient" may be one of the most important unresolved capabilities.

---

## 16. Conclusions

Benchmark v3 materially changes the status of the 2B decision model in the Decision Model Lab.

The 0.8B candidate demonstrated that compact structured decision models were viable.

The 2B model demonstrates something stronger:

**they may be operationally useful.**

The most important findings are:

1. Scaling from 0.8B to 2B produces large and consistent quality improvements.

2. The optimized semantic profile becomes effective only at the larger capacity, suggesting a meaningful capacity threshold.

3. The strongest runs achieve 86 correct decisions out of 99.

4. Calibration improves substantially alongside accuracy.

5. Q4 retains most of the Q8 decision quality, making aggressive quantization particularly promising.

6. PT-BR performance approaches English performance at 2B scale.

7. Selective prediction changes from largely unusable in the 0.8B model to potentially practical in the 2B family.

8. GGUF execution substantially improves observed latency and throughput in the current laboratory environment.

9. Remaining errors are increasingly concentrated in identifiable semantic failure modes rather than broad random degradation.

10. Recognition of insufficient evidence appears to be a particularly important frontier for the next evaluation cycle.

The central result can therefore be summarized as:

```text
0.8B:
proof that structured decision inference works

2B:
evidence that structured decision inference may be deployable
```

The next phase of the Decision Model Lab should focus less on demonstrating raw accuracy and more on characterizing the operational envelope of the model:

```text
When should it be trusted?
When should it abstain?
When should it escalate?
Which errors remain systematic?
How stable are decisions across language and quantization?
```

Answering those questions will determine whether compact decision models can serve as a reliable low-latency cognitive layer inside larger AI systems.
