# NER Evaluation Template — [Language]

Use this template to document the evaluation for each target language. Fill in each section as you progress through the three-stage workflow.

---

## Language

[Language name and ISO code]

## Datasets

[Dataset name, size, source URL, and any access requirements]

## Candidate models

[List of models identified on HuggingFace or leaderboards, with HuggingFace IDs and license]

---

## Model evaluation

### Evaluation links

For each model, record:

| Model | Benchmark notebook | Project-specific notebook | Project-specific sample |
|---|---|---|---|
| [Model name] | [link] | [link] | [link] |

### Benchmark evaluation

**Table 1 — Benchmark scores (seqeval, entity-level F1)**

| Model | PER | LOC | ORG | MISC | Micro F1 |
|---|---|---|---|---|---|
| [Model 1] | | | | | |
| [Model 2] | | | | | |

**Scores interpretation**

[Which model performs best overall? Any entity type where a lower-ranked model outperforms? Any caveats about the benchmark domain vs project data domain?]

---

## Project-specific validation

### Models output validation

For each model, record observations from the analyst validation sample:

**[Model name]**
- Precision (mistakes): [observations]
- Recall (missed entities): [observations]
- Tokenisation issues: [observations]
- Boundary errors: [observations]

**Table 2 — Project-specific scores**

[Analyst-reviewed precision/recall counts from the validation sample]

**Scores interpretation**

[Which model is recommended for this project and why? What trade-offs exist?]

### Model output observations

- Tokenisation issues:
- Misclassifications:
- Missed entities:
- Preprocessing recommendations:

---

## Project-specific dataset considerations

- Missing or malformed text:
- Sequence length / truncation issues:
- Code switching:
- Preprocessing required before running NER:

---

## Analyst validation guidelines (project-specific)

1. Examine predicted entities for misclassifications
2. Examine sentences for missing entities
3. Flag tokenisation and boundary issues
4. Note any preprocessing patterns (hashtags, mentions, links)
5. Provide a breakdown of the most frequently erroneous entity types

---

## Considerations / observations

[Any additional notes on model selection, annotation scheme choices, or post-processing recommendations]
