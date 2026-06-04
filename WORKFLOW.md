# NER Evaluation Workflow

This document outlines the evaluation workflow for Named Entity Recognition in multilingual projects. The workflow consists of three stages:

1. **Model and benchmark identification** — finding candidate models and suitable benchmark datasets for each target language
2. **Benchmark evaluation** — ranking candidate models based on performance against benchmark datasets
3. **Project-specific validation** — validating top-ranked models against actual project data using analyst review

The workflow is project-dependent: when benchmark data and candidate models are available before project data arrives, start with Stage 2. Once project data is available, proceed to Stage 3.

---

## Stage 1 — Model and Benchmark Identification

### Identifying candidate models

This stage involves compiling a list of candidate NER models for each target language.

1. **Leaderboards** — platforms such as BERT LANG STREET serve as a directory of BERT-based NER models and provide a useful summary of the literature. Not all listed models are available online, so availability should be confirmed before proceeding.

2. **HuggingFace** — the recommended source. Prefer models available on HuggingFace for ease of implementation. Models not on HuggingFace can still be used but require additional handling.

3. **Annotation scheme** — each model uses its own annotation scheme. The evaluation process follows the CoNLL BIO scheme (PER, LOC, ORG, MISC). If a candidate model uses a different scheme, label alignment is required before evaluation (see the template notebooks for handling).

4. **Preprocessing** — some models include a preprocessing layer before tokenisation, particularly for low-resource languages. Check HuggingFace model cards for this detail; if the model is not on HuggingFace, determine preprocessing requirements before running evaluation.

### Identifying benchmarks

1. **Leaderboards** — review leaderboards to identify benchmarks commonly used for each language. Multilingual benchmarks on HuggingFace (Wikiann, MasakhaNER, CoNLL) are useful starting points for low-resource languages.

2. **HuggingFace Datasets** — navigate to the token classification task in the datasets section to find suitable benchmarks.

3. **Alternative sources** — if the standard benchmark for a language is not on HuggingFace, check GitHub or institutional websites. Some datasets require registration or a formal request.

4. **Training data** — confirm the benchmark was not used to train any of the candidate models. Use only the test split where possible to avoid data leakage.

5. **Out-of-domain benchmarks** — where available, prefer a benchmark from a domain the model was not trained on (e.g. a news-trained model tested against a Wikipedia dataset). This better approximates how the model will perform on project-specific data.

---

## Stage 2 — Benchmark Evaluation

The purpose of this stage is to rank candidate models and expose initial model-specific issues before project data is available. It is treated as a black-box process: given a list of candidates, rank them by performance, exclude poor performers, and flag issues such as specialised preprocessing requirements or unusual annotation schemes.

Each model is evaluated and produces a classification report in two formats:

- **Seqeval** — entity-level evaluation (F1 by entity type: PER, LOC, ORG, MISC)
- **Sklearn** — token-level boundary evaluation (flat F1, boundary-sensitive)

**Caveats:**

- Performance on the benchmark is indicative but not definitive for project-specific data — domain mismatch is common.
- Exhaustive error analysis at this stage is not required; the purpose is ranking and flagging, not diagnosis.
- F1 scores are abstract; a model with a lower benchmark score may outperform a higher-ranked model on project-specific data due to domain alignment.
- This stage identifies potential model weaknesses for further exploration in Stage 3.

---

## Stage 3 — Project-Specific Validation

Once project data is available and a ranked list of candidates exists, the top-ranked models are applied to a sample of the actual project data and reviewed by analysts.

### Methodological caveats

- **Sample size and representativeness** — the sample must be representative of the full data distribution. Frame the validation as a sanity check against the benchmark ranking rather than a definitive evaluation — analysts are checking whether the top model holds up on real data, not running a full assessment from scratch.
- **Time and complexity** — this stage is time-intensive. Analysts typically need an initial briefing and ongoing support to follow the guidelines consistently.
- **Analyst focus** — instructions must direct analysts to evaluate named entity recognition specifically, not general text quality or content interest.

### Evaluation approaches

**Per-model evaluation** — analysts review a sample for each model individually. If a model is deemed inadequate, the process moves to the next on the ranked list. Simpler to run and easier to write instructions for.

**Comparative evaluation** — two models are reviewed side by side on the same sample. More informative but more complex to manage: instructions are harder to write, and the validation file is harder for analysts to navigate. Use only when analyst capacity and deadlines allow.

### Sample generation guidelines

Before providing samples to analysts, the following preprocessing steps must be applied:

1. **Clean the text** — remove links, hashtags, mentions, and similar elements. NER models are typically not optimised for these and they produce noise in the output.

2. **Handle long sentences** — sentences that approach or exceed the tokeniser's maximum length should be split or excluded. Long sentences are also harder for analysts to review accurately.

3. **Remove missing or off-language text** — remove segments of missing text and exclude sentences in a language other than the primary one being analysed.

4. **Sampling strategy** — in some projects, stratified sampling (e.g. by account or platform) may be appropriate. This is project-dependent.

### Analyst validation guidelines

Analysts work with files containing sentences and their corresponding model extractions. The goal is to assess model suitability for the project data — not to annotate exhaustively, but to identify patterns of error.

1. **Entity identification** — read each sentence and identify potential named entities, then examine the model's predictions for misclassifications.

2. **Missed entities** — flag entities the model failed to identify. Confirm they are genuine named entities, not just content the analyst finds interesting.

3. **Tokenisation issues** — report malformed token predictions. Example: in "United Arab Emirates," the model predicts "United Arabs" as LOC and drops "Emirates." Note these for post-processing.

4. **Preprocessing issues** — identify elements consistently misclassified (hashtags, mentions, long sequences) and note the pattern.

5. **Model weaknesses** — provide a summary of the model's weaknesses and suggestions for handling specific issues.

6. **Error counts** — for each sentence, record the number of misclassified entities and the number of missed entities to enable quantitative summary across the sample.
