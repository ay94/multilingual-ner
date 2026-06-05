# NER Evaluation — Arabic

## Language

Arabic (`ar`)

## Datasets

**AQMAR Corpus**  
74,000-token corpus of 28 Arabic Wikipedia articles, hand-annotated for named entities.  
Source: https://www.cs.cmu.edu/~ark/ArabicNER/  
Selected because one of the candidate models was trained on the other available benchmark, making AQMAR the appropriate out-of-distribution evaluation set. The data domain is also similar to the project-specific social media data in vocabulary and register.

## Candidate models

Arabic has various leaderboards but not all models are available on HuggingFace. After reviewing the available options:

- **CAMeL** (`CAMeL-Lab/bert-base-arabic-camelbert-mix-ner`) — well-established model in the Arabic NLP literature, available on HuggingFace
- **hatmimoha** (`hatmimoha/arabic-ner`) — less prominent in the literature, but the training and evaluation data appeared similar in domain to the project-specific data, making it worth investigating

---

## Model evaluation

### Evaluation links

| Model | Benchmark notebook |
|---|---|
| CAMeL | [`arabic_camel_benchmark.ipynb`](arabic_camel_benchmark.ipynb) |
| hatmimoha | [`arabic_hatmimoha_benchmark.ipynb`](arabic_hatmimoha_benchmark.ipynb) |

### Benchmark evaluation

**Scores interpretation**

The purpose of the benchmark evaluation is to assess both models on out-of-distribution data (AQMAR). CAMeL outperforms hatmimoha on all entity types except ORG. This is relevant: if the project data is heavily weighted towards organisation names, hatmimoha becomes a stronger candidate despite its lower overall benchmark score.

---

## Project-specific validation

### Models output validation

**CAMeL**  
More accurate in precision — correct entity predictions are generally right. However, struggles with entity boundaries. Example: predicts "United Arabs" as LOC but misses "Emirates", so the full span "United Arab Emirates" is not captured. This pattern of correct entity type but incomplete boundary is consistent across the sample.

**hatmimoha**  
Better recall — captures more entities overall, including organisation names that CAMeL misses. Also better at identifying full entity boundaries for ORG entities. Some inaccuracies in entity type, but the boundary coverage is stronger.

**Recommendation**  
Despite CAMeL's higher benchmark scores, hatmimoha was selected for this project. The project data is heavily ORG-weighted, and hatmimoha is better at capturing both the presence and full boundaries of organisation entities. CAMeL's boundary errors (predicting partial spans) produce more redundant and less analytically useful output for this use case.

### Model output observations

- **Tokenisation issues** — instances where tokenisation fragments entities, producing malformed predictions (e.g. `'ah'`, `'ط'`, English words or link fragments predicted as entities)
- **Boundary errors** — both models sometimes predict partial spans; e.g. for `قناة_النجباء_الفضائية`, CAMeL predicts only `النجباء` as ORG, missing the flanking words
- **Misclassifications** — majority of misclassifications are secondary to tokenisation issues rather than semantic errors
- **MISC entities** — whether MISC is analytically important depends on project requirements; hashtags and mentions should be removed before running NER if they are not targets
- **Code switching** — text with mixed Arabic/English tends to produce misclassifications and should be handled in preprocessing

---

## Project-specific dataset considerations

- Missing text segments present in the data — remove before running NER
- Sequence length / truncation: set `model_max_length` on the tokeniser; longer sequences miss more entities even without truncation
- Preprocessing recommended: remove mentions, hashtags, links before extraction
- Code-switched text is problematic for both models

---

## Considerations / observations

- The benchmark ranking (CAMeL > hatmimoha) did not hold on project-specific data — this is a documented example of why Stage 3 validation is run as a separate step
- For datasets where ORG is the dominant entity type, hatmimoha's boundary recall outweighs CAMeL's overall precision advantage
