# Benchmarks

Model evaluation notebooks for NER across 15 languages. Each notebook evaluates one or more models against a benchmark dataset using seqeval (entity-level) and sklearn (token-level) metrics.

## Structure

```
benchmarks/
  ar/   Arabic — CAMeL, hatmimoha models vs WikiANN
  de/   German — gunghio/xlm, julian/roberta, fhswf/bert models vs WikiANN + GermEval 2014
```

## Pattern

One notebook per model. Each notebook:

1. Installs from `pip install git+https://github.com/ay94/multilingual-ner.git`
2. Loads the benchmark dataset(s) for the language
3. Checks and aligns dataset labels to standard BIO scheme (PER / LOC / ORG / MISC)
4. Loads the model and inspects `id2label` to define model label alignment
5. Evaluates using seqeval (entity-level F1) and sklearn (token-level F1)

See [`template.ipynb`](../template.ipynb) for a blank notebook template and [`evaluation-template.md`](../evaluation-template.md) for the structured documentation template.
