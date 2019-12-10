# Detecting Synonymous Relationships by Shared Data-driven Definitions

This repository contains the code which allows to reproduce our results in the paper.

## System Requirements
- OS capable of running java and python code
- minimum 16GB RAM

## Dependencies
- minimum OpenJRE 8 / Oracle JRE 8
- Python3
- PyPi Packages
    - matplotlib
    - networkx
    - numpy

## FIRST STEPS

### Download **AMIE+** Association Rule Mining Code

[Download from the official website][1] or [get the code on GitHub][2].

### Install Python requirements

```shell
$ python3 -m pip install -r requirements.py
```

## Python Files

### amie\_rule\_wrapper.py

This module contains classes which allow to wrap rules from **AMIE+** standard output into an object oriented data structure.

Example usage:

```python
from amie_rule_wrapper import AMIERule, NoAMIERuleInLineError

amie_rules = set()

with open(amie_output_fn, "r") as f:
    lines = f.readlines()
    for line in lines:
        try:
            amie_rule = AMIERule(line)
            amie_rules.add(amie_rule)
        except NoAMIERuleInLineError:
            continue
```

### amie\_synonyms.py

This module implements our approach as described in the paper.
It takes the output from **AMIE+** as input and produces a dictionary of relation pairs with their confidences in being synonym in the same directory where the **AMIE+** output is located in.
Currently, it also requires the `relation2id.txt` mapping from the benchmark as input for parallelization.

For usage details and more options, see:

```shell
$ python3 amie_synonyms.py -h
```

# evaluate\_synonyms.py

This module contains code for classification and evaluation of the results produced by `amie_synonyms.py`.
It will produce precision-recall or precision@topK plots (depending on the experiment) in the same directory where the input is located in.
Currently, it also requires the `relation2id.txt` mapping from the benchmark as input for parallelization.

For usage details and more options, see:

```shell
$ python3 evaluate_synonyms.py -h
```

## Experiments

The procedure for each experiment given a benchmark \(B\) consists of three steps:

1. Mining association rules using **AMIE+** with \(B\) as input, producing the rules as output.
2. Calculating synonymous relationship pairs with their confidences using our approach with the rules as input.
3. Classifying, evaluating and plotting precision-recall or precision@topK plots for the experiment using the list calculated in the previous step.

For each experiment, we provide the exact shell commands for each step to reproduce our results.

### Datasets

We focused on the Wikidata and DBpedia datasets of our [previous work][3] for evaluation to extend our pool of baselines.
See the paper / repository for details on the samples.
The rules used for the experiments are available here: https://cloudstorage.tu-braunschweig.de/public?folderID=MjRQc2FHd3VYUThuWEQ5V3E2am1p
The gold datasets for the evaluation are also available for download: https://doi.org/10.6084/m9.figshare.11343785.v1

### Precision-Recall Evaluation in Wikidata

```shell
$ java -jar amie_plus.jar -optimcb -optimfh -minhc 0.005 wikidata-20181221TN-1k_2000_50/wikidata-20181221TN-1k_2000_50.new.tsv |& tee wikidata-20181221TN-1k_2000_50/wikidata-20181221TN-1k_2000_50.new.tsv.amie-output
$ python3 amie_synonyms.py -r wikidata-20181221TN-1k_2000_50/relation2id.txt -a wikidata-20181221TN-1k_2000_50/wikidata-20181221TN-1k_2000_50.new.tsv.amie-output -p 28 -n jaccard --min-std 0.05 --max-diff-std-hc 1.0
$ python3 evaluate_synonyms.py -r wikidata-20181221TN-1k_2000_50/relation2id.txt -g wikidata-20181221TN-1k_2000_50/synonyms_uri_filtered.txt -b wikidata-20181221TN-1k_2000_50/baselines -s wikidata-20181221TN-1k_2000_50/wikidata-20181221TN-1k_2000_50.new.tsv.amie-output.synonyms.dict
```

### Manual Quality Evaluation in DBpedia

```shell
$ java -jar amie_plus.jar -optimcb -optimfh -minhc 0.005 dbpedia-201610N-1k-filtered/dbpedia-201610N-1k-filtered.tsv |& tee dbpedia-201610N-1k-filtered/dbpedia-201610N-1k-filtered.tsv.amie-output
$ python3 amie_synonyms.py -r dbpedia-201610N-1k-filtered/relation2id.txt -a dbpedia-201610N-1k-filtered/dbpedia-201610N-1k-filtered.tsv.amie-output -p 28 -n jaccard --min-std 0.05 --max-diff-std-hc 1.0
$ python3 evaluate_synonyms.py -r dbpedia-201610N-1k-filtered/relation2id.txt -s dbpedia-201610N-1k-filtered/dbpedia-201610N-1k-filtered.tsv.amie-output.synonyms.dict -g dbpedia-201610N-1k-filtered/synonyms_uri_combined.txt -b dbpedia-201610N-1k-filtered/baselines -c simple -p 28 -f precision_topk
```


[1]: https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/
[2]: https://github.com/samehkamaleldin/amie_plus
[3]: https://github.com/JanKalo/RelAlign

