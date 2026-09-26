# Optimizing LLMs for Contextual Reasoning in Multi-Task Environments

Official code release for the paper **"Optimizing Large Language Models for Contextual Reasoning in Multi-Task Environments"**, accepted at the **Conference on Artificial Intelligence (COAI 2025)**.

## Overview

This repository contains the implementation used in the paper to evaluate and optimize large language models on contextual reasoning across multiple tasks, including multi-task fine-tuning, prompting strategies, and evaluation harnesses.

## Installation

```bash
pip install -r requirements.txt
```

## Repository structure

```
.
├── benchmarks/      # evaluation benchmarks
├── data/            # data preparation
├── experiments/     # experiment configs
├── notebooks/       # analysis notebooks
├── results/         # experimental results
├── src/             # model and training code
├── tests/           # unit tests
├── requirements.txt
└── README.md
```

## Usage

Train the model:

```bash
python src/train.py --config experiments/config.yaml
```

Run evaluation:

```bash
python benchmarks/benchmark.py
```

## Reference

If you use this code, please cite:

```bibtex
@inproceedings{optimizing2025,
  title={Optimizing Large Language Models for Contextual Reasoning in Multi-Task Environments},
  booktitle={Conference on Artificial Intelligence (COAI 2025)},
  year={2025},
}
```
