# LLM Project

This project is based on the paper [LongRoPE](https://arxiv.org/pdf/2402.13753) (Ding et al. 2024). The goal of this project is to explore a method to consistenly increase the context length of an LLM. The main idea start from an empirical fact: if we increase the context length of an LLM, the performances tends to deteriorate. Various methods has been explored to help the model to break through the limits of context size.

This project is in the context of the cours [LLM](https://www.master-mva.com/cours/large-language-models-introduction-and-applications-for-code/) @ MVA ( [M. Fijalkow](https://github.com/nathanael-fijalkow) & [M. Lelarge](https://github.com/mlelarge))

## Repo

The repo has the following architecture:

```
Directory structure:
└── paulcauch-llm-project-mva-2025/
    ├── README.md
    ├── setup.py
    ├── data/
    │   ├── __init__.py
    │   └── input.txt
    ├── ntbk/
    │   ├── __init__.py
    │   └── simple_use.ipynb
    ├── report/
    │   └── __init__.py
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── extension.py
    │   ├── finetune.py
    │   ├── longrope.py
    │   ├── longrope_utils.py
    │   ├── rope.py
    │   ├── search.py
    │   ├── utils_data.py
    │   └── utils_general.py
    └── test/
        ├── __init__.py
        └── test.ipynb
```

* ``src`` contains all the function used to perform LongRoPE
* ``report`` contains the slides of the presentation
* ``ntbk`` contains an example of the run of the code for LongRoPE, and some visual representations
* ``data`` contains a simple dataset, it's actually just a poem (the same one used by [Karpathy](https://www.youtube.com/@AndrejKarpathy) in his videos)

Please note that the github was mainly based on the two following repo: [official LongRoPE](https://github.com/microsoft/LongRoPE) [non-official LongRoPE](https://github.com/jshuadvd/LongRoPE). The first one is developped to handle big llm (mistral, llama etc...) while the second one is adapted to a much smaller size but contains several mistakes in the code (some of them are just details, purely dev details).


Since the computations are really heavy (For the ``search`` function, we take about 1 hour for 1 iteration with non trivial parameters with a GPU), the code here is to give a clear insight on how LongRoPE works, how can it be implemented, but we can not reach results since they are experimentals and were showed on large LLM with (really) large datasets and by definition of the problem, we <u>need</u> to perform it on huge context lengths to have results

## Get Start

To run the code, please first run

```bash
pip install -e .
```


## Contributors

* Paul CAUCHETEUX
* Sacha HAKIM
* Tom WELCH
* Quentin MOAYEDPOUR