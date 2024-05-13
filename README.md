# RAG-Framework-using-DSPy 

This repo contains jupyter notebooks for building a RAG Application using Stanford's DSPy which is a framework for algorithmically optimizing LM prompts. 


## Compiled Baleen RAG output
![image](https://github.com/Jeremyugo/RAG-Framework-using-DSPy/assets/36512525/eb3150a4-cc56-429d-b85b-70ca883587b6)

**Context & Reasoning steps**
![image](https://github.com/Jeremyugo/RAG-Framework-using-DSPy/assets/36512525/af222d67-42f7-426e-9fcb-ca8a3bd0b31a)


### Why DSPy?
To build a complex RAG application using for example LangChain, you generally have to 
1. break down the problem into steps
2. prompt the language models until each step in the application works well in isolation
3. continuously tweak all steps till they work well together
4. use synthetic data to fine-tune each step

because there are so many parts in the application, tweaking and finetuning them to get optimum performance can both be messy and time-consuming. What DSPy does is that it separates the RAG program from the parameters that need to be tweaked, and also introduces optimizers which are LM algorithms that can tune the prompts and weights of LM calls. 

# Project Organization
The repo contains only the notebooks directory which includes three (3) notebooks for:
- Training a Retrieval model on a custom dataset
- Creating Indexes from the trained retrieval model
- Building a compiled Baleen RAG with Optimizer

These notebooks can easily be converted to Python scripts to build an interactive DSPy RAG application.

## Getting Started
To reproduce this repo you need to:
1. Create a new Python or conda environment using
shell
```
conda create -n <environment-name> python=3.11 -y
```
2. install the required packages by running the command below
shell
```
pip install dspy-ai openai rich ragatouille
```
3. create a `data` directory to store the trained retrieval model and indexes
