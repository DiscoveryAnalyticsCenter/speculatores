# Codebase for the LLM Analytical Reasoning Paper in IEEE Big Data 2024

:male_detective: Fun Fact :male_detective: The name of our repo, *Speculatores*, is a nod to the ancient roman military intelligence agency with the same name. In this repo and associated paper, we describe how we are augmenting LLMs to improve analytical reasoning over multiple documents, within the context of intelligence analysis.

### Installation
Create a virtual environment and install the `requirements.txt` packages. Conda example is given below.
Note that we are using python=3.10.0.
```console
conda create --prefix env python=3.10.0
conda activate env/
pip install -r requirements.txt
```

### Run the models
`main.py` is the entry point of the algorithm. `classes` has all necessary function and definitions. If you run OpenAI models, please place the API keys in an `.env` file. See the script for details. Please refer to `docs/prompt.md` for an overview of the prompts used.