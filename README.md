# Speculative Reasoning with LLMs 🌐

See `docs/prompt.md` for an overview of the prompts used.

### Installation
Create a virtual environment and install the `requirements.txt` packages. Conda example is given below.
Note that we are using python=3.10.0.
```console
conda create --prefix env python=3.10.0
conda activate env/
pip install -r requirements.txt
```

### Run the models
`main.py` is the entry point of the algorithm. `classes` has all necessary function and definitions. If you run OpenAI models, please place the API keys in an `.env` file. See the script for details.