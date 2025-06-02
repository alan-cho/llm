# LLM VRAM Usage Tester

This script lets you run any Hugging Face transformer model locally on Apple Silicon (M1/M2/M3), test different quantization types, and measure VRAM usage.

## Setup

1. **Create and activate a Python virtual environment:**

```sh
python3 -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies:**

```sh
pip install -r requirements.txt
```

## Usage

Run the script with your desired model and quantization type:

```sh
python run_llm.py --model <huggingface-model-name> --quant <quant-type>
```

- `<huggingface-model-name>`: e.g. `meta-llama/Llama-2-7b-hf`, `gpt2`, etc.
- `<quant-type>`: `int8`, `float16`, or omit for default.

**Example:**

```sh
python run_llm.py --model meta-llama/Llama-2-7b-hf --quant int8
```

## Notes

- Not all models support all quantization types. If you get an error, try a different quant or model.
- For Apple Silicon, the script uses the MPS backend.
