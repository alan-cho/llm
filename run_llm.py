import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline

def get_vram_usage():
    # For Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        try:
            # Try the more accurate driver_allocated_memory if available
            return torch.mps.driver_allocated_memory() / (1024 ** 2)  # MB
        except AttributeError:
            # Fallback to current_allocated_memory
            return torch.mps.current_allocated_memory() / (1024 ** 2)  # MB
    # For CUDA (not used here, but for completeness)
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    else:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Hugging Face model name')
    parser.add_argument('--quant', type=str, default=None, choices=['int8', 'float16', None], help='Quantization type')
    args = parser.parse_args()

    print(f"Loading model: {args.model} with quantization: {args.quant}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load model config
    config = AutoConfig.from_pretrained(args.model)

    # Quantization config
    quantization_config = None
    if args.quant == 'int8':
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == 'float16':
        quantization_config = {'torch_dtype': torch.float16}

    # Load model
    if args.quant == 'int8':
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="mps",
            quantization_config=quantization_config
        )
    elif args.quant == 'float16':
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="mps",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="mps"
        )

    # Prepare pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Measure VRAM before
    vram_before = get_vram_usage()

    # Run inference
    prompt = "Hello, my name is"
    output = pipe(prompt, max_new_tokens=20)
    print("Output:", output[0]['generated_text'])

    # Measure VRAM after
    vram_after = get_vram_usage()
    vram_used = vram_after - vram_before
    print(f"VRAM used: {vram_used:.2f} MB")
    if vram_used == 0:
        print("Note: VRAM usage reporting on Apple Silicon (MPS) may be inaccurate. Use Activity Monitor for more accurate results.")

if __name__ == "__main__":
    main() 