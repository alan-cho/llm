import argparse
import torch
import json
import os
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
    parser.add_argument('--model_id', type=str, required=True, help='Hugging Face model ID')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face access token (optional)')
    parser.add_argument('--context_size', type=int, required=True, help='Context window size (e.g., 2048, 4096)')
    parser.add_argument('--quant_format', type=str, required=True, choices=['gguf', 'exl2'], help='Quantization format')
    parser.add_argument('--gguf_quant_preset', type=str, help='GGUF quantization preset (required if quant_format=gguf)')
    parser.add_argument('--gguf_batch_size', type=int, help='GGUF batch size (required if quant_format=gguf)')
    parser.add_argument('--exl2_bpw', type=float, help='EXL2 bits per weight (required if quant_format=exl2)')
    parser.add_argument('--exl2_kv_cache_bits', type=int, help='EXL2 kv_cache_bits (required if quant_format=exl2)')
    args = parser.parse_args()

    warnings = []
    model_kwargs = {}
    if args.hf_token:
        model_kwargs['token'] = args.hf_token
        os.environ['HF_TOKEN'] = args.hf_token

    # Quantization logic (simulate, as transformers does not support gguf/exl2 directly)
    quant_desc = None
    if args.quant_format == 'gguf':
        if not args.gguf_quant_preset or not args.gguf_batch_size:
            raise ValueError('gguf_quant_preset and gguf_batch_size are required for quant_format=gguf')
        quant_desc = f"GGUF preset: {args.gguf_quant_preset}, batch size: {args.gguf_batch_size}"
    elif args.quant_format == 'exl2':
        if args.exl2_bpw is None or args.exl2_kv_cache_bits is None:
            raise ValueError('exl2_bpw and exl2_kv_cache_bits are required for quant_format=exl2')
        quant_desc = f"EXL2 bpw: {args.exl2_bpw}, kv_cache_bits: {args.exl2_kv_cache_bits}"

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **model_kwargs)
    config = AutoConfig.from_pretrained(args.model_id, **model_kwargs)

    # Load model (simulate quantization by dtype if possible)
    model_dtype = torch.float32
    if args.quant_format == 'gguf' and '4' in (args.gguf_quant_preset or ''):
        model_dtype = torch.float16
    elif args.quant_format == 'exl2' and args.exl2_bpw and args.exl2_bpw <= 8:
        model_dtype = torch.float16
    # Note: True quantization would require special loaders, not supported in transformers directly

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="mps" if torch.backends.mps.is_available() else "auto",
        torch_dtype=model_dtype,
        **model_kwargs
    )

    # Prepare pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Estimate model size (weights)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_gb = model_size_bytes / (1024 ** 3)

    # Estimate context size (attention cache, buffers, etc)
    # This is a rough estimate: batch_size=1, context_size tokens, 2 layers of cache per layer
    num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 0))
    hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 0))
    if not num_layers or not hidden_size:
        warnings.append('Could not determine num_layers or hidden_size from config.')
    # Each token in the context window needs 2 * hidden_size * num_layers * dtype_size bytes
    dtype_size = torch.tensor([], dtype=model_dtype).element_size()
    context_size_bytes = args.context_size * num_layers * hidden_size * dtype_size * 2  # 2 for key+value
    context_size_gb = context_size_bytes / (1024 ** 3)

    # Run inference to measure actual VRAM usage
    prompt = tokenizer.decode(tokenizer(["Hello"])[0][:1])
    vram_before = get_vram_usage()
    _ = pipe(prompt, max_new_tokens=args.context_size)
    vram_after = get_vram_usage()
    vram_used_gb = (vram_after - vram_before) / 1024  # MB to GB
    if vram_used_gb < 0:
        vram_used_gb = 0
    if vram_used_gb == 0:
        warnings.append("VRAM usage reporting on Apple Silicon (MPS) may be inaccurate. Use Activity Monitor for more accurate results.")

    # Output as JSON
    output = {
        "model_size_gb": round(model_size_gb, 4),
        "context_size_gb": round(context_size_gb, 4),
        "total_vram_gb": round(model_size_gb + context_size_gb, 4),
        "actual_vram_used_gb": round(vram_used_gb, 4),
        "quantization": quant_desc,
        "warnings": warnings
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main() 