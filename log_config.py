import argparse
import json
from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Hugging Face model ID')
    parser.add_argument('--hf_token', type=str, help='Hugging Face token (optional)')
    args = parser.parse_args()

    model_kwargs = {}
    if args.hf_token:
        model_kwargs['token'] = args.hf_token

    # Load config
    config = AutoConfig.from_pretrained(args.model_id, **model_kwargs)
    config_dict = config.to_dict()

    # Only keep fields that influence model size
    keys_of_interest = [
        'model_type',
        'vocab_size',
        'hidden_size',
        'num_attention_heads',
        'num_hidden_layers',
        'intermediate_size',
    ]
    filtered = {k: config_dict[k] for k in keys_of_interest if k in config_dict}

    print(json.dumps(filtered, indent=2))


if __name__ == "__main__":
    main() 