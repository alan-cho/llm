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

    # Print config parameters as JSON
    print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main() 