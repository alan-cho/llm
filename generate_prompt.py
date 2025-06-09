import json

# Each word is about 1 token, so 128,000 tokens ≈ 128,000 words.
# We'll use a repeated sentence of 16 words.
sentence = (
    "Explain planes, cars, bikes, and trains in detail. "
)
words_per_sentence = len(sentence.split())
repeat_count = 15000 // words_per_sentence + 1

long_prompt = " ".join([sentence] * repeat_count)
long_prompt = long_prompt[:512000]  # Truncate to ~128k tokens (assuming 4 chars/token)

prompt_json = {
    "role": "user",
    "content": long_prompt
}

with open("prompt.json", "w") as f:
    json.dump(prompt_json, f)
