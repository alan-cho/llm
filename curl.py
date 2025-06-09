import requests
import json
import os

url = "http://103.196.86.136:30010/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "X-Targon-Model": "deepseek-ai/DeepSeek-R1-trt",
    "Authorization": f"Bearer {os.getenv('API_KEY')}"
}

with open('prompt.json', 'r') as f:
    user_prompt = json.load(f)

data = {
    "model": "deepseek-ai/DeepSeek-R1-trt",
    "stream": True,
    "messages": [
        {"role": "system", "content": "You are a helpful programming assistant."},
        user_prompt
    ],
    "temperature": 0.7,
    "max_tokens": 128000,
    "top_p": 0.1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

# Make the request with streaming enabled
response = requests.post(url, headers=headers, json=data, stream=True)

# Process the streaming response
for line in response.iter_lines():
    if line:
        # Remove the "data: " prefix if present
        line = line.decode('utf-8')
        if line.startswith('data: '):
            line = line[6:]
        
        # Skip the "[DONE]" message
        if line == "[DONE]":
            continue
            
        try:
            # Parse the JSON response
            json_response = json.loads(line)
            if 'choices' in json_response and len(json_response['choices']) > 0:
                content = json_response['choices'][0].get('delta', {}).get('content', '')
                if content:
                    print(content, end='', flush=True)
        except json.JSONDecodeError:
            continue
