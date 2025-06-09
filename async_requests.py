import asyncio
import aiohttp
import json
import os

async def send_request(session, url, headers, data):
    async with session.post(url, headers=headers, json=data) as response:
        return await response.text()

async def main():
    url = "http://103.196.86.136:30010/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Targon-Model": "deepseek-ai/DeepSeek-R1-trt",
        "Authorization": f"Bearer {os.getenv('API_KEY')}"
    }

    # Read the user prompt from JSON file
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
        "max_tokens": 10000,
        "top_p": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    # Number of requests to send
    num_requests = 5  # Change this to the number of requests you want to send

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, headers, data) for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses):
        print(f"Response {i + 1}: {response}")

if __name__ == "__main__":
    asyncio.run(main())
