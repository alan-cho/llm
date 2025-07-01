import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.targon.com/v1",
    api_key=os.getenv("TARGON_API_KEY")
)

const_model = "deepseek-ai/DeepSeek-R1"

def run_base_case():
    """Test 1: Base Case"""
    print("=== Test 1: Base Case ===")
    try:
        response = client.chat.completions.create(
            model=const_model,
            stream=True,
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": "Write a bubble sort implementation in Python with comments explaining how it works"}
            ],
        )
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
        print("\n")
    except Exception as e:
        print(f"Error: {e}")

def run_function_calling_case():
    """Test 2: Function Calling"""
    print("\n=== Test 2: Function Calling ===")
    def get_weather(location, unit="celsius"):
        """Get the current weather for a location"""
        return f"Weather in {location}: 22°{unit.upper()[0]} and sunny"

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            stream=False,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can call functions."},
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The temperature unit to use"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "get_weather":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    
                    result = get_weather(**args)
                    
                    response2 = client.chat.completions.create(
                        model="deepseek-ai/DeepSeek-R1",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that can call functions."},
                            {"role": "user", "content": "What's the weather like in San Francisco?"},
                            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                            {"role": "tool", "content": result, "tool_call_id": tool_call.id}
                        ]
                    )
                    
                    print(response2.choices[0].message.content)
        else:
            print(message.content)
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Targon API with different features')
    parser.add_argument('--test', type=int, choices=[1, 2], 
                       help='Test to run: 1 for base case, 2 for function calling')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.test == 1:
        run_base_case()
    elif args.test == 2:
        run_function_calling_case()
    elif args.all:
        run_base_case()
        run_function_calling_case()
    else:
        print("Available tests:")
        print("1. Base Case")
        print("2. Function Calling")
        print("3. Run all tests")
        
        try:
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            if choice == "1":
                run_base_case()
            elif choice == "2":
                run_function_calling_case()
            elif choice == "3":
                run_base_case()
                run_function_calling_case()
            else:
                print("Invalid choice. Please run with --help for usage information.")
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()