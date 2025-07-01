import os
import sys
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.targon.com/v1",
    api_key=os.getenv("TARGON_API_KEY")
)

def run_bubble_sort_test():
    """Test 1: Bubble sort implementation"""
    print("=== Test 1: Bubble Sort Implementation ===")
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
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

def run_function_calling_test():
    """Test 2: Function calling"""
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
                       help='Test to run: 1 for bubble sort, 2 for function calling')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.test == 1:
        run_bubble_sort_test()
    elif args.test == 2:
        run_function_calling_test()
    elif args.all:
        run_bubble_sort_test()
        run_function_calling_test()
    else:
        # Interactive mode
        print("Available tests:")
        print("1. Bubble sort implementation")
        print("2. Function calling")
        print("3. Run all tests")
        
        try:
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            if choice == "1":
                run_bubble_sort_test()
            elif choice == "2":
                run_function_calling_test()
            elif choice == "3":
                run_bubble_sort_test()
                run_function_calling_test()
            else:
                print("Invalid choice. Please run with --help for usage information.")
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()