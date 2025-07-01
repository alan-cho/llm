import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# "Targon" or "OpenAI"
const_config_flag = "Targon"

TARGON_CONFIG = {
    "base_url": "https://api.targon.com/v1",
    "api_key": os.getenv("TARGON_API_KEY"),
    "model": "deepseek-ai/DeepSeek-V3-0324"
}

OPENAI_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-mini"
}

CONFIG = TARGON_CONFIG if const_config_flag == "Targon" else OPENAI_CONFIG

client = OpenAI(
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"]
)

def run_base_case():
    """Test 1: Base Case"""
    print("=== Test 1: Base Case ===")
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
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
            model=CONFIG["model"],
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
                        model=CONFIG["model"],
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

def run_parameter_case():
    """Test 3: Parameter Case"""
    print("\n=== Test 3: Parameter Case ===")
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            stream=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short story about robots."}
            ],
            # Parameters Here
        )
        
        full_response = ""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
        
        print("\n") 
    except Exception as e:
        print(f"Error: {e}")

def run_structured_output_case():
    """Test 4: Structured Output Case"""
    print("\n=== Test 4: Structured Output Case ===")
    
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            stream=False,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides structured information in JSON format."},
                {"role": "user", "content": "Analyze the following text and extract key information in JSON format: 'The movie Inception, directed by Christopher Nolan and released in 2010, stars Leonardo DiCaprio and Ellen Page. It's a science fiction thriller about dreams within dreams.'"}
            ],
            response_format={"type": "json_object"},
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "extract_movie_info",
                        "description": "Extract structured information about a movie from text",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title of the movie"
                                },
                                "director": {
                                    "type": "string", 
                                    "description": "The director of the movie"
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "The release year of the movie"
                                },
                                "genre": {
                                    "type": "string",
                                    "description": "The genre of the movie"
                                },
                                "main_actors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of main actors in the movie"
                                },
                                "plot_summary": {
                                    "type": "string",
                                    "description": "Brief summary of the movie plot"
                                }
                            },
                            "required": ["title", "director", "year", "genre", "main_actors", "plot_summary"]
                        }
                    }
                }
            ],
            tool_choice={"type": "function", "function": {"name": "extract_movie_info"}}
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "extract_movie_info":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    
                    print("=== Extracted Movie Information ===")
                    print(f"Title: {args.get('title', 'N/A')}")
                    print(f"Director: {args.get('director', 'N/A')}")
                    print(f"Year: {args.get('year', 'N/A')}")
                    print(f"Genre: {args.get('genre', 'N/A')}")
                    print(f"Main Actors: {', '.join(args.get('main_actors', []))}")
                    print(f"Plot Summary: {args.get('plot_summary', 'N/A')}")
                    
                    # Also show the raw JSON
                    print("\n=== Raw JSON Response ===")
                    print(json.dumps(args, indent=2))
        else:
            print("No structured output received")
            print(f"Response: {message.content}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Structured output may not be supported by this model/API")

def run_simple_structured_output():
    """Simple structured output test"""
    print("\n=== Simple Structured Output Test ===")
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            stream=False,
            messages=[
                {"role": "user", "content": "Create a JSON object with your name, age, and favorite color. Use realistic values and return it as JSON."}
            ],
            response_format={"type": "json_object"}
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Targon API with different features')
    parser.add_argument('--test', type=int, choices=[1, 2, 3, 4], 
                       help='Test to run: 1 for base case, 2 for function calling, 3 for parameter case, 4 for structured output')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.test == 1:
        run_base_case()
    elif args.test == 2:
        run_function_calling_case()
    elif args.test == 3:
        run_parameter_case()
    elif args.test == 4:
        run_structured_output_case()
    elif args.all:
        run_base_case()
        run_function_calling_case()
        run_parameter_case()
        run_structured_output_case()
    else:
        print("Available tests:")
        print("1. Base Case")
        print("2. Function Calling")
        print("3. Parameter Case")
        print("4. Structured Output")
        print("5. Run all tests")
        
        try:
            choice = input("Enter your choice (1, 2, 3, 4, or 5): ").strip()
            if choice == "1":
                run_base_case()
            elif choice == "2":
                run_function_calling_case()
            elif choice == "3":
                run_parameter_case()
            elif choice == "4":
                run_structured_output_case()
            elif choice == "5":
                run_base_case()
                run_function_calling_case()
                run_parameter_case()
                run_structured_output_case()
            else:
                print("Invalid choice. Please run with --help for usage information.")
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()