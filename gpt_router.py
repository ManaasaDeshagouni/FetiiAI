# Enhanced GPT Router 3.0 - Using OpenAI Function Calling
# This version uses OpenAI's native function calling instead of JSON parsing

import json
import os
import time
import random
from typing import Dict, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI
from function_registry import FUNCTION_REGISTRY

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Convert function registry to OpenAI function calling format
def get_openai_functions():
    """Convert our function registry to OpenAI function calling format"""
    functions = []
    
    for func_name, func_info in FUNCTION_REGISTRY.items():
        # Build properties for the function
        properties = {}
        required = []
        
        for param_name, param_desc in func_info["signature"].items():
            # Extract type from description
            if "str" in param_desc:
                param_type = "string"
            elif "int" in param_desc:
                param_type = "integer"
            else:
                param_type = "string"
            
            properties[param_name] = {
                "type": param_type,
                "description": param_desc
            }
            required.append(param_name)
        
        function_def = {
            "name": func_name,
            "description": func_info["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
        functions.append(function_def)
    
    return functions

# Cache for repeated questions
_route_cache = {}


def _retry_call(fn, max_retries: int = 3, base_delay: float = 0.6):
    """Retry with exponential backoff and jitter."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.2)
            time.sleep(delay)
    if last_err:
        raise last_err


def _structured_error(message: str, user_fallback: str) -> Dict[str, Any]:
    return {
        "type": "natural_response",
        "function": None,
        "parameters": {},
        "thought": message,
        "confidence": 0.1,
        "response": user_fallback,
        "suggestions": [
            "Try rephrasing your question",
            "Ask about a different day or location",
            "Ask about busiest times or quietest days"
        ]
    }


def enhanced_gpt_route_v3(user_question: str, conversation_history: list = None) -> Optional[Dict[str, Any]]:
    """
    Enhanced GPT Router 3.0 using OpenAI function calling with conversation context
    """
    if not client:
        # Return a structured message instead of None so UI can handle clearly
        return _structured_error(
            "OPENAI client not initialized (missing API key)",
            "I'm temporarily unavailable. Please set the API key and try again."
        )

    # Check cache first (but include conversation context in cache key)
    cache_key = f"{user_question.lower().strip()}_{len(conversation_history or [])}"
    if cache_key in _route_cache:
        return _route_cache[cache_key]

    try:
        # Get available functions
        functions = get_openai_functions()
        
        # Build conversation messages
        messages = [
            {
                "role": "system", 
                "content": """You are Assistentee, a friendly and intelligent rideshare data assistant for Austin, Texas. 

You help users understand ride patterns, demographics, and trends through natural conversation and smart data analysis.

Your approach:
1. Understand the user's intent - Even if they use vague, slang, or indirect language
2. Reason through the problem - Think step-by-step about what they're really asking  
3. Choose the best function - If one exists that can help, use it with appropriate parameters
4. Provide friendly responses - Always be conversational and helpful
5. Offer suggestions - Guide users to better questions when appropriate
6. Remember conversation context - If the user is answering a clarifying question, combine it with previous context

Be flexible with language - understand slang, typos, and casual speech. If no function fits perfectly, provide a helpful natural response.

IMPORTANT: If the user is providing a follow-up answer (like "Friday" after you asked "Which day?"), combine it with the previous question to understand the full intent."""
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for entry in conversation_history[-6:]:  # Keep last 6 exchanges for context
                if isinstance(entry, tuple) and len(entry) == 2:
                    messages.append({"role": "user", "content": entry[0]})
                    messages.append({"role": "assistant", "content": entry[1]})
        
        # Add current question
        messages.append({"role": "user", "content": user_question})
        
        def _call_openai():
            return client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=functions,
                function_call="auto",
                temperature=0.3,
                timeout=30,
            )
        
        response = _retry_call(_call_openai)
        message = response.choices[0].message
        
        # Check if GPT wants to call a function
        if message.function_call:
            func_name = message.function_call.name
            try:
                func_args = json.loads(message.function_call.arguments)
            except Exception:
                func_args = {}
            
            # Validate the function exists
            if func_name in FUNCTION_REGISTRY:
                result = {
                    "type": "function_call",
                    "function": func_name,
                    "parameters": func_args,
                    "thought": f"User asked: '{user_question}'. I'm calling {func_name} to analyze this data.",
                    "confidence": 0.9,
                    "response": f"Let me analyze that for you using {func_name.replace('_', ' ')}!",
                    "suggestions": [
                        "What about other days?",
                        "How about different age groups?", 
                        "What are the busiest times?"
                    ]
                }
                _route_cache[cache_key] = result
                return result
        
        # If no function call, provide natural response
        natural_response = {
            "type": "natural_response",
            "function": None,
            "parameters": {},
            "thought": f"User asked: '{user_question}'. This doesn't require specific data analysis.",
            "confidence": 0.8,
            "response": message.content or "I'm here to help you analyze Fetii rideshare data! What would you like to know?",
            "suggestions": [
                "What are the most popular dropoff locations?",
                "When do large groups ride most?",
                "What's the age distribution of riders?"
            ]
        }
        _route_cache[cache_key] = natural_response
        return natural_response

    except Exception as e:
        error_response = _structured_error(
            f"GPT Router v3 Error: {e}",
            "Sorry, I'm having trouble processing that right now. Please try again in a moment."
        )
        _route_cache[cache_key] = error_response
        return error_response
