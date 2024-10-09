"""
Implements methods to call the LLM APIs.

Example openai API call:
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)
"""

import requests
from openai import OpenAI
import json

class LLMCalls:

    @staticmethod
    def call_openai_chat_completion(model, user_message, system_message="You are a helpful assistant"):
        client = OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content
    
    