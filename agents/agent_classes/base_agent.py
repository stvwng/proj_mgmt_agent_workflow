import numpy as np
from openai import OpenAI

MODELS_WITHOUT_TEMPERATURE = {"gpt-5"}

class BaseAgent:
    def __init__(
        self, 
        openai_instance: OpenAI, 
        instructions: str=None,
        name: str=None,
        description: str=None,
        func: callable=None):
        
        self.openai_instance = openai_instance
        self.instructions = instructions
        self.agent_dict = {
            "name": name,
            "description": description,
            "func": func
        }
        
    def get_response_text(
        self, 
        input: str, 
        model: str="gpt-3.5-turbo", 
        temperature: float=0.):
        # Generate a response using the OpenAI API
        try:
            if model in MODELS_WITHOUT_TEMPERATURE:
                response = self.openai_instance.responses.create(
                    model=model,
                    instructions=self.instructions,
                    input=input
                )
            else:
                response = self.openai_instance.responses.create(
                    model=model,
                    input=input,
                    instructions=self.instructions,
                    temperature=temperature
                )
            return response.output_text
        except Exception as e:
            print(f"Response failed: {e}")
