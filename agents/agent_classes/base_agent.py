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
        func: callable=None
        ):
        
        '''
        BaseAgent is the parent class for the ActionPlanningAgent, KnowledgeAugmentedPromptAgent, and EvaluationAgent.
        This is the base initiation method.
        
        Args:
        openai_instance (OpenAI): an OpenAI client
        instructions (str): the instructions for the agent, i.e., the system prompt
        name (str): name of the agent
        description (str): description of the agent
        func (callable): optional function to be called by the agent.
        '''
        
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
        temperature: float=0.
        ) -> str:
        
        '''
        Makes a call to OpenAI's responses API and returns the output_text of the response object
        
        Args:
        input (str): the user prompt
        model (str): the OpenAI model to be used
        temperature (float): value for how creative the response should be; note that this is not available for gpt-5
        
        Returns:
        string representing the output from the OpenAI response
        '''

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
