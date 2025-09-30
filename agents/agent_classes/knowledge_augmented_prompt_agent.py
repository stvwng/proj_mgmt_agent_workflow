# KnowledgeAugmentedPromptAgent class definition
from .base_agent import BaseAgent
from openai import OpenAI

class KnowledgeAugmentedPromptAgent(BaseAgent):
    def __init__(
        self,
        openai_instance: OpenAI, 
        knowledge: str, 
        instructions:str="You are a knowledge-based assistant.",
        name: str=None, 
        description: str=None, 
        func: callable=None):
        
        augmented_instructions = f"""
        {instructions} 
        Forget all previous content.
        Use only the following knowledge to answer, do not use your own knowledge:
        {knowledge}
        Answer the prompt based on this knowledge, not your own.
        """
        super().__init__(openai_instance, augmented_instructions, name, description, func)