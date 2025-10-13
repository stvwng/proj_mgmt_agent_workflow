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
        
        '''
        Initialize a KnowledgeAugmentedPromptAgent instance. A KnowledgeAugmentedPromptAgent has its prompt
        augmented with instructions and knowledge provided by the user.
        
        Args:
        openai_instance (OpenAI): an OpenAI client
        knowledge (str): the knowledge that the agent should use to perform its task.
        instructions (str): The role that the agent should take on, i.e., the system prompt
        name (str): string identifying the agent
        description (str): string description of the agent
        func (callable): function that the agent can call.
        '''
        
        augmented_instructions = f"""
        {instructions} 
        Forget all previous content.
        Use only the following knowledge to answer, do not use your own knowledge:
        {knowledge}
        Answer the prompt based on this knowledge, not your own.
        """
        super().__init__(openai_instance, augmented_instructions, name, description, func)