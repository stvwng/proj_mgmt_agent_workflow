from openai import OpenAI
from .base_agent import BaseAgent

class ActionPlanningAgent(BaseAgent):

    def __init__(self, openai_instance: OpenAI, knowledge: str):
        instructions = f"""You are an action planning agent.
        Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for.
        You return the steps as a list. Only return the steps in your knowledge. 
        Forget any previous context. This is your knowledge: {knowledge}"""

        super().__init__(openai_instance, instructions)

    def extract_steps_from_input(self, input):
        response_text = self.get_response_text(input)

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        return steps