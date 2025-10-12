from openai import OpenAI
from .base_agent import BaseAgent

import re
from typing import List, Dict, Optional

class ActionPlanningAgent(BaseAgent):

    def __init__(self, openai_instance: OpenAI, knowledge: str):
        '''
        Initializes an ActionPlanningAgent instance. An ActionPlanningAgent creates a plan for other agents to execute, based on knowledge provided by the user.
        
        Args:
            openai_instance (OpenAI): an OpenAI client
            knowledge (str): Knowledge for the ActionPlanningAgent to use to create a plan.
        
        '''
        
        
        instructions = f"""You are an action planning agent.
        Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for.
        You return the steps as a list. Only return the steps in your knowledge. 
        Forget any previous context. This is your knowledge: {knowledge}"""

        super().__init__(openai_instance, instructions)
        
    def extract_steps(self, text: str, return_dict: bool = False) -> List:
        """
        Extract numbered or bulleted steps from a text string.
        
        Args:
            text: Input text containing steps (can have surrounding context)
            return_dict: If True, returns list of dicts with 'number' and 'content'.
                        If False, returns list of strings (content only)
            
        Returns:
            List of steps (either strings or dictionaries)
        """
        steps = []
        
        # Pattern matches various step formats with better handling of multi-line content
        # Matches: "1.", "1)", "Step 1:", "Step 1.", etc.
        pattern = r'^\s*(?:step\s*)?(\d+)[.:)\-]\s*(.+?)(?=^\s*(?:step\s*)?\d+[.:)\-]|\Z)'
        
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        for match in matches:
            number = match.group(1)
            content = match.group(2).strip()
            
            # Clean up the content (normalize whitespace but preserve structure)
            content = re.sub(r'\s+', ' ', content)
            # Remove any trailing punctuation or whitespace
            content = content.rstrip('.,;')
            
            if return_dict:
                steps.append({
                    'number': int(number),
                    'content': content
                })
            else:
                steps.append(content)
        
        # Fallback: try bullet points if no numbered steps found
        if not steps:
            bullet_pattern = r'^\s*([•\-\*])\s+(.+?)(?=^\s*[•\-\*]|\Z)'
            matches = re.finditer(bullet_pattern, text, re.MULTILINE | re.DOTALL)
            
            for i, match in enumerate(matches, 1):
                content = match.group(2).strip()
                content = re.sub(r'\s+', ' ', content)
                content = content.rstrip('.,;')
                
                if return_dict:
                    steps.append({
                        'number': i,
                        'content': content
                    })
                else:
                    steps.append(content)
        
        return steps


    def extract_steps_from_input(self, input: str) -> List:
        '''
        Get steps for plan to be executed by agents based on input (user prompt)
        
        Args:
            input (str): instructions from user, i.e., the user prompt
            
        Returns:
            list[str]: list of steps in plan as strings
        
        '''
        
        response_text = self.get_response_text(input)
        steps = self.extract_steps(response_text, False)
        return steps