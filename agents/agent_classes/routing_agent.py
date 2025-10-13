import numpy as np
from openai import OpenAI

class RoutingAgent():

    def __init__(self, openai_instance: OpenAI, agents: list):
        # Initialize the agent with given attributes
        self.openai_instance = openai_instance
        self.agents = agents
        '''
        self.agents is a list of dicts with information about available agents.
        The structure of each dict is:
        {
            "name": str,
            "description": str,
            "func": function
        }
        '''

    def get_embedding(self, text: str):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Args:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        response = self.openai_instance.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def route_prompt(self, user_input: str):
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)
            
            if similarity > best_score:
                best_agent = agent
                best_score = similarity

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)