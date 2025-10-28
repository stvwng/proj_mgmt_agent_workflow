import numpy as np
import pandas as pd
import re
import csv
import uuid
import os
from datetime import datetime
from openai import OpenAI
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from .base_agent import BaseAgent

class RAGAgent(BaseAgent):
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(
        self, 
        openai_instance: OpenAI, 
        instructions: str,
        chromadb_instance:  chromadb.PersistentClient=None,
        collection_name: str=None,
        ):
        """
        Initializes the RAGAgent OpenAI and ChromaDB clients, instructions, and collection

        Parameters:
        openai_instance (OpenAI): OpenAI client
        instructions (str): Instructions for the agent, i.e., system prompt.
        chromadb_instance (chromadb.PersistentClient): ChromaDB client for vector search
        collection_name (str): name of ChromaDB collection

        """
        super().__init__(openai_instance=openai_instance, instructions=instructions)
        self.chromadb_instance = chromadb_instance
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(os.getenv("OPENAI_API_KEY"))
        self.collection = self.chromadb_instance.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_function
            )
        
    def add_doc_to_collection(self, doc):
        pass
    
    def query_collection(self, query:str):
        pass