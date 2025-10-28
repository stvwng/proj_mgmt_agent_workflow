# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from .base_agent import BaseAgent

# TODO: use chroma for vector db

class RAGKnowledgePromptAgent(BaseAgent):
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(
        self, 
        openai_instance: OpenAI, 
        instructions: str,
        chromadb_instance:  chromadb.PersistentClient=None,
        chunk_size: int=2000, 
        chunk_overlap: int=100
        ):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_instance (OpenAI): OpenAI client
        instructions (str): Instructions for the agent, i.e., system prompt.
        chromadb_instance (chromadb.PersistentClient): ChromaDB client for vector search
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        super().__init__(openai_instance=openai_instance, instructions=instructions)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_instance = openai_instance
        self.chromadb_instance = chromadb_instance
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(
        self, 
        text: str, 
        model: str="text-embedding-3-large", 
        encoding_format: str="float"
        ) -> List[float]:
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.
        model (str): OpenAI embedding model to use
        encoding_format (str): Format of the embeddings

        Returns:
        list: The embedding vector.
        """
        response = self.openai_instance.embeddings.create(
            model=model,
            input=text,
            encoding_format=encoding_format
        )
        return response.data[0].embedding

    def calculate_similarity(
        self, 
        vector_one: List[float], 
        vector_two: List[float]
        ) -> float:
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(
        self, 
        text: str
        ) -> List[Dict]:
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })
            
            if end == len(text):
                break
            else:
                start = end - self.chunk_overlap
                chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self) -> pd.DataFrame:
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def get_response_text(
        self, 
        input: str, 
        model: str="gpt-3.5-turbo", 
        temperature: float=0.
        ) -> str:
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        input (str): User input prompt.
        model (str): OpenAI model
        temperature (float): temperature for OpenAI model

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(input)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']
        
        if model == "gpt-5":
            temperature = 0

        try:
            response = self.openai_instance.responses.create(
                model=model,
                # instructions=f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                instructions=self.instructions,
                input=f"Answer based only on this information: {best_chunk}. Prompt: {input}",
                temperature=temperature
            )

            # return response.choices[0].message.content
            return response.output_text
        except Exception as e:
            return f"Response failed: {e}"