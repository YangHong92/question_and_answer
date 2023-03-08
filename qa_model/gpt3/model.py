from transformers import GPT2TokenizerFast
import openai
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv

with open('config.json') as json_file:
    config = json.load(json_file)

class Model(object):
    def __init__(self):
        self.COMPLETIONS_MODEL = config["COMPLETIONS_MODEL"]
        self.DOC_EMBEDDINGS_MODEL = f"text-search-{config['MODEL_NAME']}-doc-001"
        self.QUERY_EMBEDDINGS_MODEL = f"text-search-{config['MODEL_NAME']}-query-001"
        self.MAX_SECTION_LEN = config["MAX_SECTION_LEN"]
        self.MAX_COMPLETION_LEN = config["MAX_COMPLETION_LEN"]
        self.SEPARATOR = config["SEPARATOR"]
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.separator_len = len(tokenizer.tokenize(config["SEPARATOR"]))

        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.OPENAI_API_KEY
        self.TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM")

        self.document_embeddings = self.load_embeddings(config["EMBEDDING_PATH"])
        self.knowledge_base_text = self.load_knowledge_base_text(config["KNOWLEDGE_BASE_PATH"])
        self.COMPLETIONS_API_PARAMS = {
            "temperature": 0.0, # temperature of 0.0 gives the most factual answer, 1 gives the most creative answer.
            "max_tokens": self.MAX_COMPLETION_LEN, # the maximum number of tokens to generate in the completion/answer. The token count of prompt + max_tokens <= model's context length (usually 2048)
            "model": self.COMPLETIONS_MODEL,
        }
        

    def load_knowledge_base_text(self, fname: str) -> pd.DataFrame:
        df = pd.read_csv(fname)
        df = df.set_index(["title", "heading"])
        return df

    def load_embeddings(self, fname: str) -> Dict[Tuple[str, str], List[float]]:
        """
        Read the document embeddings and their keys from a CSV.
        
        fname is the path to a CSV with exactly these named columns: 
            "title", "heading", "0", "1", ... up to the length of the embedding vectors.
        """
    
        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
        return {
            (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }

    def get_embedding(self, text: str, model: str) -> List[float]:
        result = openai.Embedding.create(
            model=model,
            input=text
        )
        return result["data"][0]["embedding"]

    def get_query_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text, self.QUERY_EMBEDDINGS_MODEL)

    def vector_similarity(self, x: List[float], y: List[float]) -> float:
        """
        We could use cosine similarity or dot product to calculate the similarity between vectors.
        In practice, we have found it makes little difference. 
        """
        return np.dot(np.array(x), np.array(y))

    def order_document_sections_by_query_similarity(self, query: str, contexts: dict) -> list:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_query_embedding(query)
        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
        
        return document_similarities

    def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
        """
        Fetch relevant 
        """
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(question, context_embeddings)
        
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index]
            
            chosen_sections_len += document_section.tokens + self.separator_len
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break
                
            chosen_sections.append(self.SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
                
        # Useful diagnostic information
        # print(f"Selected {len(chosen_sections)} document sections:")
        # print("\n".join(chosen_sections_indexes))
        
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        
        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    def answer(self, query: str, show_prompt: bool = True) -> str:
        
        prompt = self.construct_prompt(
            query,
            self.document_embeddings,
            self.knowledge_base_text
        )
        
        if show_prompt:
            print(prompt)

        response = openai.Completion.create(
                    prompt=prompt,
                    **self.COMPLETIONS_API_PARAMS
                )

        return (
            response["choices"][0]["text"].strip(" \n")
        )

model = Model

# return the single instance of the inference model
def get_model():
    return model