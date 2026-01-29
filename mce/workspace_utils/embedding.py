from typing import List
import numpy as np
from numpy.typing import NDArray
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv

load_dotenv(override=True)

EMBEDDING_MODEL = "text-embedding-3-small"  # Cannot be changed

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_API_BASE"),
)

def compute_embedding_similarity(
    strings_a: List[str],
    strings_b: List[str],
) -> NDArray[np.float64]:
    """
    Compute cosine similarity between embeddings of two lists of strings.
    
    Args:
        strings_a: First list of strings
        strings_b: Second list of strings
        
    Returns:
        A 2D numpy array of shape (len(strings_a), len(strings_b)) containing
        cosine similarity scores between each pair of strings.
    """
    
    # Get embeddings for both lists
    embeddings_a = embeddings.embed_documents(strings_a)
    embeddings_b = embeddings.embed_documents(strings_b)
    
    # Convert to numpy arrays
    embeddings_a_np = np.array(embeddings_a)
    embeddings_b_np = np.array(embeddings_b)
    
    # Compute cosine similarity
    # Normalize the embeddings
    embeddings_a_norm = embeddings_a_np / np.linalg.norm(embeddings_a_np, axis=1, keepdims=True)
    embeddings_b_norm = embeddings_b_np / np.linalg.norm(embeddings_b_np, axis=1, keepdims=True)
    
    # Compute dot product (cosine similarity for normalized vectors)
    similarity_matrix = embeddings_a_norm @ embeddings_b_norm.T
    
    return similarity_matrix