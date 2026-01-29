from langchain_openai import ChatOpenAI
from typing import List, Type, TypeVar, Union
from pydantic import BaseModel, Field
import asyncio
import os

from dotenv import load_dotenv

load_dotenv(override=True)


T = TypeVar('T', bound=BaseModel)

MAX_CONCURRENCY = 50
MAX_LLM_CALLS = 100

llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_API_BASE"),
    temperature=0,
)

class TextResponse(BaseModel):
    """Simple text response from LLM."""
    response: str = Field(description="The LLM's response text")

async def call_llm_async(
    prompts: List[str],
    schema: Type[BaseModel],
) -> List[T]:
    """
    Call llms with structured output
    
    Args:
        prompts: List of prompts to send to the LLM
        schema: Pydantic BaseModel class defining the output structure
        
    Returns:
        List of instances of the schema class with LLM outputs
        
    Raises:
        ValueError: If batch limit is exceeded
    """
    if len(prompts) > MAX_LLM_CALLS:
        raise ValueError(f"Number of prompts ({len(prompts)}) exceeds maximum allowed per batch ({MAX_LLM_CALLS})")
    
    llm_with_structure = llm.with_structured_output(schema).with_retry(stop_after_attempt=3)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async def process_single(prompt: str) -> T:
        """Process a single prompt with semaphore control."""
        async with semaphore:
            result = await llm_with_structure.ainvoke(prompt)
            return result
    
    results = await asyncio.gather(*[process_single(prompt) for prompt in prompts])
    return results


def call_llm(
    prompts: Union[str, List[str]],
    schema: Type[BaseModel] = None,
) -> Union[str, List[str], BaseModel, List[BaseModel]]:
    """
    Synchronous wrapper for LLM calls. Supports both single and batch prompts.
    
    Args:
        prompts: Single prompt string or list of prompts
        schema: Optional Pydantic BaseModel class. If None, returns plain text responses.
        
    Returns:
        - If prompts is a string and schema is None: returns string
        - If prompts is a string and schema is provided: returns schema instance
        - If prompts is a list and schema is None: returns list of strings
        - If prompts is a list and schema is provided: returns list of schema instances
        
    Examples:
        # Simple text response
        response = call_llm("What is 2+2?")
        print(response)  # "4"
        
        # Batch text responses
        responses = call_llm(["What is 2+2?", "What is 3+3?"])
        print(responses)  # ["4", "6"]
        
        # Structured response
        class Analysis(BaseModel):
            pattern: str
            confidence: float
        
        result = call_llm("Analyze this...", schema=Analysis)
        print(result.pattern)
        
        # Batch structured responses
        results = call_llm(["Analyze A", "Analyze B"], schema=Analysis)
        for r in results:
            print(r.pattern)
    """
    is_single = isinstance(prompts, str)
    prompt_list = [prompts] if is_single else prompts
    use_schema = schema if schema is not None else TextResponse
    results = asyncio.run(call_llm_async(prompt_list, use_schema))
    if schema is None:
        results = [r.response for r in results]
    return results[0] if is_single else results

