"""
Base agent prompt builder.

Injects interface signatures into the prompt for the agent to implement.
"""

from pathlib import Path
from typing import List, Optional

from env.base import InterfaceSignature


def build_base_agent_prompt(
    task_instruction: str,
    interface_signatures: List[InterfaceSignature],
    iter_dir: str,
    workspace_base: str = None,
    initial_prompt: str = None,
) -> str:
    """
    Build the base agent prompt with interface signatures.
    
    Args:
        task_instruction: Task-specific instruction from env
        interface_signatures: List of interfaces to implement
        iter_dir: Iteration directory path
        workspace_base: Base workspace directory
        initial_prompt: Optional additional prompt
        
    Returns:
        Formatted prompt string
    """
    iter_name = Path(iter_dir).name
    
    # Build interface section
    interface_section = _build_interface_section(interface_signatures)
    
    # Build directory structure based on whether interfaces are required
    if interface_signatures:
        dir_structure = f'''```
{iter_name}/
  .claude/skills/learning-context/SKILL.md  # Your skill guidance (MUST READ THIS)
  context/                                   # Write static resources here (knowledge, rules, etc.)
  interfaces/                                # Implement required interfaces here
    __init__.py                              # Export all functions
    <function_name>.py                       # One file per interface
  data/
    train.json                               # Training results to learn from
  utils/
    llm.py                                   # LLM calls (call_llm)
    embedding.py                             # Embeddings (compute_embedding_similarity)
```'''
    else:
        dir_structure = f'''```
{iter_name}/
  .claude/skills/learning-context/SKILL.md  # Your skill guidance (MUST READ THIS)
  context/                                   # Write static resources here (knowledge, rules, etc.)
  data/
    train.json                               # Training results to learn from
  utils/
    llm.py                                   # LLM calls (call_llm)
    embedding.py                             # Embeddings (compute_embedding_similarity)
```

**NOTE**: This task has NO interface requirements. Focus on curating high-quality context files in `context/`.'''
    
    prompt = f'''# Context Engineer

## Task Overview

{task_instruction}

{interface_section}

## Working Directory

**Working Directory**: `{iter_dir}`

Your directory contains:
{dir_structure}

**File Access**:
- Read/Write: Only files within `{iter_dir}/`
- You CANNOT access other directories

## Skill Guidance

**IMPORTANT**: Read `.claude/skills/learning-context/SKILL.md` for your learning methodology.

## Available Utilities - Expensive, Use Sparingly

```python
# LLM calls (use sparingly - expensive)
from utils.llm import call_llm

# Simple text responses
responses = call_llm(["Question 1?", "Question 2?"])
for r in responses:
    print(r)  # Each r is a string

# Structured response with Pydantic schema
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    pattern: str = Field(description="The identified pattern")
    confidence: float = Field(description="Confidence score 0-1")

results = call_llm(["Analyze A", "Analyze B"], schema=Analysis)
for r in results:
    print(r.pattern)

# Embeddings
from utils.embedding import compute_embedding_similarity
similarity_matrix = compute_embedding_similarity(
    ["text 1", "text 2"],
    ["text A", "text B"]
)
# Returns shape (len(a), len(b)) with cosine similarities
```

## Core Objective: Learn from Training Data

**CRITICAL**: Analyze `data/train.json` and curate context to fix incorrect predictions.

**IMPORTANT**: In most cases, high-quality context files are MORE impactful than complex interface logic.

- **Context files** (`context/`) provide knowledge, patterns, and guidelines that directly improve LLM reasoning'''
    
    if interface_signatures:
        prompt += '''
- **Interface functions** (`interfaces/`) can often be simple and naive while still achieving good results'''
    
    prompt += '''

### Training Data Analysis

1. **Load and inspect** `data/train.json`:
    - `summary`: Overall metrics
    - `detailed_results`: List of rollouts

2. **Analyze predictions**:
   - **Incorrect**: Why did it fail? What knowledge/pattern was missing?
   - **Correct**: What patterns led to success? How to reinforce?

3. '''
    
    if interface_signatures:
        prompt += f'''**Curate context and implement interfaces**:
   - Write context files in `context/` directory
   - Implement required functions in `interfaces/` directory
   - **IMPORTANT**: Use ABSOLUTE paths to access context files in your interface code:
     ```python
     # Good: absolute path
     context_path = "{iter_dir}/context/knowledge.md"
     
     # Bad: relative path (will fail)
     context_path = "context/knowledge.md"
     ```'''
    else:
        prompt += '''**Curate context**:
   - Write context files in `context/` directory'''
    
    prompt += '''

## Environment

Use `uv run python ...` for all Python execution.
'''
    
    if interface_signatures:
        prompt += '''
## Validation

The system will automatically validate your interface implementations.
If validation fails, you'll receive specific error messages to fix.
Just keep working until all interfaces are valid.
'''
    
    prompt += '''
Work efficiently: focus on impactful changes, avoid over-analysis, finish promptly.
'''
    
    if initial_prompt:
        prompt += f"\n\n## Additional Instructions\n\n{initial_prompt}"
    
    return prompt


def _build_interface_section(signatures: List[InterfaceSignature]) -> str:
    """Build the interface requirements section of the prompt."""
    if not signatures:
        return "## Required Interfaces\n\nNo specific interfaces required."
    
    lines = [
        "## Required Interfaces",
        "",
        "You MUST implement these interfaces in the `interfaces/` directory:",
        "",
    ]
    
    for sig in signatures:
        lines.append(sig.to_prompt())
        lines.append("")
    
    lines.extend([
        "### Output Structure",
        "",
        "```",
        "interfaces/",
        "  __init__.py          # Export all functions",
    ])
    
    for sig in signatures:
        lines.append(f"  {sig.name}.py         # Implementation of {sig.name}")
    
    lines.extend([
        "```",
        "",
        "**Example `interfaces/__init__.py`**:",
        "",
        "```python",
    ])
    
    for sig in signatures:
        lines.append(f"from .{sig.name} import {sig.name}")
    
    lines.append("")
    lines.append(f"__all__ = [{', '.join(repr(s.name) for s in signatures)}]")
    lines.append("```")
    
    return "\n".join(lines)
