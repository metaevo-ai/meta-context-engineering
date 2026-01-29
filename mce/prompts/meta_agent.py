"""
Meta agent prompt builder.

The meta-agent evolves skills across iterations. It needs to understand
the interface signatures that the base-agent must implement.
"""

from pathlib import Path
import json
import re
from typing import List, Optional

from env.base import InterfaceSignature


def build_meta_agent_prompt(
    task_instruction: str,
    interface_signatures: List[InterfaceSignature],
    iter_dir: str,
    workspace_base: str,
) -> str:
    """
    Build the meta agent prompt.
    
    Args:
        task_instruction: Task-specific instruction from env
        interface_signatures: Required interface signatures
        iter_dir: Iteration directory path
        workspace_base: Workspace base directory path
        
    Returns:
        Formatted prompt string
    """
    iter_name = Path(iter_dir).name
    iter_part = iter_name.split("iter")[1]
    current_iteration = int(iter_part.split("_")[0])
    
    skill_database = _build_skill_database(workspace_base, current_iteration)
    interface_section = _build_interface_section(interface_signatures)
    
    skill_output_path = f"{workspace_base}/{iter_name}/.claude/skills/learning-context/SKILL.md"
    
    return f"""# Meta-Level Agent: Skill Evolution for Context Engineering

## Task Overview

{task_instruction}

{interface_section}

## Your Role

You are a **meta-level agent** that evolves context engineering skills across iterations. Your goal is to design self-contained skills that teach a base agent how to learn optimal task-specific context from training data.

Each skill you create should be a complete learning procedure that can be understood and executed independently.

## Architecture

**Meta-Level (You)**:
- Analyze iteration history (skills → implementations → results)
- Perform agentic crossover to evolve better skills
- Output: `{skill_output_path}`

**Base-Level (Context Engineer)**:
- Receives your skill + training data + prior best context
- Executes the skill to implement required interfaces
- Output: `interfaces/` functions + `context/` files

**Key Flow**: Base-agent starts with the BEST context from previous iterations and UPDATES it based on your skill's instructions.

## Working Directory

**Working Directory**: `{workspace_base}`

```
{workspace_base}/
  meta_agent/                                # Reference data (read-only)
    train.jsonl                              # Full training dataset
    evaluations.json                         # Aggregated metrics per iteration
    skills/iter*/SKILL.md                    # Archived skills from each iteration
  {iter_name}/                               # YOUR OUTPUT DIRECTORY
    .claude/skills/learning-context/SKILL.md # <-- WRITE YOUR SKILL HERE
    context/                                  # Static resources (base-agent writes)
    interfaces/                               # Implemented interfaces (base-agent writes)
    data/train.json                           # Training results
```

**Write Access**: Only `{workspace_base}/{iter_name}/.claude/skills/`
**IMPORTANT**: Write SKILL.md to `{skill_output_path}`

## Skill Database (Iteration History)

{skill_database}

## Your Task

1. **Review Iteration History**: 
   - Read `meta_agent/evaluations.json` for performance metrics
   - Read skills from `meta_agent/skills/iter*/SKILL.md`
   - Analyze: What strategies worked? What failed?
   - **Overfitting Check**: Is train >> val accuracy?
   - **Underfitting Check**: Are both accuracies low?

2. **Agentic Crossover**: Combine successful elements, address failures, innovate

3. **Evolve Skill**: Design a skill that guides the base-agent

## Skill Examples

### Example Skill A: Direct Agentic Curation

```markdown
## Skill Overview
Directly analyze training data and curate context in a fully agentic manner.

## Methodology
1. **Load prior context**: Read existing `context/` files
2. **Scan evaluation results**: Load `data/train.json`
3. **Analyze incorrect patterns**: Group by mistake type
4. **Update context incrementally**: ADD/UPDATE/REMOVE sections
5. **Implement interfaces**: Create functions in `interfaces/`

## Key Principles
- Build upon existing context
- Prioritize high-impact patterns
- Focus on generalizable patterns
```

### Example Skill B: LLM-Assisted Reflection

```markdown
## Skill Overview
Use LLM calls for structured reflection on incorrect predictions.

## Methodology
1. **Load existing context**
2. **Load training results**
3. **Reflect on errors**: Call LLM to analyze each incorrect sample
4. **Curate insights incrementally**
5. **Implement interfaces** based on learned patterns

## Implementation Hint
```python
from utils.llm import call_llm
reflection = call_llm(f"Model answered '{{answer}}' but correct is '{{target}}'. What was missing?")
```
```

## Output Requirements

**Write SKILL.md to**: `{skill_output_path}`

Requirements:
- MUST include `## Skill Overview` section
- Describe a complete learning procedure
- NO iteration-specific references
- Mention useful utilities (`utils/llm.py`, `utils/embedding.py`)
- Include clear methodology and implementation guidance

**Before finishing, verify**:
- SKILL.md exists at `{skill_output_path}`
- SKILL.md has a clear `## Skill Overview` section

Begin by analyzing the skill database and evolving the next generation skill.
"""


def _build_interface_section(signatures: List[InterfaceSignature]) -> str:
    """Build interface signatures section for meta-agent."""
    if not signatures:
        return "## Required Interfaces\n\nNo specific interfaces defined."
    
    lines = [
        "## Required Interfaces",
        "",
        "The base-agent must implement these interfaces:",
        "",
    ]
    
    for sig in signatures:
        inputs_str = ", ".join(f"{name}: {typ}" for name, typ, _ in sig.inputs)
        lines.append(f"- `{sig.name}({inputs_str}) -> {sig.output[0]}`: {sig.description}")
    
    lines.append("")
    lines.append("Your skill should guide the base-agent in implementing these effectively.")
    
    return "\n".join(lines)


def _build_skill_database(workspace_base: str, current_iteration: int) -> str:
    """Build a summary of the skill database (history of all previous iterations)."""
    if current_iteration == 0:
        return "No previous iterations (this is iteration 0). Design an initial skill based on the task."
    
    if current_iteration == 1:
        return "No previous iterations (this is iteration 1, iter0 is baseline). Design an initial skill based on the task."
    
    workspace_base = Path(workspace_base)
    meta_agent_dir = workspace_base / "meta_agent"
    
    evaluations_file = meta_agent_dir / "evaluations.json"
    if not evaluations_file.exists():
        raise FileNotFoundError(
            f"Evaluations file not found at {evaluations_file}. "
            "This file should have been created by previous iterations."
        )
    
    with open(evaluations_file) as f:
        evaluations = json.load(f)
    
    database_entries = []
    
    for i in range(1, current_iteration):
        iter_key = f"iter{i}"
        
        if iter_key not in evaluations:
            continue
        
        iter_data = evaluations[iter_key]
        
        train_metrics = iter_data.get('train_metrics', {})
        val_metrics = iter_data.get('val_metrics', {})
        
        primary_metric_name = next(iter(val_metrics)) if val_metrics else "accuracy"
        
        train_value = iter_data.get(f'train_{primary_metric_name}')
        val_value = iter_data.get(f'val_{primary_metric_name}')
        
        assert train_value is not None, f"train_{primary_metric_name} missing for {iter_key}"
        assert val_value is not None, f"val_{primary_metric_name} missing for {iter_key}"
        
        train_str = f"{train_value:.2%}"
        val_str = f"{val_value:.2%}"
        
        metrics_display = f"**Train**: {train_str} | **Val**: {val_str}"
        
        skill_file = meta_agent_dir / "skills" / iter_key / "SKILL.md"
        skill_overview = _extract_skill_overview(skill_file)
        
        num_sub_iters = iter_data.get('num_sub_iters', 1)
        total_rollouts = iter_data.get('total_rollouts', 0)
        last_sub_folder = iter_data.get('last_sub_folder', f'iter{i}')

        entry = f"""### Iteration {i}
- {metrics_display}
- **Rollouts**: {total_rollouts} ({num_sub_iters} sub-iteration{"s" if num_sub_iters > 1 else ""})
- **Skill Overview**:
{skill_overview}
- **Files**: `meta_agent/skills/iter{i}/SKILL.md`, `{last_sub_folder}/`"""
        database_entries.append(entry)
    
    if not database_entries:
        return "No previous iterations available (only baseline iter0 exists)."
    
    return "\n\n".join(database_entries)


def _extract_skill_overview(skill_path: Path) -> str:
    """Extract the '## Skill Overview' section from SKILL.md."""
    if not skill_path.exists():
        return "  (SKILL.md not found)"
    
    try:
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"  (error reading file: {e})"
    
    pattern = r'^##\s*Skill\s+Overview\s*$'
    match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
    
    if not match:
        return "  (no '## Skill Overview' section found)"
    
    start_pos = match.end()
    next_match = re.search(r'\n##\s+[^#]', content[start_pos:])
    
    if next_match:
        overview_content = content[start_pos:start_pos + next_match.start()].strip()
    else:
        overview_content = content[start_pos:].strip()
    
    if not overview_content:
        return "  (Skill Overview section is empty)"
    
    return "\n".join(f"  {line}" if line.strip() else "" for line in overview_content.split("\n"))


if __name__ == "__main__":
    # Example usage
    from env.base import InterfaceSignature
    
    sigs = [
        InterfaceSignature(
            name="get_context",
            inputs=[("question", "str", "The question")],
            output=("str", "Context string"),
            description="Return relevant context."
        )
    ]
    
    print(build_meta_agent_prompt(
        task_instruction="Example task",
        interface_signatures=sigs,
        iter_dir="/workspace/iter2",
        workspace_base="/workspace"
    ))
