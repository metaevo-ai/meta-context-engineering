# MCE Task Environments

This tutorial covers the implemented environments and guides you through creating custom ones.

## Overview

MCE environments define how tasks are evaluated. The core abstraction is **signature-driven**: environments declare `InterfaceSignature` specs that agents must implement, and MCE validates and injects them.

Three environment patterns are supported:
1. **Single Interface** - One context retrieval function (simplest)
2. **Multi Interface** - Multiple functions for staged reasoning
3. **Agentic** - No interfaces; agent explores context files autonomously

---

## Implemented Environments

### 1. `symptom_diagnosis` — Single Interface

The simplest pattern: one interface, one LLM call.

**Flow:** `get_context(symptoms)` → LLM inference → diagnosis

```python
def get_interface_signatures(self) -> List[InterfaceSignature]:
    return [
        InterfaceSignature(
            name="get_context",
            inputs=[("symptoms", "str", "Patient symptom description")],
            output=("str", "Relevant medical context for diagnosis"),
            description="Return context that helps the LLM diagnose based on symptoms."
        )
    ]
```

**Evaluation pipeline:**
1. Call `get_context(sample.question)` to retrieve context
2. Build prompt with context + symptoms
3. LLM makes diagnosis
4. Compare against ground truth

**Use when:** Single-step reasoning, classification, information retrieval.

---

### 2. `symptom_diagnosis_twostep` — Multi Interface

Two-stage reasoning with specialized interfaces for each stage.

**Flow:** `get_narrowing_context(symptoms)` → LLM narrows candidates → `get_diagnosis_context(symptoms, candidates)` → LLM final diagnosis

```python
def get_interface_signatures(self) -> List[InterfaceSignature]:
    return [
        InterfaceSignature(
            name="get_narrowing_context",
            inputs=[("symptoms", "str", "Patient symptom description")],
            output=("str", "Context for narrowing down candidate diseases"),
            description="Provide context to help LLM identify 3-5 most likely diseases."
        ),
        InterfaceSignature(
            name="get_diagnosis_context",
            inputs=[
                ("symptoms", "str", "Patient symptom description"),
                ("candidates", "str", "Comma-separated candidate diseases")
            ],
            output=("str", "Context for final diagnosis from candidates"),
            description="Provide context to help LLM select final diagnosis from candidates."
        ),
    ]
```

**Evaluation pipeline:**
1. Call `get_narrowing_context(symptoms)` for stage 1 context
2. LLM identifies 3-5 candidate diseases
3. Call `get_diagnosis_context(symptoms, candidates)` for stage 2 context
4. LLM selects final diagnosis
5. Compare against ground truth

**Use when:** Multi-step workflows, chain-of-thought tasks, refinement pipelines.

---

### 3. `symptom_diagnosis_agent` — Agentic (No Interface)

Fully autonomous agent that explores context files using tools.

**Flow:** Agent given working directory → explores context/ → reasons → diagnosis

```python
def get_interface_signatures(self) -> List[InterfaceSignature]:
    return []  # Empty - no interfaces
```

**Evaluation pipeline:**
1. Agent receives working directory path (context/ folder)
2. Agent uses file tools (Read, Glob, etc.) to explore context
3. Agent reasons through multiple turns
4. Agent outputs final diagnosis
5. Compare against ground truth

**Use when:** Complex reasoning, tool-using agents, tasks requiring exploration.

---

## Creating a New Environment

### Step 1: Directory Structure

```
env/
└── my_task/
    ├── __init__.py                 # Empty file
    ├── my_task_environment.py      # Environment implementation
    └── data/
        ├── train.jsonl             # Training data
        └── val.jsonl               # Validation data
```

### Step 2: Data Format

Standard JSONL with `question` and `answer` fields (minimum):

```jsonl
{"question": "Input text here", "answer": "expected_output"}
{"question": "Another input", "answer": "another_expected", "extra_field": "optional"}
```

### Step 3: Environment Implementation

```python
"""
My Task Environment.

Interface:
- get_context(input: str) -> str
"""

import json
import random
import re
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from env.base import (
    InterfaceSignature,
    Sample,
    EnvironmentResult,
    TaskEnvironment,
)

logger = logging.getLogger(__name__)


class MyTaskEnvironment(TaskEnvironment):
    """Description of your task."""
    
    # -------------------------------------------------------------------------
    # Required: Define interfaces
    # -------------------------------------------------------------------------
    
    def get_interface_signatures(self) -> List[InterfaceSignature]:
        return [
            InterfaceSignature(
                name="get_context",
                inputs=[("input", "str", "The task input")],
                output=("str", "Relevant context"),
                description="Return context that helps complete the task."
            )
        ]
    
    # -------------------------------------------------------------------------
    # Required: Task instruction for agent
    # -------------------------------------------------------------------------
    
    def get_task_instruction(self) -> str:
        return """My Task Description

Brief explanation of what the agent should do.

Curate context files in context/ folder and implement get_context(input) 
to retrieve relevant information.

IMPORTANT: Use ABSOLUTE paths to access context files in your implementation.
"""
    
    # -------------------------------------------------------------------------
    # Required: Evaluation logic
    # -------------------------------------------------------------------------
    
    async def aevaluate(
        self,
        sample: Sample,
        interfaces: Dict[str, Callable],
        llm_client: Any = None,
        context_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> EnvironmentResult:
        assert llm_client is not None, "LLM client is required"
        trajectory = []
        
        # Get context (handle missing interface gracefully - first round has none)
        get_context = interfaces.get("get_context") if interfaces else None
        
        if get_context:
            try:
                context = get_context(sample.question)
                print(f"[Context] Retrieved {len(context)} chars", flush=True)
                trajectory.append({
                    "step": "get_context",
                    "input": sample.question,
                    "output": context,
                })
            except Exception as e:
                logger.warning(f"get_context error: {e}")
                context = ""
                trajectory.append({"step": "get_context", "error": str(e)})
        else:
            context = ""
            print(f"[Context] No interface, using empty", flush=True)
            trajectory.append({"step": "get_context", "output": "(no interface)"})
        
        # LLM inference
        prompt = self._build_prompt(sample.question, context)
        try:
            response = await llm_client.ainvoke(prompt)
            answer = self._extract_answer(response)
            trajectory.append({
                "step": "llm_inference",
                "response": response[:500],
                "answer": answer,
            })
        except Exception as e:
            return EnvironmentResult(
                feedback=f"LLM error: {e}",
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 0.0},
                trajectory=trajectory + [{"step": "llm_inference", "error": str(e)}]
            )
        
        # Evaluate
        is_correct = self._normalize(answer) == self._normalize(sample.ground_truth)
        trajectory.append({
            "step": "evaluate",
            "prediction": answer,
            "ground_truth": sample.ground_truth,
            "is_correct": is_correct,
        })
        
        return EnvironmentResult(
            feedback=f"Predicted: {answer}, Expected: {sample.ground_truth}",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if is_correct else 0.0},
            trajectory=trajectory
        )
    
    # -------------------------------------------------------------------------
    # Required: Data loading
    # -------------------------------------------------------------------------
    
    def load_samples(
        self, 
        path: str, 
        limit: int = 10, 
        random_sample: bool = False,
        shuffle: bool = False
    ) -> List[Sample]:
        samples = []
        with open(path, encoding="utf-8") as f:
            for i, row in enumerate(f):
                if not random_sample and limit and i >= limit:
                    break
                data = json.loads(row)
                samples.append(Sample(
                    id=i,
                    question=data["question"],
                    ground_truth=data["answer"],
                    extras={}  # Add domain-specific fields here
                ))
        
        if random_sample and limit and len(samples) > limit:
            samples = random.sample(samples, limit)
        if shuffle:
            random.shuffle(samples)
        return samples
    
    # -------------------------------------------------------------------------
    # Required: Format results for training rollouts
    # -------------------------------------------------------------------------
    
    def format_result_for_training(self, item: Dict[str, Any]) -> Dict[str, Any]:
        sample = item.get("sample", {})
        evaluation = item.get("evaluation", {})
        trajectory = evaluation.get("trajectory", [])
        metrics = evaluation.get("metrics", {})
        
        answer = ""
        for step in trajectory:
            if step.get("step") == "llm_inference":
                answer = step.get("answer", "")
        
        return {
            "id": sample.get("id"),
            "question": sample.get("question"),
            "ground_truth": sample.get("ground_truth"),
            "prediction": answer,
            "is_correct": metrics.get("accuracy", 0.0) == 1.0,
        }
    
    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    
    def _build_prompt(self, question: str, context: str) -> str:
        context_section = f"\n{context}\n" if context else ""
        return f"""You are an expert assistant.
{context_section}
Input: {question}

Provide your answer in format: [ANSWER]your_answer[/ANSWER]"""
    
    def _extract_answer(self, response: str) -> str:
        match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip().split('\n')[-1]
    
    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip()).rstrip('.!?')
```

### Step 4: Register Environment

Edit `env/registry.py`:

```python
from env.my_task.my_task_environment import MyTaskEnvironment

EnvironmentRegistry.register("my_task", MyTaskEnvironment)
```

### Step 5: Training Script

```bash
#!/bin/bash

uv run python -m mce.main \
    --workspace "workspace/my_task" \
    --env "my_task" \
    --train-data "env/my_task/data/train.jsonl" \
    --val-data "env/my_task/data/val.jsonl" \
    --model "deepseek/deepseek-chat-v3.1" \
    --iterations 3 \
    --train-limit 50 \
    --val-limit 20 \
    --train-batch-size 25 \
    --log-dir "logs/my_task"
```

---

## Key Implementation Details

### Handling Missing Interfaces

First training round has no interfaces (agent hasn't implemented them yet):

```python
# WRONG - will crash
context = interfaces["get_context"](query)

# CORRECT - handle missing interface
get_context = interfaces.get("get_context") if interfaces else None
if get_context:
    context = get_context(query)
else:
    context = ""  # Fallback
```

### LLM Client API

Use `ainvoke()`:

```python
response = await llm_client.ainvoke(prompt)
```

---

## Optional Overrides

### Custom Metric Name

```python
def get_primary_metric_name(self) -> str:
    return "f1_score"  # Instead of default "accuracy"
```

---

## Quick Reference

| Method | Required | Description |
|--------|----------|-------------|
| `get_interface_signatures()` | Yes | Define interfaces agent must implement |
| `get_task_instruction()` | Yes | Brief task description |
| `aevaluate()` | Yes | Evaluation logic |
| `load_samples()` | Yes | Data loading |
| `format_result_for_training()` | Yes | Format results for training |
| `get_primary_metric_name()` | No | Metric name (default: "accuracy") |

---

## Testing

```python
# scripts/test_my_env.py
import asyncio
from env.registry import EnvironmentRegistry
from mce.llm_client import LLMClient

async def main():
    env = EnvironmentRegistry.get("my_task")
    llm = LLMClient(model="deepseek/deepseek-chat-v3.1")
    
    samples = env.load_samples("env/my_task/data/val.jsonl", limit=3)
    print(f"Loaded {len(samples)} samples")
    
    for sample in samples:
        result = await env.aevaluate(sample, interfaces={}, llm_client=llm)
        print(f"Sample {sample.id}: {result.metrics}")

asyncio.run(main())
```

Run:
```bash
uv run python scripts/test_my_env.py
```

---

## Checklist

Before training:

- [ ] Environment extends `TaskEnvironment`
- [ ] `get_interface_signatures()` returns list (or empty for agentic)
- [ ] `get_task_instruction()` mentions absolute paths
- [ ] `aevaluate()` handles missing interfaces gracefully
- [ ] `load_samples()` parses JSONL correctly
- [ ] Registered in `env/registry.py`
- [ ] Training data exists
