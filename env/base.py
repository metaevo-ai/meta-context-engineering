"""
Base classes and interfaces for MCE task environments.

MCE is signature-driven: tasks define InterfaceSignature specs,
MCE injects them into agent prompts, validates implementations,
and returns callables to the task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# Interface Signature System
# =============================================================================

@dataclass
class InterfaceSignature:
    """
    Defines a callable interface that the agent must implement.
    
    This is the core abstraction for MCE's signature-driven approach.
    Tasks define what functions they need, MCE ensures agent implements them.
    
    Example:
        InterfaceSignature(
            name="get_context",
            inputs=[("question", "str", "The user's question")],
            output=("str", "Relevant context string"),
            description="Return context that helps answer the question."
        )
    """
    name: str                              # Function name (e.g., "get_context")
    inputs: List[Tuple[str, str, str]]     # [(param_name, type_hint, description), ...]
    output: Tuple[str, str]                # (return_type_hint, description)
    description: str                       # What this function should do
    
    def to_prompt(self) -> str:
        """Convert to human-readable format for agent prompt."""
        # Build function signature
        inputs_str = ", ".join(f"{name}: {typ}" for name, typ, _ in self.inputs)
        return_str = self.output[0]
        
        # Build docstring params
        params_doc = ""
        if self.inputs:
            params_doc = "    Args:\n"
            for name, typ, desc in self.inputs:
                params_doc += f"        {name} ({typ}): {desc}\n"
        
        returns_doc = f"    Returns:\n        {self.output[0]}: {self.output[1]}\n"
        
        return f"""### `{self.name}`

{self.description}

```python
def {self.name}({inputs_str}) -> {return_str}:
    \"\"\"
{params_doc}{returns_doc}    \"\"\"
    # Your implementation here
    pass
```"""
    
    def to_stub(self) -> str:
        """Generate minimal Python stub."""
        inputs_str = ", ".join(f"{name}: {typ}" for name, typ, _ in self.inputs)
        return f"def {self.name}({inputs_str}) -> {self.output[0]}:\n    ..."
    
    def get_expected_params(self) -> List[str]:
        """Return list of expected parameter names."""
        return [name for name, _, _ in self.inputs]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Sample:
    """
    Single task instance.
    
    Core fields are defined, additional fields stored in extras.
    This allows environments to include domain-specific data.
    """
    id: int
    question: str
    context: str = ""
    ground_truth: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary with all fields flattened."""
        result = {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "ground_truth": self.ground_truth,
        }
        result.update(self.extras)
        return result


@dataclass
class EnvironmentResult:
    """Feedback returned by task environment after evaluation."""
    feedback: str
    ground_truth: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)  # Multi-step interaction history


# =============================================================================
# Task Environment Interface
# =============================================================================

class TaskEnvironment(ABC):
    """
    Abstract interface for task environments.
    
    Each task environment defines:
    1. Interface signatures - what callables the agent must implement
    2. Data loading - how to load samples
    3. Evaluation - how to evaluate agent outputs
    
    The task handles all pipeline complexity internally.
    MCE only provides validated callables based on signatures.
    """

    @abstractmethod
    def get_interface_signatures(self) -> List[InterfaceSignature]:
        """
        Define the callable interfaces the agent must implement.
        
        This is the core method for MCE's signature-driven approach.
        
        Returns:
            List of InterfaceSignature objects defining required functions.
        
        Example:
            return [
                InterfaceSignature(
                    name="get_context",
                    inputs=[("question", "str", "The question")],
                    output=("str", "Relevant context"),
                    description="Return context for answering the question."
                )
            ]
        """
        pass

    @abstractmethod
    async def aevaluate(
        self, 
        sample: Sample, 
        interfaces: Dict[str, Callable],
        llm_client: Any = None,
        context_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> EnvironmentResult:
        """
        Evaluate a sample using the implemented interfaces.
        
        The task receives validated callables and uses them however needed.
        All pipeline complexity (multi-turn, multi-agent, etc.) is handled here.
        
        Args:
            sample: The input sample with question and context
            interfaces: Dict mapping interface names to callable functions
            llm_client: Optional LLM client for inference
            context_dir: Optional path to context/ folder (for agent environments)
            log_dir: Optional path to log directory (for storing agent trajectories)
        
        Returns:
            EnvironmentResult with feedback and metrics
        """
        pass

    @abstractmethod
    def load_samples(
        self, 
        path: str, 
        limit: int = 10, 
        random_sample: bool = False, 
        shuffle: bool = False
    ) -> List[Sample]:
        """
        Load samples from the benchmark.
        
        Args:
            path: Path to data file
            limit: Maximum samples to load
            random_sample: If True, randomly sample; else take first
            shuffle: If True, shuffle loaded samples
        
        Returns:
            List of Sample objects
        """
        pass
    
    def get_task_instruction(self) -> str:
        """
        Return the task instruction/description.
        
        Override to provide task-specific instructions.
        """
        return "Complete the task according to the interface requirements."
    
    def get_primary_metric_name(self) -> str:
        """
        Get name of the primary metric.
        
        Used for determining best iteration, logging, comparisons.
        Default: "accuracy"
        """
        return "accuracy"
    
    @abstractmethod
    def format_result_for_training(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format evaluation result for training rollouts.
        
        Override to customize fields in training rollouts.
        """
        sample = item.get("sample", {})
        llm_output = item.get("llm_output", {})
        evaluation = item.get("evaluation", {})
        metrics = evaluation.get("metrics", {})
        
        primary_metric = self.get_primary_metric_name()
        is_correct = metrics.get(primary_metric, 0.0) == 1.0
        
        return {
            "id": sample.get("id"),
            "question": sample.get("question"),
            "llm_answer": llm_output.get("final_answer"),
            "target": sample.get("ground_truth"),
            "is_correct": is_correct,
        }
