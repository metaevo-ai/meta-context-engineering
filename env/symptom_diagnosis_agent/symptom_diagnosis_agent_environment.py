"""
Symptom Diagnosis Environment - Fully Agentic.

The agent is given working directory with context/ folder and must:
1. Explore context/ folder to find relevant knowledge
2. Reason about symptoms using available tools
3. Make diagnosis through multi-turn interaction

Interface: NONE (pure static context optimization)
"""

import json
import random
import re
import logging
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from env.base import (
    InterfaceSignature,
    Sample,
    EnvironmentResult,
    TaskEnvironment,
)
from mce.logging_utils import setup_logger, log_message, MessageFormatter

logger = logging.getLogger(__name__)

DISEASE_LIST = [
    "drug reaction", "allergy", "chicken pox", "diabetes", "psoriasis",
    "hypertension", "cervical spondylosis", "bronchial asthma", "varicose veins",
    "malaria", "dengue", "arthritis", "impetigo", "fungal infection", "common cold",
    "gastroesophageal reflux disease", "urinary tract infection", "typhoid",
    "pneumonia", "peptic ulcer disease", "jaundice", "migraine"
]


class SymptomDiagnosisAgentEnvironment(TaskEnvironment):
    """Fully agentic diagnosis using Claude Agent SDK."""

    def get_interface_signatures(self) -> List[InterfaceSignature]:
        """No interfaces - agent reads static context files."""
        return []
    
    def get_task_instruction(self) -> str:
        return f"""Symptom Diagnosis Task (Agentic)

Medical diagnosis using a reasoning agent with file system access.

The agent will run with context/ folder as working directory and must:
- Explore context/ to find disease knowledge, symptom patterns, guidelines
- Read relevant files to inform diagnosis
- Reason about symptoms to make final diagnosis

Diseases: {', '.join(DISEASE_LIST)}

NO interface signatures - curate high-quality context files in context/ for the agent to discover.
"""
    
    async def aevaluate(
        self,
        sample: Sample,
        interfaces: Dict[str, Callable],
        llm_client: Any = None,
        context_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> EnvironmentResult:
        """Run agent to diagnose symptoms.
        
        Args:
            sample: Input sample
            interfaces: Not used (agent environment has no interfaces)
            llm_client: Not used (agent runs via SDK)
            context_dir: Path to context/ folder for agent to explore
            log_dir: Path to log directory for storing agent trajectories
        """
        trajectory = []
        conversation_messages = []  # Store full conversation for logging
        
        # Import here to avoid circular imports
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

        # Check context directory exists
        if not context_dir or not context_dir.exists():
            logger.warning(f"Context directory not found or not provided: {context_dir}")
            context_exists = False
        else:
            context_exists = True
            trajectory.append({
                "step": "context_check",
                "context_dir": str(context_dir),
                "exists": True,
            })
        
        # Build agent prompt (NO context content - agent must explore)
        agent_prompt = self._build_agent_prompt(sample.question, context_dir)
        trajectory.append({
            "step": "build_prompt",
            "prompt_length": len(agent_prompt),
        })
        
        # Run agent with context_dir as working directory
        try:
            diagnosis, agent_trajectory = await self._run_agent(
                prompt=agent_prompt,
                working_dir=context_dir if context_exists else None,
                sample_id=sample.id,
                log_dir=log_dir
            )
            trajectory.append({
                "step": "agent_run",
                "diagnosis": diagnosis,
                "agent_turns": len(agent_trajectory),
            })
            trajectory.extend(agent_trajectory)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return EnvironmentResult(
                feedback=f"Agent error: {e}",
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 0.0},
                trajectory=trajectory + [{"step": "agent_error", "error": str(e)}]
            )
        
        # Evaluate
        is_correct = self._normalize(diagnosis) == self._normalize(sample.ground_truth)
        trajectory.append({
            "step": "evaluate",
            "prediction": diagnosis,
            "ground_truth": sample.ground_truth,
            "is_correct": is_correct,
        })
        
        return EnvironmentResult(
            feedback=f"Agent diagnosis: {diagnosis}, Expected: {sample.ground_truth}",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if is_correct else 0.0},
            trajectory=trajectory
        )
    
    def _build_agent_prompt(self, symptoms: str, context_dir: Optional[Path]) -> str:
        """Build prompt for diagnosis agent - NO context provided, agent must explore."""
        working_dir_info = ""
        if context_dir and context_dir.exists():
            working_dir_info = f"""
## Working Directory

You are in: `{context_dir}`

This directory contains context files with medical knowledge. Use Read, Glob, or other tools to explore and find relevant information for diagnosis.
"""
        
        return f"""You are a medical diagnosis agent. Analyze the patient's symptoms and provide a diagnosis.

## Task

Diagnose the patient based on their symptoms. 
{working_dir_info}

## Patient Symptoms

{symptoms}

## Possible Diagnoses

{', '.join(DISEASE_LIST)}

## Instructions

1. If context files are available in the working directory, explore and read relevant files
2. Identify key symptoms from the patient description
3. Consider possible diseases based on symptom patterns
4. Narrow down to most likely diagnosis
5. Provide final diagnosis in format: [DIAGNOSIS]disease_name[/DIAGNOSIS]

Begin your analysis."""
    
    async def _run_agent(
        self, 
        prompt: str, 
        working_dir: Optional[Path],
        sample_id: int,
        log_dir: Optional[Path] = None
    ) -> tuple[str, List[Dict]]:
        """Run Claude agent with context directory as working directory.
        
        Returns:
            tuple: (diagnosis, agent_trajectory)
        """
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        
        agent_trajectory = []
        full_response = ""
        
        # Setup logger for this agent run if log_dir provided
        agent_logger = None
        file_formatter = None
        if log_dir:
            eval_log_dir = log_dir / "eval"
            eval_log_dir.mkdir(parents=True, exist_ok=True)
            agent_logger = setup_logger(
                name=f"agent_eval_sample_{sample_id}",
                log_dir=str(eval_log_dir),
                console_colors=False,
                minimal_console=True  # Suppress console output
            )
            file_formatter = MessageFormatter(use_colors=False, minimal=False)
            agent_logger.info(f"=== Agent Evaluation: Sample {sample_id} ===")
        
        # Print simple console message
        print(f"[Agent] Running diagnosis for sample {sample_id}...", end="", flush=True)
        
        # Set working directory to context folder
        options = ClaudeAgentOptions(
            max_turns=10,
            cwd=str(working_dir) if working_dir else None,
        )
        
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            
            async for message in client.receive_response():
                # Log message to file only (no console output)
                if agent_logger and file_formatter:
                    # Manually log to file without console output
                    formatted = file_formatter.format_message(message)
                    agent_logger.info(formatted)
                
                # Extract text for diagnosis
                if hasattr(message, 'content'):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            full_response += block.text
                            agent_trajectory.append({
                                "step": "agent_message",
                                "text": block.text[:500],
                            })
        
        # Extract diagnosis from response
        diagnosis = self._extract_diagnosis(full_response)
        
        # Print completion message
        print(f" Done (diagnosis: {diagnosis})", flush=True)
        
        if agent_logger:
            agent_logger.info(f"Diagnosis: {diagnosis}")
            # Clean up logger
            for handler in agent_logger.handlers[:]:
                handler.close()
                agent_logger.removeHandler(handler)
        
        return diagnosis, agent_trajectory
    
    def _extract_diagnosis(self, response: str) -> str:
        """Extract diagnosis from agent response."""
        match = re.search(r'\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'(?:diagnosis|conclusion)[:：]\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Last line fallback
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        return lines[-1] if lines else "unknown"
    
    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip()).rstrip('.!?')
    
    def format_result_for_training(self, item: Dict[str, Any]) -> Dict[str, Any]:
        sample = item.get("sample", {})
        evaluation = item.get("evaluation", {})
        trajectory = evaluation.get("trajectory", [])
        metrics = evaluation.get("metrics", {})
        
        diagnosis = ""
        agent_turns = 0
        for step in trajectory:
            if step.get("step") == "agent_run":
                diagnosis = step.get("diagnosis", "")
                agent_turns = step.get("agent_turns", 0)
        
        return {
            "id": sample.get("id"),
            "symptoms": sample.get("question"),
            "ground_truth": sample.get("ground_truth"),
            "prediction": diagnosis,
            "is_correct": metrics.get("accuracy", 0.0) == 1.0,
            "trajectory": [
                {"stage": "agent_reasoning", "turns": agent_turns, "diagnosis": diagnosis}
            ]
        }
    
    def load_samples(self, path: str, limit: int = 10, random_sample: bool = False, shuffle: bool = False) -> List[Sample]:
        samples = []
        with open(path, encoding="utf-8") as f:
            for i, row in enumerate(f):
                if not random_sample and limit and i >= limit:
                    break
                data = json.loads(row)
                samples.append(Sample(id=i, question=data["question"], ground_truth=data["answer"]))
        
        if random_sample and limit and len(samples) > limit:
            samples = random.sample(samples, limit)
        if shuffle:
            random.shuffle(samples)
        return samples
