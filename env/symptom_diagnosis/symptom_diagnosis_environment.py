"""
Symptom Diagnosis Environment - One-step inference.

Single LLM call with context retrieved by get_context interface.

Interface:
- get_context(symptoms: str) -> str
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

DISEASE_LIST = [
    "drug reaction", "allergy", "chicken pox", "diabetes", "psoriasis",
    "hypertension", "cervical spondylosis", "bronchial asthma", "varicose veins",
    "malaria", "dengue", "arthritis", "impetigo", "fungal infection", "common cold",
    "gastroesophageal reflux disease", "urinary tract infection", "typhoid",
    "pneumonia", "peptic ulcer disease", "jaundice", "migraine"
]


class SymptomDiagnosisEnvironment(TaskEnvironment):
    """One-step diagnosis: context retrieval → LLM inference → diagnosis."""
    
    def get_interface_signatures(self) -> List[InterfaceSignature]:
        return [
            InterfaceSignature(
                name="get_context",
                inputs=[("symptoms", "str", "Patient symptom description")],
                output=("str", "Relevant medical context for diagnosis"),
                description="Return context that helps the LLM diagnose based on symptoms."
            )
        ]
    
    def get_task_instruction(self) -> str:
        return f"""Symptom Diagnosis Task (One-Step)

Medical diagnosis from patient symptoms. Single LLM call with context.

Diseases: {', '.join(DISEASE_LIST)}

Curate context files in context/ folder and implement get_context(symptoms) to provide diagnosis guidelines and knowledge.

IMPORTANT: In your interface implementation, use ABSOLUTE paths to access context files 
(e.g., use the full path like /path/to/workspace/iter1_sub0/context/diseases.md).
"""
    
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
        
        # Step 1: Get context (use empty string if interface not available)
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
            # No interface available (first training round) - use empty context
            context = ""
            print(f"[Context] No interface, using empty context", flush=True)
            trajectory.append({"step": "get_context", "output": "(no interface)", "note": "Using empty context"})
        
        # Step 2: LLM inference
        prompt = self._build_diagnosis_prompt(sample.question, context)
        try:
            response = await llm_client.ainvoke(prompt)
            diagnosis = self._extract_diagnosis(response)
            trajectory.append({
                "step": "llm_inference",
                "prompt_length": len(prompt),
                "response": response[:500],  # Truncate for logging
                "diagnosis": diagnosis,
            })
        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return EnvironmentResult(
                feedback=f"LLM inference error: {e}",
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 0.0},
                trajectory=trajectory + [{"step": "llm_inference", "error": str(e)}]
            )
        
        # Step 3: Evaluate
        is_correct = self._normalize(diagnosis) == self._normalize(sample.ground_truth)
        trajectory.append({
            "step": "evaluate",
            "prediction": diagnosis,
            "ground_truth": sample.ground_truth,
            "is_correct": is_correct,
        })

        return EnvironmentResult(
            feedback=f"Predicted: {diagnosis}, Expected: {sample.ground_truth}",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if is_correct else 0.0},
            trajectory=trajectory
        )
    
    def _build_diagnosis_prompt(self, symptoms: str, context: str) -> str:
        context_section = f"\n{context}\n" if context else ""
        return f"""You are a medical diagnostician.
{context_section}
Patient symptoms: {symptoms}

Possible diagnoses: {', '.join(DISEASE_LIST)}

Analyze and provide diagnosis in format: [DIAGNOSIS]disease_name[/DIAGNOSIS]"""
    
    def _extract_diagnosis(self, response: str) -> str:
        match = re.search(r'\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback
        match = re.search(r'(?:diagnosis|conclusion)[:：]\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip().split('\n')[-1]
    
    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip()).rstrip('.!?')
    
    def format_result_for_training(self, item: Dict[str, Any]) -> Dict[str, Any]:
        sample = item.get("sample", {})
        evaluation = item.get("evaluation", {})
        trajectory = evaluation.get("trajectory", [])
        metrics = evaluation.get("metrics", {})
        
        # Extract diagnosis from trajectory
        diagnosis = ""
        context_output = ""
        for step in trajectory:
            if step.get("step") == "get_context":
                context_output = step.get("output", "")
            elif step.get("step") == "llm_inference":
                diagnosis = step.get("diagnosis", "")
            elif step.get("step") == "evaluate":
                diagnosis = diagnosis or step.get("prediction", "")
        
        return {
            "id": sample.get("id"),
            "symptoms": sample.get("question"),
            "ground_truth": sample.get("ground_truth"),
            "llm_prediction": diagnosis,
            "is_correct": metrics.get("accuracy", 0.0) == 1.0,
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
