"""
Symptom Diagnosis Environment - Two-step workflow.

Step 1: LLM narrows down to candidate diseases
Step 2: LLM makes final diagnosis from candidates

Interfaces:
- get_narrowing_context(symptoms: str) -> str
- get_diagnosis_context(symptoms: str, candidates: str) -> str
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


class SymptomDiagnosisTwostepEnvironment(TaskEnvironment):
    """Two-step diagnosis: narrow candidates → final diagnosis."""
    
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
    
    def get_task_instruction(self) -> str:
        return f"""Symptom Diagnosis Task (Two-Step)

Medical diagnosis with two-stage reasoning:
  Step 1: Narrow down to 3-5 candidate diseases
  Step 2: Make final diagnosis from candidates

Diseases: {', '.join(DISEASE_LIST)}

Implement two context interfaces for each reasoning stage.

IMPORTANT: In your interface implementation, use ABSOLUTE paths to access context files.
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
        
        # Get interfaces (use None if not available)
        get_narrowing_ctx = interfaces.get("get_narrowing_context") if interfaces else None
        get_diagnosis_ctx = interfaces.get("get_diagnosis_context") if interfaces else None
        
        # === Step 1: Narrow down candidates ===
        if get_narrowing_ctx:
            try:
                narrowing_context = get_narrowing_ctx(sample.question)
                print(f"[Context] Stage1: {len(narrowing_context)} chars", flush=True)
                trajectory.append({
                    "step": "get_narrowing_context",
                    "input": sample.question,
                    "output": narrowing_context,
                })
            except Exception as e:
                logger.warning(f"get_narrowing_context error: {e}")
                narrowing_context = ""
                trajectory.append({"step": "get_narrowing_context", "error": str(e)})
        else:
            narrowing_context = ""
            print(f"[Context] Stage1: No interface, using empty", flush=True)
            trajectory.append({"step": "get_narrowing_context", "output": "(no interface)", "note": "Using empty context"})
        
        # LLM call 1: Get candidates
        narrowing_prompt = self._build_narrowing_prompt(sample.question, narrowing_context)
        try:
            narrowing_response = await llm_client.ainvoke(narrowing_prompt)
            candidates = self._extract_candidates(narrowing_response)
            trajectory.append({
                "step": "llm_narrowing",
                "response": narrowing_response[:500],
                "candidates": candidates,
            })
        except Exception as e:
            logger.error(f"Narrowing LLM error: {e}")
            return EnvironmentResult(
                feedback=f"Narrowing LLM error: {e}",
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 0.0},
                trajectory=trajectory + [{"step": "llm_narrowing", "error": str(e)}]
            )
        
        # === Step 2: Final diagnosis ===
        candidates_str = ", ".join(candidates)
        if get_diagnosis_ctx:
            try:
                diagnosis_context = get_diagnosis_ctx(sample.question, candidates_str)
                print(f"[Context] Stage2: {len(diagnosis_context)} chars", flush=True)
                trajectory.append({
                    "step": "get_diagnosis_context",
                    "input": {"symptoms": sample.question, "candidates": candidates_str},
                    "output": diagnosis_context,
                })
            except Exception as e:
                logger.warning(f"get_diagnosis_context error: {e}")
                diagnosis_context = ""
                trajectory.append({"step": "get_diagnosis_context", "error": str(e)})
        else:
            diagnosis_context = ""
            print(f"[Context] Stage2: No interface, using empty", flush=True)
            trajectory.append({"step": "get_diagnosis_context", "output": "(no interface)", "note": "Using empty context"})
        
        # LLM call 2: Final diagnosis
        diagnosis_prompt = self._build_diagnosis_prompt(sample.question, candidates_str, diagnosis_context)
        try:
            diagnosis_response = await llm_client.ainvoke(diagnosis_prompt)
            diagnosis = self._extract_diagnosis(diagnosis_response)
            trajectory.append({
                "step": "llm_diagnosis",
                "response": diagnosis_response[:500],
                "diagnosis": diagnosis,
            })
        except Exception as e:
            logger.error(f"Diagnosis LLM error: {e}")
            return EnvironmentResult(
                feedback=f"Diagnosis LLM error: {e}",
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 0.0},
                trajectory=trajectory + [{"step": "llm_diagnosis", "error": str(e)}]
            )
        
        # === Evaluate ===
        is_correct = self._normalize(diagnosis) == self._normalize(sample.ground_truth)
        trajectory.append({
            "step": "evaluate",
            "prediction": diagnosis,
            "ground_truth": sample.ground_truth,
            "is_correct": is_correct,
        })
        
        return EnvironmentResult(
            feedback=f"Candidates: {candidates_str} → Final: {diagnosis} (Expected: {sample.ground_truth})",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if is_correct else 0.0},
            trajectory=trajectory
        )
    
    def _build_narrowing_prompt(self, symptoms: str, context: str) -> str:
        context_section = f"\n{context}\n" if context else ""
        return f"""You are a medical diagnostician. First narrow down possible diseases.
{context_section}
Patient symptoms: {symptoms}

All possible diseases: {', '.join(DISEASE_LIST)}

List 3-5 most likely diseases in format: [CANDIDATES]disease1, disease2, disease3[/CANDIDATES]"""
    
    def _build_diagnosis_prompt(self, symptoms: str, candidates: str, context: str) -> str:
        context_section = f"\n{context}\n" if context else ""
        return f"""You are a medical diagnostician. Make final diagnosis from candidates.
{context_section}
Patient symptoms: {symptoms}

Candidate diseases: {candidates}

Select ONE final diagnosis in format: [DIAGNOSIS]disease_name[/DIAGNOSIS]"""
    
    def _extract_candidates(self, response: str) -> List[str]:
        match = re.search(r'\[CANDIDATES\](.*?)\[/CANDIDATES\]', response, re.IGNORECASE | re.DOTALL)
        if match:
            return [c.strip() for c in match.group(1).split(',')]
        # Fallback: look for numbered list
        matches = re.findall(r'\d+\.\s*([^\n,]+)', response)
        if matches:
            return [m.strip() for m in matches[:5]]
        return DISEASE_LIST[:3]  # Default fallback
    
    def _extract_diagnosis(self, response: str) -> str:
        match = re.search(r'\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
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
        
        candidates = []
        diagnosis = ""
        
        for step in trajectory:
            if step.get("step") == "llm_narrowing":
                candidates = step.get("candidates", [])
            elif step.get("step") == "llm_diagnosis":
                diagnosis = step.get("diagnosis", "")
        
        return {
            "id": sample.get("id"),
            "symptoms": sample.get("question"),
            "ground_truth": sample.get("ground_truth"),
            "candidates": candidates,
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
