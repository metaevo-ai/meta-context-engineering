"""
Environment registry for MCE tasks.

Register custom environments here or use EnvironmentRegistry.register() directly.
"""

from typing import Dict, Type, Optional
from env.base import TaskEnvironment


class EnvironmentRegistry:
    """Registry for task environments."""
    
    _environments: Dict[str, Type[TaskEnvironment]] = {}
    
    @classmethod
    def register(cls, name: str, env_class: Type[TaskEnvironment]) -> None:
        """
        Register an environment class.
        
        Args:
            name: Unique name for the environment
            env_class: TaskEnvironment subclass
        """
        cls._environments[name] = env_class
    
    @classmethod
    def get(cls, name: str) -> TaskEnvironment:
        """
        Get an environment instance by name.
        
        Args:
            name: Registered environment name
            
        Returns:
            Instance of the environment
            
        Raises:
            ValueError: If environment not found
        """
        if name not in cls._environments:
            available = ", ".join(cls._environments.keys()) or "(none)"
            raise ValueError(
                f"Environment '{name}' not found. Available: {available}"
            )
        return cls._environments[name]()
    
    @classmethod
    def list_environments(cls) -> list:
        """List all registered environment names."""
        return list(cls._environments.keys())
    
    @classmethod
    def get_task_instruction(cls, name: str) -> str:
        """Get task instruction for an environment."""
        env = cls.get(name)
        return env.get_task_instruction()


# =============================================================================
# Register environments here
# =============================================================================

from env.symptom_diagnosis.symptom_diagnosis_environment import SymptomDiagnosisEnvironment
from env.symptom_diagnosis_twostep.symptom_diagnosis_twostep_environment import SymptomDiagnosisTwostepEnvironment
from env.symptom_diagnosis_agent.symptom_diagnosis_agent_environment import SymptomDiagnosisAgentEnvironment

EnvironmentRegistry.register("symptom_diagnosis", SymptomDiagnosisEnvironment)
EnvironmentRegistry.register("symptom_diagnosis_twostep", SymptomDiagnosisTwostepEnvironment)
EnvironmentRegistry.register("symptom_diagnosis_agent", SymptomDiagnosisAgentEnvironment)
