"""
Workspace management utilities for MCE.

Handles creation and setup of iteration workspaces.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def compute_avg_metrics(successful_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute average metrics from successful evaluation results.
    
    Args:
        successful_results: List of result dicts with "evaluation.metrics" field
    
    Returns:
        Dictionary of averaged metrics
    """
    # Collect all metric values
    all_metrics_values = {}
    for r in successful_results:
        metrics = r.get("evaluation", {}).get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if metric_name not in all_metrics_values:
                all_metrics_values[metric_name] = []
            all_metrics_values[metric_name].append(metric_value)
    
    # Simple average for all metrics (only numeric values)
    avg_metrics = {}
    for metric_name, values in all_metrics_values.items():
        if not values:
            avg_metrics[metric_name] = 0.0
        elif isinstance(values[0], (int, float)):
            avg_metrics[metric_name] = sum(values) / len(values)

    return avg_metrics


def init_embeddings(
    model: str = "text-embedding-3-small",
    **kwargs
) -> OpenAIEmbeddings:
    """
    Initialize an embeddings model with API credentials.
    
    Args:
        model: Model name (default: "text-embedding-3-small")
        **kwargs: Additional arguments to pass to OpenAIEmbeddings
    
    Returns:
        Initialized OpenAIEmbeddings instance
    
    Raises:
        ValueError: If no API key is found
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENAI_API_BASE")
    
    if not api_key:
        raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
    
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )

# Helper function to ignore __pycache__
def ignore_pycache(directory, contents):
    return ['__pycache__'] if '__pycache__' in contents else []


def setup_meta_agent_reference(
    workspace_base: Path,
    train_data_path: str,
    train_limit: int = None,
    logger: logging.Logger = None,
) -> None:
    """
    Set up meta-agent reference data at the beginning of MCE loop.
    
    Creates meta_agent/ directory and copies training dataset (up to train_limit) for holistic task understanding.
    
    Args:
        workspace_base: Base workspace directory (e.g., workspace/finer)
        train_data_path: Path to training data file
        train_limit: Maximum number of training samples to copy (None = copy all)
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    workspace_base = Path(workspace_base)
    meta_agent_dir = workspace_base / "meta_agent"
    meta_agent_dir.mkdir(exist_ok=True)
    
    # Create skills directory
    skills_dir = meta_agent_dir / "skills"
    skills_dir.mkdir(exist_ok=True)
    
    # Copy training data for holistic understanding (only if not already exists)
    train_data_dest = meta_agent_dir / "train.jsonl"
    if not train_data_dest.exists():
        train_data_src = Path(train_data_path)
        if train_data_src.exists():
            # Copy all training samples (no limit)
            with open(train_data_src, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
            
            total_lines = len(lines)
            logger.info(f"✅ Copied all {total_lines} training samples to meta_agent/train.jsonl")
            
            with open(train_data_dest, 'w', encoding='utf-8') as f_out:
                f_out.writelines(lines)
        else:
            logger.warning(f"⚠️  Training data not found at {train_data_path}")
    else:
        logger.debug(f"meta_agent/train.jsonl already exists, skipping copy")


def aggregate_iteration_results(
    workspace_base: Path,
    iteration: int,
    sub_iterations: list,
    val_primary_metric: float,
    val_metrics: dict,
    val_total: int,
    cumulative_rollouts: int,
    num_sub_iters: int,
    last_sub_folder_name: str,
    batch_size: int,
    environment,
    logger: logging.Logger = None,
) -> None:
    """
    Aggregate iteration results to meta_agent/evaluations.json and copy skill.
    
    Uses the environment's primary metric (from get_primary_metric_name()) for
    aggregation and comparison across iterations.
    
    Args:
        workspace_base: Base workspace directory
        iteration: Iteration number
        sub_iterations: List of sub-iteration results with metrics
        val_primary_metric: Final validation primary metric value
        val_metrics: All validation metrics
        val_total: Total validation samples
        cumulative_rollouts: Total rollouts across all sub-iterations
        num_sub_iters: Number of sub-iterations
        last_sub_folder_name: Name of the last sub-iteration folder (e.g., "iter1_sub3")
        batch_size: Batch size used for sub-iterations
        environment: TaskEnvironment instance
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    workspace_base = Path(workspace_base)
    meta_agent_dir = workspace_base / "meta_agent"
    meta_agent_dir.mkdir(exist_ok=True)
    
    # Compute aggregated train metrics
    # Weight by batch size to get proper average
    total_weighted = sum(sr['batch_train_primary_metric'] * sr['batch_size'] for sr in sub_iterations)
    total_samples = sum(sr['batch_size'] for sr in sub_iterations)
    train_primary_metric = total_weighted / total_samples if total_samples > 0 else 0.0
    
    # Aggregate all train metrics (weighted average across batches)
    train_metrics = {}
    if sub_iterations and 'batch_train_metrics' in sub_iterations[0]:
        # Get all metric names from first sub-iteration
        all_metric_names = sub_iterations[0]['batch_train_metrics'].keys()
        for metric_name in all_metric_names:
            weighted_sum = sum(
                sr['batch_train_metrics'].get(metric_name, 0.0) * sr['batch_size']
                for sr in sub_iterations
            )
            train_metrics[metric_name] = weighted_sum / total_samples if total_samples > 0 else 0.0
    
    # Get primary metric name from environment
    primary_metric_name = environment.get_primary_metric_name()
    
    # Load existing evaluations or create new
    evaluations_file = meta_agent_dir / "evaluations.json"
    if evaluations_file.exists():
        with open(evaluations_file, 'r') as f:
            evaluations = json.load(f)
    else:
        evaluations = {}
    
    # Add current iteration results (summary for meta-agent)
    evaluations[f"iter{iteration}"] = {
        f"train_{primary_metric_name}": train_primary_metric,
        "train_metrics": train_metrics,  # All train metrics
        f"val_{primary_metric_name}": val_metrics.get(primary_metric_name, 0.0),
        "val_metrics": val_metrics,  # All val metrics
        "val_total": val_total,
        "total_rollouts": cumulative_rollouts,
        "num_sub_iters": num_sub_iters,
        "last_sub_folder": last_sub_folder_name,
    }
    
    # Save summary evaluations
    with open(evaluations_file, 'w') as f:
        json.dump(evaluations, f, indent=2)
    
    logger.info(f"✅ Aggregated iter{iteration} results to meta_agent/evaluations.json: train_{primary_metric_name}={train_primary_metric:.2%}, val_{primary_metric_name}={val_metrics[primary_metric_name]:.2%}")
    
    # Copy skill from last sub-iteration to meta_agent/skills/
    if iteration >= 1:
        last_sub_folder = workspace_base / last_sub_folder_name
        source_skill = last_sub_folder / ".claude" / "skills" / "learning-context" / "SKILL.md"
        target_skill_dir = meta_agent_dir / "skills" / f"iter{iteration}"
        target_skill_dir.mkdir(parents=True, exist_ok=True)
        target_skill = target_skill_dir / "SKILL.md"
        
        if source_skill.exists():
            shutil.copy2(source_skill, target_skill)
            logger.info(f"✅ Copied skill to meta_agent/skills/iter{iteration}/SKILL.md")
        else:
            logger.warning(f"⚠️  No skill found at {source_skill}")


def get_sub_iteration_folder_name(iteration: int, sub_iter: int = None) -> str:
    """
    Get folder name for iteration or sub-iteration.
    
    Examples:
        iter=0, sub_iter=None  → "iter0"
        iter=1, sub_iter=None  → "iter1"
        iter=1, sub_iter=0     → "iter1_sub0" 
        iter=1, sub_iter=3     → "iter1_sub3"
    """
    if sub_iter is None:
        return f"iter{iteration}"
    else:
        return f"iter{iteration}_sub{sub_iter}"


def create_iteration_workspace(
    workspace_base: Path,
    iteration: int,
    sub_iter: int = None,
) -> Path:
    """
    Create workspace folder for a specific iteration or sub-iteration.

    Args:
        workspace_base: Base workspace directory (e.g., workspace/finer)
        iteration: Iteration number (0+)
        sub_iter: Sub-iteration number (None for legacy mode, 0+ for sub-iterations)
    
    Returns:
        Path to created iteration directory
    
    Raises:
        FileNotFoundError: If seed folder doesn't exist
        FileExistsError: If iteration directory already exists
    """
    workspace_base = Path(workspace_base)
    folder_name = get_sub_iteration_folder_name(iteration, sub_iter)
    iter_folder = workspace_base / folder_name
    
    logger.debug(f"Creating iteration workspace: {iter_folder}")

    # Check if iteration folder already exists
    if iter_folder.exists():
        error_msg = (
            f"Iteration folder already exists at {iter_folder}. "
            "Please remove it or use a different iteration number."
        )
        logger.error(error_msg)
        raise FileExistsError(error_msg)

    # Create iteration folder
    iter_folder.mkdir(parents=True, exist_ok=True)
    
    # Create .claude/skills directory for iter >= 1 (when meta-agent is used)
    if iteration >= 1:
        skills_dir = iter_folder / ".claude" / "skills" / "learning-context"
        skills_dir.mkdir(parents=True, exist_ok=True)

    return iter_folder


def setup_base_agent_workspace(
    workspace_base: Path,
    iter_folder: Path,
    iteration: int,
    env,
    logger: logging.Logger = None,
    source_folder: Path = None,
) -> None:
    """
    Set up base agent workspace for the current iteration or sub-iteration.
    
    For iteration >= 1:
    1. Copy context/ and interfaces/ from source folder (best previous iteration or previous sub-iteration)
    2. Copy mce/workspace_utils/ to iter_folder/utils/
    
    Note: Training data (train.json) is NOT copied here. It will be generated by evaluating
    the current batch BEFORE running base-agent.
    
    Args:
        workspace_base: Base workspace directory (e.g., workspace/finer)
        iter_folder: Current iteration folder path
        iteration: Current iteration number
        env: Environment instance
        logger: Optional logger instance
        source_folder: Explicit source folder to copy from (for sub-iterations). If None, finds best previous iteration.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if iteration < 1:
        return
    
    workspace_base = Path(workspace_base)
    
    # Determine source folder
    if source_folder is not None:
        source_iter_dir = Path(source_folder)
        source_name = source_iter_dir.name
        logger.info(f"Copying from {source_name}")
    else:
        # Find source iteration
        if iteration == 1:
            source_iter_dir = workspace_base / "iter0"
            source_name = "iter0"
            logger.info(f"Copying from iter0 (baseline); if any seed context exists, it will be copied")
        else:
            best_iter_info = _find_best_iteration(workspace_base, iteration, env)
            if best_iter_info:
                source_iter_num = best_iter_info['iteration']
                best_metric_value = best_iter_info['primary_metric_value']
                last_sub_folder = best_iter_info.get('last_sub_folder', f'iter{source_iter_num}')
                source_iter_dir = workspace_base / last_sub_folder
                source_name = last_sub_folder
                logger.info(f"Copying from {source_name} (iter{source_iter_num} val_primary_metric: {best_metric_value:.2%})")
            else:
                raise ValueError(f"No previous iterations found for iteration {iteration}")
    
    # Copy context/ directory
    source_context = source_iter_dir / "context"
    target_context = iter_folder / "context"
    if source_context.exists() and not target_context.exists():
        shutil.copytree(source_context, target_context, ignore=ignore_pycache)
        logger.info(f"✅ Copied context/ from {source_name}")
    elif not target_context.exists():
        target_context.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created empty context/ folder")
    
    # Copy interfaces/ directory (new interface-based system)
    source_interfaces = source_iter_dir / "interfaces"
    target_interfaces = iter_folder / "interfaces"
    if source_interfaces.exists() and not target_interfaces.exists():
        shutil.copytree(source_interfaces, target_interfaces, ignore=ignore_pycache)
        logger.info(f"✅ Copied interfaces/ from {source_name}")
    elif not target_interfaces.exists():
        target_interfaces.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created empty interfaces/ folder")
    
    # Copy workspace_utils to utils/
    project_root = workspace_base.parent.parent
    source_utils = project_root / "mce" / "workspace_utils"
    target_utils = iter_folder / "utils"
    if source_utils.exists() and not target_utils.exists():
        shutil.copytree(source_utils, target_utils, ignore=ignore_pycache)
        logger.info(f"✅ Copied workspace_utils/ to utils/")
    
    # Create data directory (training data will be written after batch evaluation)
    data_dir = iter_folder / "data"
    data_dir.mkdir(exist_ok=True)


def copy_skills_to_sub_iteration(
    source_folder: Path,
    target_folder: Path,
    logger: logging.Logger = None,
) -> None:
    """
    Copy .claude/skills from source to target folder.
    
    Args:
        source_folder: Source iteration folder (with skills)
        target_folder: Target sub-iteration folder
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    source_skills = source_folder / ".claude" / "skills"
    target_skills = target_folder / ".claude" / "skills"

    # Ensure parent directory exists
    target_skills.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy with dirs_exist_ok=True to overwrite existing directory
    shutil.copytree(source_skills, target_skills, ignore=ignore_pycache, dirs_exist_ok=True)
    logger.info(f"✅ Copied skills from {source_folder.name}")



def _find_best_iteration(workspace_base: Path, current_iteration: int, env) -> Optional[Dict[str, Any]]:
    """
    Find the best performing previous iteration by reading meta_agent/evaluations.json.
    Uses validation primary metric to select the best iteration.
    
    Args:
        workspace_base: Base workspace directory
        current_iteration: Current iteration number (to know which previous iters to check)
        env: Environment instance
    
    Returns:
        Dictionary with 'iteration', 'primary_metric_value', and 'last_sub_folder' keys, or None if no previous iterations
    """
    workspace_base = Path(workspace_base)
    evaluations_file = workspace_base / "meta_agent" / "evaluations.json"
    
    # For iteration 0, no previous iterations
    if current_iteration == 0:
        return None
    
    # For iteration 1, always use iter0 (baseline)
    if current_iteration == 1:
        # Check if iter0 exists
        iter0_dir = workspace_base / "iter0"
        if iter0_dir.exists():
            return {
                'iteration': 0,
                'primary_metric_value': 0.0,
                'last_sub_folder': 'iter0',
            }
        return None
    
    # For iteration >= 2, read from aggregated evaluations
    if not evaluations_file.exists():
        logger.warning(f"No evaluations file found at {evaluations_file}")
        return None
    
    with open(evaluations_file, 'r') as f:
        evaluations = json.load(f)
    
    best_iter = None
    best_metric_value = -1e9
    best_last_sub_folder = None
    
    for i in range(current_iteration):
        iter_key = f"iter{i}"
        if iter_key not in evaluations:
            logger.warning(f"No evaluation found for {iter_key}")
            continue
        
        iter_data = evaluations[iter_key]
        
        # Get primary metric value from the generic key
        primary_metric_name = env.get_primary_metric_name()
        primary_metric_value = iter_data.get(f'val_{primary_metric_name}')
        if primary_metric_value is None:
            logger.warning(f"No val_{primary_metric_name} found for {iter_key}")
            continue
        
        if primary_metric_value > best_metric_value:
            best_metric_value = primary_metric_value
            best_iter = i
            best_last_sub_folder = iter_data.get('last_sub_folder', f'iter{i}')
    
    if best_iter is not None:
        return {
            'iteration': best_iter,
            'primary_metric_value': best_metric_value,
            'last_sub_folder': best_last_sub_folder,
        }
    
    return None


def cleanup_irrelevant_files(
    iter_dir: Path,
    agent_type: str,
    logger: logging.Logger = None,
) -> None:
    """
    Clean up irrelevant files generated by agents.
    
    Meta-agent should only generate:
    - .claude/skills/learning-context/SKILL.md
    
    Base-agent should only generate:
    - retrieve_context.py
    - context/ directory (with markdown files)
    
    Args:
        iter_dir: Iteration directory
        agent_type: "meta" or "base"
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    iter_dir = Path(iter_dir)
    
    # Files/directories that should always be preserved
    protected_paths = {
        "data",
        "utils",
        "__pycache__",
        ".claude",
        "context",
        "interfaces",  # New interface-based system
    }
    
    deleted_files = []
    
    if agent_type == "meta":
        # Meta-agent: Only keep .claude/skills/learning-context/SKILL.md
        # Delete everything else at root level (except protected paths)
        for item in iter_dir.iterdir():
            item_name = item.name
            
            # Skip protected paths
            if item_name in protected_paths:
                continue
            
            # Skip hidden files/directories (except .claude which is protected)
            if item_name.startswith("."):
                continue
            
            # Delete everything else
            try:
                if item.is_file():
                    item.unlink()
                    deleted_files.append(item_name)
                    logger.debug(f"Deleted irrelevant file: {item_name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_files.append(f"{item_name}/")
                    logger.debug(f"Deleted irrelevant directory: {item_name}/")
            except Exception as e:
                logger.warning(f"Failed to delete {item_name}: {e}")
    
    elif agent_type == "base":
        # Base-agent: Only keep context/ and interfaces/
        # Delete other files at root level (except protected paths)
        for item in iter_dir.iterdir():
            item_name = item.name
            
            # Skip protected paths
            if item_name in protected_paths:
                continue
            
            # Skip hidden files/directories
            if item_name.startswith("."):
                continue
            
            # Skip context/ and interfaces/ (they are protected)
            if item_name in ["context", "interfaces"]:
                continue
            
            # Delete everything else
            try:
                if item.is_file():
                    item.unlink()
                    deleted_files.append(item_name)
                    logger.debug(f"Deleted irrelevant file: {item_name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_files.append(f"{item_name}/")
                    logger.debug(f"Deleted irrelevant directory: {item_name}/")
            except Exception as e:
                logger.warning(f"Failed to delete {item_name}: {e}")
    
    if deleted_files:
        logger.info(f"🧹 Cleaned up {len(deleted_files)} irrelevant file(s): {', '.join(deleted_files)}")
    else:
        logger.debug("No irrelevant files found to clean up")
