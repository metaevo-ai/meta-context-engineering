"""
Evaluate learned interfaces on samples with async and concurrent execution.
"""

import asyncio
import json
import sys
import logging
import importlib.util
from typing import Callable, Dict, List, Any, Optional
from env.base import Sample, InterfaceSignature
from env.registry import EnvironmentRegistry
from pathlib import Path
from mce.llm_client import LLMClient
from mce.utils import compute_avg_metrics
from mce.validation import load_interfaces_from_init

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

MAX_CONCURRENCY = 30


def load_interfaces(iter_dir: Path, signatures: List[InterfaceSignature]) -> Dict[str, Callable]:
    """
    Load interfaces from an iteration directory.
    
    First tries to load from interfaces/__init__.py (new format).
    
    Args:
        iter_dir: Iteration directory
        signatures: Required interface signatures
        
    Returns:
        Dict mapping interface names to callables
    """
    # No signatures required = no interfaces to load
    if not signatures:
        logger.info("No interface signatures required, skipping interface loading")
        return {}
    
    iter_dir = Path(iter_dir)
    
    # Try new interfaces/ directory first
    interfaces_dir = iter_dir / "interfaces"
    if interfaces_dir.exists() and (interfaces_dir / "__init__.py").exists():
        logger.info(f"Loading interfaces from {interfaces_dir}")
        return load_interfaces_from_init(iter_dir)
    
    raise FileNotFoundError(
        f"No interfaces found in {iter_dir}. "
        f"Expected interfaces/__init__.py"
    )


async def batch_evaluate(
    interfaces: Dict[str, Callable],
    samples: List[Sample],
    env_name: str,
    llm: Optional[LLMClient] = None,
    iter_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate samples using loaded interfaces.
    
    Args:
        interfaces: Dict mapping interface names to callables
        samples: List of Sample objects to evaluate
        env_name: Environment name
        llm: Optional LLM client for environments that need inference
        iter_dir: Optional iteration directory (for context_dir in agent environments)
        log_dir: Optional log directory (for storing agent trajectories)
    
    Returns:
        Dictionary with evaluation results
    """
    environment = EnvironmentRegistry.get(env_name)
    
    # Prepare context_dir for agent environments
    context_dir = None
    if iter_dir:
        context_dir = Path(iter_dir) / "context"
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    completed_count = 0
    total_count = len(samples)
    
    async def evaluate_single(sample: Sample) -> Dict[str, Any]:
        """Evaluate a single sample using interfaces."""
        nonlocal completed_count
        async with semaphore:
            try:
                # Call environment's evaluate with interfaces
                result = await environment.aevaluate(
                    sample=sample,
                    interfaces=interfaces,
                    llm_client=llm,
                    context_dir=context_dir,
                    log_dir=log_dir,
                )
                
                completed_count += 1
                print(f"\rProgress: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%)", end="", flush=True)
                
                return {
                    "sample": sample.to_dict(),
                    "evaluation": {
                        "feedback": result.feedback,
                        "ground_truth": result.ground_truth,
                        "metrics": result.metrics,
                        "trajectory": result.trajectory,
                    }
                }
            except Exception as e:
                completed_count += 1
                logger.error(f"Error evaluating sample {sample.id}: {e}")
                return {
                    "sample": sample.to_dict(),
                    "error": str(e),
                    "evaluation": {
                        "feedback": f"Error: {e}",
                        "ground_truth": sample.ground_truth,
                        "metrics": {},
                    }
                }
    
    logger.info(f"Processing {total_count} samples with max concurrency of {MAX_CONCURRENCY}...")
    results = await asyncio.gather(*[evaluate_single(sample) for sample in samples])
    print()  # newline after progress
    
    # Separate errors and successful evaluations
    eval_errors = [r for r in results if "error" in r]
    successful = [r for r in results if "error" not in r]
    
    # Compute average metrics
    avg_metrics = compute_avg_metrics(successful)
    
    primary_metric_name = environment.get_primary_metric_name()
    primary_metric_value = avg_metrics.get(primary_metric_name, 0.0)
    
    log_data = {
        "summary": {
            "metrics": avg_metrics,
            "primary_metric": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "total": len(results),
            "successful": len(successful),
            "errors": len(eval_errors),
            "environment": env_name,
        },
        "errors": eval_errors,
        "results": successful,
    }
    
    logger.info(f"Evaluation Summary:")
    logger.info(f"  Primary Metric ({primary_metric_name}): {primary_metric_value:.2%}")
    logger.info(f"  Total: {len(results)} ({len(successful)} successful, {len(eval_errors)} errors)")

    return log_data


async def main():
    """Standalone evaluation script."""
    import argparse
    import time
    from .logging_utils import setup_logger
    
    parser = argparse.ArgumentParser(
        description="Evaluate learned interfaces from a workspace folder"
    )
    parser.add_argument(
        "--iter_dir",
        type=str,
        default="",
        help="Path to the iteration directory"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat-v3.1",
        help="LLM model to use"
    )
    parser.add_argument(
        "--save-results-to",
        type=str,
        required=True,
        help="Directory to save results to"
    )
    
    args = parser.parse_args()
    
    # Setup log directory
    log_dir = Path("logs") / "eval_standalone"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    eval_logger = setup_logger(name="eval", log_dir="logs", console_colors=True)
    
    if args.iter_dir:
        iter_dir = Path(args.iter_dir).resolve()
        if not iter_dir.exists() or not iter_dir.is_dir():
            raise ValueError(f"Iteration directory error: {iter_dir}")
    else:
        iter_dir = None
    
    eval_logger.info("="*80)
    eval_logger.info("EVALUATION SETUP")
    eval_logger.info("="*80)
    eval_logger.info(f"Iteration directory: {iter_dir}")
    eval_logger.info(f"Environment: {args.env}")
    eval_logger.info(f"Data: {args.data}")
    eval_logger.info(f"Sample limit: {args.limit}")
    
    # Get environment and its interface signatures
    environment = EnvironmentRegistry.get(args.env)
    signatures = environment.get_interface_signatures()
    
    # Load interfaces
    if iter_dir:
        eval_logger.info("Loading interfaces...")
        interfaces = load_interfaces(iter_dir, signatures)
        eval_logger.info(f"✓ Loaded interfaces: {list(interfaces.keys())}")
    else:
        interfaces = {}
        eval_logger.info("No iteration directory - using empty interfaces")
    
    # Initialize LLM if needed
    eval_logger.info(f"Initializing LLM: {args.model}")
    llm = LLMClient(model=args.model)
    eval_logger.info("✓ LLM initialized")
    
    eval_logger.info("="*80)
    eval_logger.info("STARTING EVALUATION")
    eval_logger.info("="*80)
    
    samples = environment.load_samples(path=args.data, limit=args.limit, random_sample=False)
    eval_logger.info(f"📦 Loaded {len(samples)} samples from: {args.data}")
    
    start_time = time.time()
    
    results = await batch_evaluate(
        interfaces=interfaces,
        samples=samples,
        env_name=args.env,
        llm=llm,
        iter_dir=iter_dir,
        log_dir=log_dir,
    )
    
    elapsed = time.time() - start_time
    summary = results["summary"]

    print(f"\n✓ Completed: {summary['primary_metric_value']:.2%} {summary['primary_metric']} ({summary['total']} samples) in {elapsed:.0f}s")
    
    save_dir = Path(args.save_results_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "evaluation.json"
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    eval_logger.info(f"📁 Results saved to: {log_path}")
    eval_logger.info("="*80)
    

if __name__ == "__main__":
    asyncio.run(main())
