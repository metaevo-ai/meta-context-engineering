"""
MCE Main Loop: Orchestrate multiple iterations of context engineering.
"""

import os
import time
import json
import asyncio
import argparse
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from mce.utils import (
    create_iteration_workspace, 
    setup_base_agent_workspace, 
    setup_meta_agent_reference,
    copy_skills_to_sub_iteration,
    get_sub_iteration_folder_name,
    aggregate_iteration_results,
)
from mce.eval import batch_evaluate, load_interfaces
from mce.logging_utils import setup_logger, setup_run_logger
from mce.meta_agent import run_meta_agent
from mce.base_agent import run_base_agent
from mce.llm_client import LLMClient
from env.registry import EnvironmentRegistry
from dotenv import load_dotenv

load_dotenv(override=True)


async def run_iteration(
    workspace_base: Path,
    iteration: int,
    env_name: str,
    val_data_path: str,
    train_data_path: str,
    train_limit: int,
    val_limit: int,
    model: str,
    logger: logging.Logger,
    run_dir: Path,
    e2b_sandbox_manager = None,
    train_batch_size: int = None,
    skill_path: str = None,
    no_meta_agent: bool = False,
) -> dict:
    """
    Run a single iteration of the MCE loop with optional sub-iterations.
    
    When train_batch_size is specified, the iteration is split into sub-iterations:
    - Each sub-iteration processes a batch of training samples
    - Base-agent learns incrementally from each batch
    - Final validation is done at the end of all sub-iterations
    
    Args:
        workspace_base: Base workspace directory (absolute path)
        iteration: Iteration number
        env_name: Environment name
        val_data_path: Path to validation data file
        train_data_path: Path to training data file
        train_limit: Number of training samples to evaluate
        val_limit: Number of validation samples to evaluate
        model: LLM model to use
        logger: Logger instance
        run_dir: Run directory for organized logging
        e2b_sandbox_manager: E2B sandbox manager (None = run locally)
        train_batch_size: Batch size for sub-iterations (None = process all at once)
        skill_path: Path to pre-evolved skill directory (None = run meta-agent)
        no_meta_agent: Skip meta-agent entirely (no skills will be used)
    
    Returns:
        Dictionary with iteration results (accuracy, errors, etc.)
    """
    logger.info(f"\n🔄 ITERATION {iteration}")

    env = EnvironmentRegistry.get(env_name)
    task_instruction = env.get_task_instruction()
    interface_signatures = env.get_interface_signatures()
    
    llm = LLMClient(model=model)

    # ========== ITERATION 0: Baseline (validation only) ==========
    if iteration == 0:
        iter_folder = create_iteration_workspace(workspace_base, iteration)
        logger.info(f"\n🔧 BASELINE ITERATION 0 - Validation only (empty interfaces)")
        
        # Only evaluate validation set for baseline with empty interfaces
        try:
            val_samples = env.load_samples(path=val_data_path, limit=val_limit, random_sample=False)
            logger.info(f"Evaluating {len(val_samples)} validation samples (no interfaces)...")
            val_data = await batch_evaluate(
                interfaces={},  # Empty interfaces for baseline
                samples=val_samples,
                env_name=env_name,
                llm=llm,
                iter_dir=iter_folder,
                log_dir=run_dir,
            )
            val_summary = val_data["summary"]
            primary_metric_value = val_summary["primary_metric_value"]
            
            # Save evaluation results
            eval_data = {
                "summary": {
                    "val_primary_metric": primary_metric_value,
                    "val_metrics": val_summary["metrics"],
                    "val_total": val_summary["total"],
                    "environment": env_name,
                },
            }
            
            logger.info(f"✅ Baseline validation (not saved to file): {primary_metric_value:.2%}")
            
            return {
                "iteration": iteration,
                "train_primary_metric": 0.0,
                "val_primary_metric": primary_metric_value,
                "train_total": 0,
                "val_total": val_summary["total"],
                "cumulative_rollouts": 0,
            }
        except Exception as e:
            logger.error(f"❌ Error during baseline evaluation: {e}", exc_info=True)
            return {"iteration": iteration, "error": str(e), "train_accuracy": 0.0, "val_accuracy": 0.0}

    # ========== ITERATION >= 1: Bi-level agents with sub-iterations ==========
    
    # Load all training samples upfront
    train_samples = env.load_samples(path=train_data_path, limit=train_limit, random_sample=True, shuffle=True)

    train_limit = min(train_limit, len(train_samples))
    
    # Calculate number of sub-iterations
    if train_batch_size is None or train_batch_size >= train_limit:
        num_sub_iters = 1
        train_batch_size = train_limit
    else:
        num_sub_iters = (train_limit + train_batch_size - 1) // train_batch_size
    
    logger.info(f"  📊 Sub-iterations: {num_sub_iters} (batch_size={train_batch_size}, total={train_limit})")
    
    # Step 1: Create first sub-iteration folder and setup skills
    first_sub_folder = create_iteration_workspace(workspace_base, iteration, sub_iter=0)
    
    # Conditional: use pre-evolved skill, skip meta-agent, or run meta-agent
    if no_meta_agent:
        logger.info("\n🧠 STEP 1: Skipping Meta-Agent (no skills will be used)")
        meta_result = {'success': True}
    elif skill_path:
        logger.info("\n🧠 STEP 1: Using Pre-Evolved Skill (skipping meta-agent)")
        logger.info(f"  Copying skills from: {skill_path}")
        
        # Copy skills from provided path to first sub-iteration
        skill_source = Path(skill_path)
        target_skills = first_sub_folder / ".claude" / "skills"
        target_skills.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(skill_source, target_skills, dirs_exist_ok=True)
        logger.info(f"✅ Successfully copied pre-evolved skills")
        meta_result = {'success': True}
    else:
        logger.info("\n🧠 STEP 1: META-AGENT - Generating/Evolving Skills")
        meta_result = await run_meta_agent(
            iter_dir=first_sub_folder,
            task_instruction=task_instruction,
            interface_signatures=interface_signatures,
            iteration=iteration,
            workspace_base=workspace_base,
            run_dir=run_dir,
            e2b_sandbox_manager=e2b_sandbox_manager,
        )
        
        if not meta_result['success']:
            logger.error(f"Meta-agent failed: {meta_result['error']}")
            raise Exception(f"Meta-agent failed: {meta_result['error']}")
    
    # Step 2: Run sub-iterations
    logger.info(f"\n🔄 STEP 2: SUB-ITERATIONS - Batch Learning ({num_sub_iters} batches)")
    
    intermediate_results = []
    cumulative_rollouts = 0
    current_folder = first_sub_folder
    
    for sub_iter in range(num_sub_iters):
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 SUB-ITERATION {iteration}.{sub_iter} ({sub_iter + 1}/{num_sub_iters})")
        logger.info(f"{'='*60}")

        # Create sub-iteration folder (first one already created for meta-agent)
        if sub_iter == 0:
            sub_iter_folder = first_sub_folder
            # Setup workspace: copy from best previous iteration
            setup_base_agent_workspace(workspace_base, sub_iter_folder, iteration, env, logger)
        else:
            sub_iter_folder = create_iteration_workspace(workspace_base, iteration, sub_iter=sub_iter)
            # Copy skills and context from previous sub-iteration
            prev_sub_folder = workspace_base / get_sub_iteration_folder_name(iteration, sub_iter - 1)
            copy_skills_to_sub_iteration(prev_sub_folder, sub_iter_folder, logger)
            setup_base_agent_workspace(
                workspace_base, sub_iter_folder, iteration, env, logger,
                source_folder=prev_sub_folder
            )
        
        # Calculate batch range
        start_idx = sub_iter * train_batch_size
        end_idx = min(start_idx + train_batch_size, len(train_samples))
        batch_samples = train_samples[start_idx:end_idx]
        batch_size = len(batch_samples)
        cumulative_rollouts += batch_size
        
        logger.info(f"  📊 Batch: samples [{start_idx}:{end_idx}] ({batch_size} samples)")
        logger.info(f"  📊 Cumulative rollouts: {cumulative_rollouts}")
        
        # Evaluate current batch
        logger.info(f"\n📊 Evaluating batch {sub_iter}...")
        
        # Load interfaces (or use empty if first run)
        try:
            interfaces = load_interfaces(sub_iter_folder, interface_signatures)
        except FileNotFoundError:
            interfaces = {}  # Empty interfaces for first evaluation
        
        batch_data = await batch_evaluate(
            interfaces=interfaces,
            samples=batch_samples,
            env_name=env_name,
            llm=llm,
            iter_dir=sub_iter_folder,
            log_dir=run_dir,
        )
        batch_summary = batch_data["summary"]
        primary_metric_value = batch_summary["primary_metric_value"]
        
        logger.info(f"  ✅ Batch {batch_summary['primary_metric']}: {primary_metric_value:.2%}")
        
        # Save batch results for base-agent to learn from
        # Use environment's format_result_for_training to decide what fields to include
        batch_results = [
            env.format_result_for_training(item)
            for item in batch_data.get("results", [])
        ]
        
        primary_metric_name = env.get_primary_metric_name()
        batch_eval_data = {
            "summary": {
                f"train_{primary_metric_name}": primary_metric_value,
                "train_metrics": batch_summary["metrics"],
                "train_total": batch_summary["total"],
                "train_errors": batch_summary["errors"],  # Program failures
                "batch_idx": sub_iter,
                "cumulative_rollouts": cumulative_rollouts,
            },
            "detailed_results": batch_results,
        }
        
        # Save batch evaluation results as train.json for base-agent to learn from
        data_dir = sub_iter_folder / "data"
        data_dir.mkdir(exist_ok=True)
        
        train_file = data_dir / "train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(batch_eval_data, f, indent=2, ensure_ascii=False)
    
        # Run base-agent to learn from this batch
        logger.info(f"\n🤖 BASE-AGENT: Learning from batch {sub_iter}...")
        base_result = await run_base_agent(
            iter_dir=sub_iter_folder,
            task_instruction=task_instruction,
            interface_signatures=interface_signatures,
            workspace_base=workspace_base,
            run_dir=run_dir,
            iteration=iteration,
            e2b_sandbox_manager=e2b_sandbox_manager,
        )
        
        if not base_result['success']:
            logger.error(f"Base-agent failed: {base_result['error']}")
            raise Exception(f"Base-agent failed at sub-iter {sub_iter}: {base_result['error']}")
        
        logger.info(f"✅ Base-agent completed for sub-iter {sub_iter}")
        
        # Record intermediate result
        intermediate_results.append({
            "sub_iter": sub_iter,
            "folder": str(sub_iter_folder),
            "batch_start": start_idx,
            "batch_end": end_idx,
            "batch_size": batch_size,
            "cumulative_rollouts": cumulative_rollouts,
            "batch_train_primary_metric": primary_metric_value,
            "batch_train_metrics": batch_summary["metrics"],
        })
        
        current_folder = sub_iter_folder
    
    # Step 3: Final validation evaluation
    last_sub_iter = num_sub_iters - 1
    final_folder = current_folder  # Use the last sub-iteration folder directly
    
    logger.info(f"\n📊 STEP 3: FINAL VALIDATION")

    interfaces = load_interfaces(final_folder, interface_signatures)
    val_samples = env.load_samples(path=val_data_path, limit=val_limit, random_sample=False)
    logger.info(f"Evaluating {len(val_samples)} validation samples...")
    val_data = await batch_evaluate(
        interfaces=interfaces,
        samples=val_samples,
        env_name=env_name,
        llm=llm,
        iter_dir=final_folder,
        log_dir=run_dir,
    )
    val_summary = val_data["summary"]
    val_primary_metric = val_summary["primary_metric_value"]
    
    # Aggregate results to meta_agent/ for meta-agent review
    # (No longer saving to individual sub-iteration folders)
    aggregate_iteration_results(
        workspace_base=workspace_base,
        iteration=iteration,
        sub_iterations=intermediate_results,
        val_primary_metric=val_primary_metric,
        val_metrics=val_summary["metrics"],
        val_total=val_summary["total"],
        cumulative_rollouts=cumulative_rollouts,
        num_sub_iters=num_sub_iters,
        last_sub_folder_name=current_folder.name,
        batch_size=train_batch_size,
        environment=env,
        logger=logger,
    )
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ITERATION {iteration} COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"  📊 Total rollouts: {cumulative_rollouts}")
    logger.info(f"  📊 Sub-iterations: {num_sub_iters}")
    logger.info(f"  ✅ Final validation {val_summary['primary_metric']}: {val_primary_metric:.2%}")
    
    # Log sub-iteration summary
    logger.info(f"\n  📈 Sub-iteration results:")
    for sr in intermediate_results:
        logger.info(f"    - Sub {sr['sub_iter']}: batch_metric={sr['batch_train_primary_metric']:.2%}, rollouts={sr['cumulative_rollouts']}")
    
    return {
        "iteration": iteration,
        "train_primary_metric": intermediate_results[-1]["batch_train_primary_metric"] if intermediate_results else 0.0,
        "val_primary_metric": val_primary_metric,
        "train_total": cumulative_rollouts,
        "val_total": val_summary["total"],
        "cumulative_rollouts": cumulative_rollouts,
        "num_sub_iters": num_sub_iters,
        "sub_iterations": intermediate_results,
    }


async def main():
    """Main entry point for MCE loop."""
    parser = argparse.ArgumentParser(
        description="Run MCE main loop: iterate, evaluate, and log results"
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path to workspace base directory (e.g., workspace/finer)"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment type"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data file"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data file"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=50,
        help="Training batch size for sub-iterations (default: None = process all at once). "
             "When set, each iteration is split into sub-iterations, each processing batch_size samples."
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=50,
        help="Number of samples in training (default: 50)"
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=20,
        help="Number of samples in validation (default: 20)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat-v3.1",
        help="LLM model used in context eval (default: deepseek/deepseek-chat-v3.1)"
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=1,
        help="Starting iteration number (default: 1)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)"
    )
    parser.add_argument(
        "--use-e2b",
        action="store_true",
        help="Run agents in E2B sandbox for isolation (requires E2B_API_KEY env var)"
    )
    parser.add_argument(
        "--skill-path",
        type=str,
        default=None,
        help="Path to pre-evolved skill directory. "
             "If provided, meta-agent will be skipped and this skill will be used for all iterations"
    )
    parser.add_argument(
        "--no-meta-agent",
        action="store_true",
        default=False,
        help="Skip meta-agent entirely (no skills will be used). "
             "This is mutually exclusive with --skill-path"
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive flags
    if args.skill_path and args.no_meta_agent:
        parser.error("--skill-path and --no-meta-agent are mutually exclusive")
    
    # Setup run directory for organized logging
    run_dir = setup_run_logger(log_base_dir=args.log_dir)
    
    # Setup main logger with run directory
    logger = setup_logger(
        name="mce_main",
        run_dir=run_dir,
        agent_type="run_summary",
        console_colors=True,
        minimal_console=False
    )

    # Print run start message
    timestamp = run_dir.name.replace("run_", "")
    print(f"[RUN {timestamp}] Starting MCE with {args.iterations} iteration(s) (iter{args.start_iter}-iter{args.start_iter + args.iterations - 1})")
    
    # Resolve workspace path
    workspace_base = Path(args.workspace).resolve()
    if not workspace_base.exists():
        logger.info(f"Creating workspace directory: {workspace_base}")
        workspace_base.mkdir(parents=True, exist_ok=True)
    
    # Validate skill path if provided
    if args.skill_path:
        skill_path = Path(args.skill_path).resolve()
        if not skill_path.exists() or not skill_path.is_dir():
            logger.error(f"Skill path not found or not a directory: {skill_path}")
            return
        logger.info(f"⚠️  Using pre-evolved skill from: {skill_path}")
        logger.info(f"⚠️  Meta-agent will be SKIPPED for all iterations")
    
    logger.info("\n🚀 MCE MAIN LOOP")
    logger.info(f"  Workspace: {workspace_base}")
    logger.info(f"  Environment: {args.env}")
    logger.info(f"  Iterations: {args.iterations} (starting from iter{args.start_iter})")
    logger.info(f"  Training data: {args.train_data}")
    logger.info(f"  Validation data: {args.val_data}")
    logger.info(f"  Train samples: {args.train_limit}")
    logger.info(f"  Val samples: {args.val_limit}")
    if args.train_batch_size:
        num_sub_iters = (args.train_limit + args.train_batch_size - 1) // args.train_batch_size
        logger.info(f"  Train batch size: {args.train_batch_size} ({num_sub_iters} sub-iterations per iteration)")
    else:
        logger.info(f"  Train batch size: None (process all samples at once)")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Use E2B sandbox: {args.use_e2b}")
    if args.no_meta_agent:
        logger.info(f"  Meta-agent: DISABLED (no skills)")
    elif args.skill_path:
        logger.info(f"  Pre-evolved skill: {args.skill_path}")
        logger.info(f"  Meta-agent: DISABLED (using pre-evolved skill)")
    else:
        logger.info(f"  Meta-agent: ENABLED")

    # Setup meta-agent reference data (copy training data once at the beginning)
    if args.train_data:
        setup_meta_agent_reference(workspace_base, args.train_data, args.train_limit, logger)

    # Check E2B API key if using E2B
    if args.use_e2b:
        if not os.getenv("E2B_API_KEY"):
            logger.error("E2B_API_KEY environment variable is not set. Get your API key from https://e2b.dev")
            return
    
    # Initialize E2B sandbox manager once for all iterations (if using E2B)
    e2b_sandbox_manager = None
    if args.use_e2b:
        raise NotImplementedError("E2B sandbox is not implemented")
        from mce.e2b_sandbox import E2BSandboxManager
        logger.info("🔒 Initializing E2B sandbox...")
        e2b_sandbox_manager = E2BSandboxManager(workspace_base, timeout=3600)  # 1 h
        e2b_sandbox_manager.initialize()
    
    # Run iterations
    results = []
    try:
        for i in range(args.start_iter, args.start_iter + args.iterations):
            result = await run_iteration(
                workspace_base=workspace_base,
                iteration=i,
                env_name=args.env,
                val_data_path=args.val_data,
                train_data_path=args.train_data,
                train_limit=args.train_limit,
                val_limit=args.val_limit,
                model=args.model,
                logger=logger,
                run_dir=run_dir,
                e2b_sandbox_manager=e2b_sandbox_manager,
                train_batch_size=args.train_batch_size,
                skill_path=args.skill_path,
                no_meta_agent=args.no_meta_agent,
            )
            results.append(result)
    finally:
        # Cleanup E2B sandbox
        if e2b_sandbox_manager:
            logger.info("🔒 Cleaning up E2B sandbox...")
            e2b_sandbox_manager.cleanup()
    
    # Print summary
    logger.info("\n🎯 FINAL SUMMARY")
    logger.info(f"Completed {len(results)} iteration(s):")
    
    total_rollouts = 0
    for result in results:
        iter_num = result["iteration"]
        if "error" in result:
            logger.error(f"  ❌ iter{iter_num}: ERROR - {result['error']}")
        else:
            val_metric = result.get("val_primary_metric", result.get("val_accuracy", 0.0))
            rollouts = result.get("cumulative_rollouts", result.get("train_total", 0))
            num_sub_iters = result.get("num_sub_iters", 1)
            total_rollouts += rollouts
            
            if num_sub_iters > 1:
                logger.info(f"  📊 iter{iter_num}: Val={val_metric:.2%}, Rollouts={rollouts} ({num_sub_iters} sub-iters)")
            else:
                train_metric = result.get("train_primary_metric", result.get("train_accuracy", 0.0))
                logger.info(f"  📊 iter{iter_num}: Train={train_metric:.2%}, Val={val_metric:.2%}, Rollouts={rollouts}")
    
    logger.info(f"\n📊 Total rollouts: {total_rollouts}")
    
    # Find best iteration based on validation primary metric
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        best = max(successful_results, key=lambda x: x.get("val_primary_metric", x.get("val_accuracy", 0.0)))
        best_val = best.get("val_primary_metric", best.get("val_accuracy", 0.0))
        logger.info(f"\n🏆 Best iteration: iter{best['iteration']} with validation metric {best_val:.2%}")
    else:
        logger.error(f"All iterations failed - Logs: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())

