"""
Base-agent implementation using Claude Agent SDK.

The base-agent learns task-specific context from training data using skills
provided by the meta-agent. It implements interfaces defined by InterfaceSignatures.

Key features:
- Multi-turn validation loop: agent runs, system validates, feeds errors back
- Signature-driven: validates against InterfaceSignature definitions
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import partial
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions
)

from mce.logging_utils import setup_logger, log_message
from mce.prompts.base_agent import build_base_agent_prompt
from mce.utils import cleanup_irrelevant_files
from mce.validation import validate_interfaces, format_validation_feedback, ValidationResult

from env.base import InterfaceSignature

from dotenv import load_dotenv
load_dotenv(override=True)


async def base_agent_permission_handler(
    tool_name: str,
    input_data: dict,
    context: dict,
    iter_dir: Path,
):
    """
    Permission handler for base-agent to restrict operations.
    
    Ensures the agent:
    1. Only uses allowed tools
    2. Can only read/write files within its iteration directory
    """
    allowed_tools = [
        "Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep", 
        "Task", "TaskOutput", "ExitPlanMode", "TodoWrite", "KillShell", "EnterPlanMode"
    ]
    
    if tool_name not in allowed_tools:
        return {
            "behavior": "deny",
            "message": f"Tool '{tool_name}' not allowed. Allowed: {', '.join(allowed_tools)}",
            "interrupt": False
        }
    
    iter_dir = iter_dir.resolve()
    
    # Tools that involve file paths
    file_tools = ["Read", "Write", "Edit", "Glob", "Grep"]

    if tool_name in file_tools:
        file_path = input_data.get("file_path") or input_data.get("path")
        if file_path:
            # Resolve the absolute path (relative to iter_dir since that's the cwd)
            if not Path(file_path).is_absolute():
                resolved_path = (iter_dir / file_path).resolve()
            else:
                resolved_path = Path(file_path).resolve()
            
            # All file operations restricted to iter_dir only
            try:
                resolved_path.relative_to(iter_dir)
            except ValueError:
                return {
                    "behavior": "deny",
                    "message": f"Access denied: restricted to {iter_dir}",
                    "interrupt": True
                }
            
            # Prevent writing to utils/
            if tool_name in ["Write", "Edit"]:
                try:
                    utils_dir = (iter_dir / "utils").resolve()
                    resolved_path.relative_to(utils_dir)
                    return {
                        "behavior": "deny",
                        "message": "Access denied: cannot write to utils/",
                        "interrupt": True
                    }
                except ValueError:
                    pass
            
            return {"behavior": "allow", "updatedInput": input_data}
    
    return {"behavior": "allow", "updatedInput": input_data}


async def run_base_agent(
    iter_dir: Path,
    task_instruction: str,
    interface_signatures: List[InterfaceSignature],
    workspace_base: Path = None,
    log_dir: str = "logs",
    run_dir: Path = None,
    iteration: int = None,
    e2b_sandbox_manager = None,
    initial_prompt: str = None,
    max_validation_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Run base-agent with multi-turn validation loop.
    
    Args:
        iter_dir: Iteration directory
        task_instruction: Task-instruction from env
        interface_signatures: Required interface signatures to implement
        workspace_base: Base workspace directory
        log_dir: Directory for log files
        run_dir: Run directory for organized logging
        iteration: Iteration number
        e2b_sandbox_manager: E2B sandbox manager (None = run locally)
        initial_prompt: Optional initial prompt
        max_validation_attempts: Max validation retry attempts

    Returns:
        Dict with success status, interfaces, and metadata
    """
    # Extract sub_iteration from iter_dir path
    sub_iteration = None
    iter_dir_name = Path(iter_dir).name
    if "_sub" in iter_dir_name:
        sub_iteration = int(iter_dir_name.split("_sub")[1])

    # Set up iteration-specific logger
    if run_dir and iteration is not None:
        logger = setup_logger(
            name=f"base_iter{iteration}_sub{sub_iteration}" if sub_iteration is not None else f"base_iter{iteration}",
            run_dir=run_dir,
            agent_type="base",
            iteration=iteration,
            sub_iteration=sub_iteration,
            minimal_console=True
        )
    else:
        logger = setup_logger(name="base_agent", log_dir=log_dir, console_colors=True)
    
    logger.info(f"\n🤖 BASE-AGENT: Learning context")
    logger.info(f"  Iteration directory: {iter_dir}")
    logger.info(f"  Required interfaces: {[s.name for s in interface_signatures]}")
    
    if workspace_base is None:
        workspace_base = iter_dir.parent
    workspace_base = Path(workspace_base)
    
    # Build prompt with interface signatures
    full_prompt = build_base_agent_prompt(
        task_instruction=task_instruction,
        interface_signatures=interface_signatures,
        iter_dir=str(iter_dir),
        workspace_base=str(workspace_base),
        initial_prompt=initial_prompt,
    )
    
    logger.info("\n" + "="*80)
    logger.info("📝 BASE-AGENT PROMPT")
    logger.info("="*80)
    logger.info(f"\n{full_prompt}\n")
    logger.info("="*80 + "\n")
    
    # Handle E2B sandbox execution
    if e2b_sandbox_manager:
        return await _run_in_e2b(
            e2b_sandbox_manager=e2b_sandbox_manager,
            iter_dir=iter_dir,
            full_prompt=full_prompt,
            interface_signatures=interface_signatures,
            logger=logger,
        )
    
    # Local execution with validation loop
    allowed_tools = [
        "Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "Task", "TaskOutput", "ExitPlanMode", "TodoWrite", "KillShell", "EnterPlanMode"
    ]
    
    options = ClaudeAgentOptions(
        cwd=str(iter_dir),
        setting_sources=["project"],
        allowed_tools=allowed_tools,
        can_use_tool=partial(base_agent_permission_handler, iter_dir=iter_dir)
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query(full_prompt)
        
        # If no interface signatures, skip validation loop
        if not interface_signatures:
            logger.info("No interface signatures required - skipping validation")
            
            # Just collect agent response
            message_count = 0
            async for message in client.receive_response():
                message_count += 1
                log_message(message, logger, minimal_console=(run_dir is not None))
            
            logger.info(f"Agent completed with {message_count} messages")
            cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
            return {
                'success': True,
                'interfaces': {},
                'message_count': message_count,
                'validation_attempts': 0,
            }
        
        # Validation loop for environments with interfaces
        for attempt in range(max_validation_attempts):
            logger.info(f"\n--- Validation attempt {attempt + 1}/{max_validation_attempts} ---")
            
            # Collect agent response
            message_count = 0
            async for message in client.receive_response():
                message_count += 1
                log_message(message, logger, minimal_console=(run_dir is not None))
            
            logger.info(f"Agent completed with {message_count} messages")
            
            # Validate interfaces
            validation_result = validate_interfaces(iter_dir, interface_signatures)
            
            if validation_result.success:
                logger.info(f"✅ All {len(interface_signatures)} interfaces validated successfully")
                cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
                return {
                    'success': True,
                    'interfaces': validation_result.interfaces,
                    'message_count': message_count,
                    'validation_attempts': attempt + 1,
                }
            
            # Log validation errors
            logger.warning(f"❌ Validation failed with {len(validation_result.errors)} errors:")
            for error in validation_result.errors:
                logger.warning(f"  - {error}")
            
            # Check if we have more attempts
            if attempt + 1 >= max_validation_attempts:
                logger.error(f"Max validation attempts ({max_validation_attempts}) exceeded")
                break
            
            # Feed errors back to agent for continuation
            feedback = format_validation_feedback(validation_result)
            logger.info(f"📤 Sending validation feedback to agent...")
            await client.query(feedback)
    
    # Validation failed after all attempts
    cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
    return {
        'success': False,
        'error': f'Validation failed after {max_validation_attempts} attempts',
        'last_errors': validation_result.errors if validation_result else [],
        'message_count': message_count,
    }


async def _run_in_e2b(
    e2b_sandbox_manager,
    iter_dir: Path,
    full_prompt: str,
    interface_signatures: List[InterfaceSignature],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run agent in E2B sandbox."""
    raise NotImplementedError("E2B sandbox execution is not implemented yet")

    allowed_tools = [
        "Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "Task", "TaskOutput", "ExitPlanMode", "TodoWrite", "KillShell", "EnterPlanMode"
    ]
    
    logger.info("🔒 Running agent in E2B sandbox")
    try:
        result = await e2b_sandbox_manager.run_agent(
            iter_dir=iter_dir,
            prompt=full_prompt,
            allowed_tools=allowed_tools,
            timeout=1800,
            logger=logger,
        )
        
        if not result["success"]:
            logger.error(f"E2B execution failed: {result.get('stderr', 'Unknown error')}")
            return {
                'success': False,
                'error': f"E2B execution failed: {result.get('stderr', 'Unknown error')}",
                'message_count': 0
            }
        
        logger.info("✓ E2B execution completed")
        
        # Validate after E2B execution
        cleanup_irrelevant_files(iter_dir, agent_type="base", logger=logger)
        validation_result = validate_interfaces(iter_dir, interface_signatures)
        
        if validation_result.success:
            return {
                'success': True,
                'interfaces': validation_result.interfaces,
                'message_count': 0,
            }
        else:
            return {
                'success': False,
                'error': 'Validation failed after E2B execution',
                'last_errors': validation_result.errors,
                'message_count': 0,
            }
            
    except Exception as e:
        logger.error(f"E2B execution failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'message_count': 0
        }


if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Run base-agent to learn context")
        parser.add_argument("iter_dir", type=str, help="Iteration directory")
        parser.add_argument("--env", type=str, required=True, help="Environment name")
        parser.add_argument("--iteration", type=int, default=None, help="Iteration number")
        args = parser.parse_args()
        
        from env.registry import EnvironmentRegistry
        
        env = EnvironmentRegistry.get(args.env)
        task_instruction = env.get_task_instruction()
        interface_signatures = env.get_interface_signatures()
        
        iter_dir = Path(args.iter_dir).resolve()
        
        skill_path = iter_dir / ".claude" / "skills" / "learning-context" / "SKILL.md"
        if not skill_path.exists():
            print(f"✗ SKILL.md not found at {skill_path}")
            print("  Run meta-agent first to generate it.")
            return
        
        result = await run_base_agent(
            iter_dir=iter_dir,
            task_instruction=task_instruction,
            interface_signatures=interface_signatures,
            iteration=args.iteration
        )
        
        if result['success']:
            print(f"\n✓ Base-agent completed successfully")
            print(f"  Validated interfaces: {list(result['interfaces'].keys())}")
        else:
            print(f"\n✗ Base-agent failed: {result['error']}")
    
    asyncio.run(main())
