"""
Signature-based validation for MCE interface implementations.

Validates that agent implementations match the required InterfaceSignatures.
"""

import ast
import sys
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from env.base import InterfaceSignature


@dataclass
class ValidationResult:
    """Result of interface validation."""
    success: bool
    errors: List[str] = field(default_factory=list)
    interfaces: Dict[str, Callable] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.success:
            return f"Validation passed: {len(self.interfaces)} interfaces loaded"
        return f"Validation failed with {len(self.errors)} errors:\n" + "\n".join(f"  - {e}" for e in self.errors)


def validate_interfaces(
    iter_dir: Path,
    signatures: List[InterfaceSignature]
) -> ValidationResult:
    """
    Validate all interface implementations against their signatures.
    
    Args:
        iter_dir: Iteration directory containing interfaces/
        signatures: List of required interface signatures
        
    Returns:
        ValidationResult with success status, errors, and loaded interfaces
    """
    errors = []
    interfaces = {}
    
    iter_dir = Path(iter_dir)
    interfaces_dir = iter_dir / "interfaces"
    
    # Check interfaces directory exists
    if not interfaces_dir.exists():
        return ValidationResult(
            success=False,
            errors=["interfaces/ directory not found. Create it and implement required functions."]
        )
    
    # Check __init__.py exists
    init_file = interfaces_dir / "__init__.py"
    if not init_file.exists():
        errors.append("interfaces/__init__.py not found. Create it to export your functions.")
    
    # Validate each signature
    for sig in signatures:
        result = _validate_single_interface(iter_dir, sig)
        if result["error"]:
            errors.append(f"[{sig.name}] {result['error']}")
        else:
            interfaces[sig.name] = result["function"]
    
    return ValidationResult(
        success=len(errors) == 0,
        errors=errors,
        interfaces=interfaces
    )


def _validate_single_interface(
    iter_dir: Path,
    sig: InterfaceSignature
) -> Dict[str, Any]:
    """
    Validate a single interface implementation.
    
    Checks:
    1. File exists: interfaces/{name}.py
    2. Function exists with correct name
    3. Function signature matches (parameter names)
    4. Function can be imported and called
    
    Returns:
        Dict with 'function' (if valid) or 'error' (if invalid)
    """
    interfaces_dir = iter_dir / "interfaces"
    file_path = interfaces_dir / f"{sig.name}.py"
    
    # 1. Check file exists
    if not file_path.exists():
        return {"error": f"File not found: interfaces/{sig.name}.py", "function": None}
    
    # 2. Parse AST and find function
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error in {sig.name}.py: {e}", "function": None}
    except Exception as e:
        return {"error": f"Failed to parse {sig.name}.py: {e}", "function": None}
    
    # Find the function definition
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == sig.name:
            func_node = node
            break
    
    if func_node is None:
        return {"error": f"Function '{sig.name}' not found in interfaces/{sig.name}.py", "function": None}
    
    # 3. Validate parameter names
    expected_params = sig.get_expected_params()
    actual_params = [arg.arg for arg in func_node.args.args]
    
    if actual_params != expected_params:
        return {
            "error": f"Parameter mismatch. Expected: ({', '.join(expected_params)}), Got: ({', '.join(actual_params)})",
            "function": None
        }
    
    # 4. Check for return statement
    has_return = False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            has_return = True
            break
    
    if not has_return:
        return {"error": f"Function '{sig.name}' has no return statement with a value", "function": None}
    
    # 5. Try to import and get the function
    try:
        func = _import_function(file_path, sig.name)
    except Exception as e:
        return {"error": f"Import failed: {e}", "function": None}
    
    return {"error": None, "function": func}


def _import_function(file_path: Path, func_name: str) -> Callable:
    """
    Dynamically import a function from a file.
    
    Args:
        file_path: Path to the Python file
        func_name: Name of the function to import
        
    Returns:
        The imported function
        
    Raises:
        Exception: If import fails
    """
    module_name = f"interfaces_{func_name}_{id(file_path)}"
    
    # Clean up if already imported
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {e}")
    
    if not hasattr(module, func_name):
        del sys.modules[module_name]
        raise ImportError(f"Module does not have function '{func_name}'")
    
    return getattr(module, func_name)


def load_interfaces_from_init(iter_dir: Path) -> Dict[str, Callable]:
    """
    Load all interfaces from interfaces/__init__.py.
    
    This is used during evaluation to load the complete interface module.
    
    Args:
        iter_dir: Iteration directory
        
    Returns:
        Dict mapping function names to callables
    """
    interfaces_dir = iter_dir / "interfaces"
    init_file = interfaces_dir / "__init__.py"
    
    if not init_file.exists():
        raise FileNotFoundError(f"interfaces/__init__.py not found in {iter_dir}")
    
    module_name = f"interfaces_module_{id(iter_dir)}"
    
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {init_file}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    # Add interfaces directory to path for relative imports
    interfaces_parent = str(interfaces_dir.parent)
    if interfaces_parent not in sys.path:
        sys.path.insert(0, interfaces_parent)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load interfaces module: {e}")
    
    # Get exported names
    if hasattr(module, '__all__'):
        names = module.__all__
    else:
        names = [n for n in dir(module) if not n.startswith('_')]
    
    interfaces = {}
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            interfaces[name] = obj
    
    return interfaces


def format_validation_feedback(result: ValidationResult) -> str:
    """
    Format validation result as feedback for the agent.
    
    Args:
        result: ValidationResult from validate_interfaces
        
    Returns:
        Formatted string for agent prompt
    """
    if result.success:
        return f"All {len(result.interfaces)} interfaces validated successfully."
    
    lines = ["Interface validation failed. Please fix the following issues:", ""]
    
    for error in result.errors:
        lines.append(f"- {error}")
    
    lines.extend([
        "",
        "Remember:",
        "- Each interface must be in `interfaces/{name}.py`",
        "- Function name must match exactly",
        "- Parameter names must match the signature",
        "- Function must have a return statement",
        "- Export functions in `interfaces/__init__.py`",
    ])
    
    return "\n".join(lines)
