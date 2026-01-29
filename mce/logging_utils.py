"""
Logging utilities for formatting Claude Agent SDK messages in a human-readable way.
Uses Python's native logging module with both console and file output.
"""

import logging
from typing import Any, Optional
from datetime import datetime
from pathlib import Path
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted


class MessageFormatter:
    """Format Claude Agent SDK messages for human-readable output."""
    
    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
        'white': '\033[97m',
    }
    
    def __init__(self, use_colors: bool = True, minimal: bool = False):
        """
        Initialize the formatter.
        
        Args:
            use_colors: Whether to use ANSI colors in output
            minimal: Whether to use minimal console output mode
        """
        self.use_colors = use_colors
        self.minimal = minimal
        self.tool_count = 0
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _format_header(self, title: str, color: str = 'blue') -> str:
        """Format a section header."""
        separator = "=" * 80
        return f"\n{self._color(separator, color)}\n{self._color(title.upper(), color)}\n{self._color(separator, color)}"
    
    def _format_subheader(self, title: str, color: str = 'cyan') -> str:
        """Format a subsection header."""
        return f"{self._color(title, color)}"
    
    def format_system_message(self, message: Any) -> str:
        """Format SystemMessage."""
        if self.minimal:
            return ""  # Don't output anything in minimal mode for system messages
        
        output = [self._format_header("🔧 SYSTEM INITIALIZATION", "blue")]
        
        if hasattr(message, 'data') and isinstance(message.data, dict):
            data = message.data
            
            # Compact session info on one line
            session_id = data.get('session_id', 'N/A')[:8] + "..."  # Truncate session ID
            cwd = data.get('cwd', 'N/A')
            model = data.get('model', 'N/A')
            permission = data.get('permissionMode', 'N/A')
            
            output.append(f"Session: {self._color(session_id, 'cyan')} | Dir: {self._color(cwd, 'cyan')} | Model: {self._color(model, 'yellow')} | Mode: {permission}")
            
            # Available tools - compact
            tools = data.get('tools', [])
            if tools:
                output.append(f"Tools ({self._color(str(len(tools)), 'green')}): {', '.join(tools)}")
            
            # Skills - compact
            skills = data.get('skills', [])
            if skills:
                output.append(f"Skills: {self._color('✓', 'green')} {', '.join(skills)}")
        
        return "\n".join(output)
    
    def format_assistant_message(self, message: Any) -> str:
        """Format AssistantMessage."""
        if self.minimal:
            # In minimal mode, just count tools
            if hasattr(message, 'content'):
                content = message.content
                if not isinstance(content, list):
                    content = [content]
                for block in content:
                    if type(block).__name__ == 'ToolUseBlock':
                        self.tool_count += 1
            return ""  # Don't output anything in minimal mode for assistant messages
        
        output = [self._format_header("🤖 ASSISTANT", "green")]
        
        model_info = f" [{message.model}]" if hasattr(message, 'model') else ""
        
        if hasattr(message, 'content'):
            content = message.content
            # Handle content as either list or single item
            if not isinstance(content, list):
                content = [content]
            
            for block in content:
                block_type = type(block).__name__
                
                if block_type == 'TextBlock':
                    text = block.text if hasattr(block, 'text') else str(block)
                    # Show text inline without subheader
                    output.append(f"{text}")
                
                elif block_type == 'ToolUseBlock':
                    # Compact tool call format
                    tool_id = block.id[-8:]  # Last 8 chars of ID
                    output.append(f"🔧 {self._color(block.name, 'magenta')} [{tool_id}]{model_info}")
                    if hasattr(block, 'input') and block.input:
                        # Show input without truncation
                        input_items = []
                        for key, value in block.input.items():
                            value_str = str(value)
                            input_items.append(f"{key}={value_str}")
                        output.append(f"   {', '.join(input_items)}")
        
        return "\n".join(output)
    
    def format_user_message(self, message: Any) -> str:
        """Format UserMessage."""
        if self.minimal:
            return ""  # Don't output anything in minimal mode for user messages
        
        output = [self._format_header("👤 USER / TOOL RESULT", "cyan")]
        
        if hasattr(message, 'content'):
            content = message.content
            # Handle content as either list or single item
            if not isinstance(content, list):
                content = [content]
            
            for block in content:
                block_type = type(block).__name__
                
                if block_type == 'TextBlock':
                    text = block.text if hasattr(block, 'text') else str(block)
                    text = str(text)
                    # Show text inline
                    output.append(text)
                
                elif block_type == 'ToolResultBlock':
                    # Compact tool result format
                    tool_id = block.tool_use_id[-8:]  # Last 8 chars
                    status_icon = self._color('✗', 'red') if (hasattr(block, 'is_error') and block.is_error) else self._color('✓', 'green')
                    
                    output.append(f"🔧 Result [{tool_id}] {status_icon}")
                    
                    if hasattr(block, 'content'):
                        content = block.content
                        
                        # Handle content as either string or list
                        if isinstance(content, list):
                            content_str = json.dumps(content, indent=2)
                        else:
                            content_str = str(content)
                        
                        # Print full output without truncation
                        output.append(content_str)
        
        return "\n".join(output)
    
    def format_result_message(self, message: Any) -> str:
        """Format ResultMessage."""
        if self.minimal:
            return ""  # Don't output anything in minimal mode for result messages
        
        output = [self._format_header("✅ FINAL RESULT", "yellow")]
        
        # Compact status line
        status_parts = []
        
        if hasattr(message, 'subtype'):
            status_color = 'green' if message.subtype == 'success' else 'red'
            status_parts.append(f"Status: {self._color(message.subtype.upper(), status_color)}")
        
        # Timing info - compact
        if hasattr(message, 'duration_ms'):
            duration = f"{message.duration_ms}ms"
            if hasattr(message, 'duration_api_ms'):
                duration += f" (API: {message.duration_api_ms}ms)"
            status_parts.append(f"Duration: {self._color(duration, 'cyan')}")
        
        # Turns
        if hasattr(message, 'num_turns'):
            status_parts.append(f"Turns: {message.num_turns}")
        
        if status_parts:
            output.append(" | ".join(status_parts))
        
        # Usage stats - compact on one line
        if hasattr(message, 'usage') and message.usage:
            usage = message.usage
            if isinstance(usage, dict):
                in_tok = usage.get('input_tokens', 0)
                out_tok = usage.get('output_tokens', 0)
                cache_tok = usage.get('cache_read_input_tokens', 0)
                
                usage_str = f"Tokens: {self._color(f'{in_tok}', 'cyan')} in, {self._color(f'{out_tok}', 'cyan')} out"
                if cache_tok > 0:
                    usage_str += f", {self._color(f'{cache_tok}', 'green')} cached"
                output.append(usage_str)
        
        # Cost
        if hasattr(message, 'total_cost_usd'):
            output.append(f"Cost: {self._color(f'${message.total_cost_usd:.4f}', 'yellow')}")
        
        # Result text
        if hasattr(message, 'result') and message.result:
            output.append(f"\n{message.result}")
        
        return "\n".join(output)
    
    def format_message(self, message: Any) -> str:
        """Format any message type."""
        message_type = type(message).__name__
        
        formatters = {
            'SystemMessage': self.format_system_message,
            'AssistantMessage': self.format_assistant_message,
            'UserMessage': self.format_user_message,
            'ResultMessage': self.format_result_message,
        }
        
        formatter = formatters.get(message_type)
        if formatter:
            return formatter(message)
        else:
            output = [self._format_header(f"❓ {message_type}", "white")]
            output.append(f"  {str(message)}")
            return "\n".join(output)


def setup_run_logger(log_base_dir: str = "logs") -> Path:
    """
    Create a timestamped run directory for organizing logs.
    
    Args:
        log_base_dir: Base directory for logs (default: "logs")
        
    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(
    name: str = "mce",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    console_colors: bool = True,
    run_dir: Optional[Path] = None,
    agent_type: Optional[str] = None,
    iteration: Optional[int] = None,
    sub_iteration: Optional[int] = None,
    minimal_console: bool = False
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files (used if run_dir is None)
        log_level: Logging level (default: INFO)
        console_colors: Whether to use colors in console output
        run_dir: Run directory for organized logging (overrides log_dir if provided)
        agent_type: Agent type for naming ("meta", "base", "eval")
        iteration: Iteration number for naming
        sub_iteration: Sub-iteration number for naming (for online learning batches)
        minimal_console: Whether to use minimal console output
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent propagation to parent loggers to avoid duplicate output
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Determine log directory and file name
    if run_dir is not None:
        log_path = Path(run_dir)
        if agent_type and iteration is not None and sub_iteration is not None:
            log_file = log_path / f"{agent_type}_iter{iteration}_sub{sub_iteration}.log"
        elif agent_type and iteration is not None:
            log_file = log_path / f"{agent_type}_iter{iteration}.log"
        elif agent_type:
            log_file = log_path / f"{agent_type}.log"
        else:
            log_file = log_path / f"{name}.log"
    else:
        log_path = Path(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"session_{timestamp}.log"
    
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler with minimal or full output
    if not minimal_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        if console_colors:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (no colors, full output)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    if not minimal_console:
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def log_message(
    message: Any, 
    logger: logging.Logger, 
    console_formatter: Optional[MessageFormatter] = None,
    file_formatter: Optional[MessageFormatter] = None,
    minimal_console: bool = False
) -> None:
    """
    Log a Claude Agent SDK message using the logger.
    Prints colored output to console and plain text to file.
    
    Args:
        message: Message object from Claude Agent SDK
        logger: Logger instance
        console_formatter: Optional MessageFormatter for console (with colors)
        file_formatter: Optional MessageFormatter for file (without colors)
        minimal_console: Whether to use minimal console output
    """
    if console_formatter is None:
        console_formatter = MessageFormatter(use_colors=True, minimal=minimal_console)
    if file_formatter is None:
        file_formatter = MessageFormatter(use_colors=False, minimal=False)
    
    message_type = type(message).__name__
    
    # Format for console (with colors, possibly minimal)
    formatted_console = console_formatter.format_message(message)
    
    # Format for file (without colors, always full)
    formatted_file = file_formatter.format_message(message)
    
    # Print to console directly (bypasses logger to show colors properly)
    if formatted_console:  # Only print if there's content (minimal mode may return empty)
        print(formatted_console)
    
    # Log to file handlers only (without colors)
    # Temporarily disable console handler
    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    for handler in console_handlers:
        logger.removeHandler(handler)
    
    # Log with appropriate level (goes to file only now)
    if message_type in ['SystemMessage', 'AssistantMessage', 'UserMessage', 'ResultMessage']:
        logger.info(f"\n{formatted_file}")
    else:
        logger.debug(f"\n{formatted_file}")
    
    # Restore console handlers
    for handler in console_handlers:
        logger.addHandler(handler)


# Example usage
if __name__ == "__main__":
    print("""
Logging Utils for Claude Agent SDK (Native Python Logging)
===========================================================

Usage:

1. Basic setup:
    from logging_utils import setup_logger, log_message
    
    # Set up logger (creates timestamped log file automatically)
    logger = setup_logger(name="my_agent", log_dir="logs")
    
    # Use in your code (formatters created automatically)
    async for message in query(prompt, options):
        log_message(message, logger)

2. Custom configuration:
    # Without colors
    logger = setup_logger(console_colors=False)
    
    # Different log level
    logger = setup_logger(log_level=logging.DEBUG)
    
    # Custom directory
    logger = setup_logger(log_dir="my_logs")

Features:
- Automatically creates timestamped log files: logs/session_YYYYMMDD_HHMMSS.log
- Colored console output (optional)
- Plain text file output
- Uses Python's native logging module
- Both console and file output simultaneously
""")
