# backend/tools/error_handling.py
import inspect
import subprocess
from functools import wraps


def resilient_tool(max_retries: int = 1):
    """
    Decorator to catch exceptions and return user-friendly error hints.
    Retries on transient errors (timeouts, network issues).
    Preserves the original function signature for LangChain tool binding.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Convert non-string results to string for LLM consumption
                    if isinstance(result, dict):
                        import json
                        return json.dumps(result, indent=2)
                    elif isinstance(result, list):
                        return "\n".join(str(x) for x in result)
                    return result if isinstance(result, str) else str(result)
                except subprocess.TimeoutExpired as e:
                    last_error = e
                    if attempt < max_retries:
                        continue
                    return (
                        f"[TimeoutError: command took >60 seconds. "
                        f"Break into smaller steps or use run_python_code for quick tests.]"
                    )
                except FileNotFoundError as e:
                    return (
                        f"[FileNotFoundError: {e}. "
                        f"Hint: use get_file_structure('.') to verify the path]"
                    )
                except PermissionError as e:
                    return (
                        f"[PermissionError: {e}. "
                        f"Check file permissions or run with appropriate user]"
                    )
                except Exception as e:
                    return f"[Error in {func.__name__}: {type(e).__name__}: {str(e)}]"
            return f"[Failed after {max_retries + 1} attempts: {last_error}]"
        
        # Preserve original signature for LangChain tool binding
        wrapper.__signature__ = inspect.signature(func)
        return wrapper
    return decorator
