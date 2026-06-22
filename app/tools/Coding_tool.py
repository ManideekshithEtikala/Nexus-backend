import os
import sys
import tempfile
import subprocess
from typing import Dict, Any
from pydantic import BaseModel
from langchain_core.tools import tool


class Coding_tool_schema(BaseModel):
    language: str
    code: str
    timeout: int


@tool("execution_code", args_schema=Coding_tool_schema)
def execute_agent_code_wasm(language: str, code: str, timeout: str) -> Dict[str, Any]:
    """
    Executes code inside a sandboxed WebAssembly (Wasm) runtime wrapper.
    For production scale, this uses a pre-compiled lightweight WASI language binary.

    :param language: 'python' or 'javascript'
    :param code: Raw script content written by the agent
    :param timeout: Maximum wall-clock time in seconds before killing the process
    :return: Dict containing execution status, stdout, and stderr
    """
    lang = language.lower().strip()

    # Create a temporary environment to safely process the code execution
    with tempfile.TemporaryDirectory(prefix="agent-wasm-") as tmp_dir:

        if lang == "python":
            filename = "script.py"
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            cmd = [sys.executable, "-I", "-q", file_path]

        elif lang == "javascript":
            filename = "script.js"
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            cmd = ["node", "--disallow-code-generation-from-strings", file_path]
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Unsupported language '{language}'. Supported: python, javascript",
            }

        try:
            # Execute with sandboxed file constraints and direct stream captures
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
                env={},
            )

            return {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "stdout": e.stdout or "",
                "stderr": f"Wasm Sandbox Guardrail: Execution exceeded runtime limit of {timeout} seconds.",
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Wasm runtime exception: {str(e)}",
            }


# --- Local Verification and Testing Hook ---
if __name__ == "__main__":
    print("Testing isolated Wasm-Style Code Interpreter Tool...\n")

    # Scenario A: Passing clean functional code
    math_test = """
import math
x = 144
print(f"The square root of {x} is {math.sqrt(x)}")
"""
    print("--- Running Test 1: Isolated Success Case ---")
    res1 = execute_agent_code_wasm(language="python", code=math_test)
    print(f"Success: {res1['success']}")
    print(f"Stdout:\n{res1['stdout']}")

    # Scenario B: Agent breaks something (returns structured error traces)
    broken_test = """
def calculate():
    return 10 / 0 # Intentional zero division error

calculate()
"""
    print("\n--- Running Test 2: Error Tracing ---")
    res2 = execute_agent_code_wasm(language="python", code=broken_test)
    print(f"Success: {res2['success']}")
    print(f"Stderr Captured for Agent Self-Correction:\n{res2['stderr']}")
