import os
import subprocess

_RUN_TERMINAL_TIMEOUT = 300
_TOOL_CALLS_DIR = None


def set_tool_calls_dir(path: str):
    global _TOOL_CALLS_DIR
    _TOOL_CALLS_DIR = path


_RUN_TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_terminal",
        "description": (
            "Execute a Linux bash command in your private tool-calls workspace. "
            "Returns exit code + stdout + stderr. Prefer non-destructive commands. "
            "Use for investigation, file work, system checks. Do not post raw dumps to chat — summarize."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command, e.g. 'ls -la', 'date', 'cat notes.txt'",
                }
            },
            "required": ["command"],
        },
    },
}


def execute_run_terminal(command: str) -> str:
    if not command or not command.strip():
        return "Error: empty command"
    workdir = _TOOL_CALLS_DIR or "."
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_RUN_TERMINAL_TIMEOUT,
            cwd=workdir,
            env={**os.environ},
        )
        parts = [f"Exit code: {result.returncode}"]
        if result.stdout:
            out = result.stdout
            if len(out) > 7000:
                out = out[:7000] + "\n[truncated]"
            parts.append("STDOUT:\n" + out)
        if result.stderr:
            err = result.stderr
            if len(err) > 4000:
                err = err[:4000] + "\n[truncated]"
            parts.append("STDERR:\n" + err)
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {_RUN_TERMINAL_TIMEOUT}s"
    except Exception as e:
        return f"Error: {str(e)}"
