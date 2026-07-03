import os
import subprocess

_RUN_TERMINAL_TIMEOUT = 30

_TOOL_CALLS_DIR = None  # set by server via set_tool_calls_dir() at startup


def set_tool_calls_dir(path: str):
    """Configure the working directory used for run_terminal executions.
    Must be called by the server after it determines AGENT_DATA/tool-calls.
    """
    global _TOOL_CALLS_DIR
    _TOOL_CALLS_DIR = path


_RUN_TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_terminal",
        "description": "Execute a Linux bash command (cwd is the dedicated tool-calls workspace under agent-data/tool-calls/). Returns exit code + stdout + stderr. Use to explore files, run commands, check processes, edit via echo/cat etc. Prefer non-destructive commands when possible. Created files stay isolated from Marmot's own data (e.g. memory.txt). After getting results you can continue with more tools; use speak() if/when you want to tell the user anything. Never output answers as plain assistant content.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run, e.g. 'ls -la', 'ps aux | head', 'cat README.md'"}
            },
            "required": ["command"]
        }
    }
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
            env={**os.environ}
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
        result_text = "\n".join(parts)
        result_text += "\n\n(Reminder: If you want to tell the user anything based on this result, call the speak() tool with natural spoken text. You can continue using other tools before speaking. Do not output the information as plain assistant content.)"
        return result_text
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {_RUN_TERMINAL_TIMEOUT}s"
    except Exception as e:
        return f"Error: {str(e)}"
