import os
import secrets
import signal
import subprocess

from clock import utcnow

_RUN_TERMINAL_TIMEOUT = 300
# Keep the tool reply readable. A bigger dump is written to the workspace.
_INLINE_LIMIT = 32_000
_PREVIEW = 2_000

_RUN_TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_terminal",
        "description": (
            "Execute a Linux bash command in your private tool-calls workspace. "
            "Returns exit code + stdout + stderr. Prefer non-destructive commands. "
            "Use for investigation, file work, system checks. Do not post raw dumps to chat — summarize. "
            "For a large page or dump, write it to a file here and inspect that file "
            "(grep, head, a short script). Do not curl a whole HTML document to stdout. "
            "If stdout or stderr is huge, it is saved to a file here and this reply "
            "only includes a short preview."
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


def _spill_stream(label: str, text: str, workdir: str) -> str:
    """Return a stream for the tool reply, spilling a huge dump to a file."""
    if not text:
        return ""
    if len(text) <= _INLINE_LIMIT:
        return f"{label}:\n{text}"
    os.makedirs(workdir, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"terminal-{label.lower()}-{stamp}-{secrets.token_hex(3)}.txt"
    path = os.path.join(workdir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    preview = text[:_PREVIEW]
    return (
        f"{label} is {len(text)} characters — saved to {name}. "
        f"Inspect that file (grep, head). First {_PREVIEW} characters:\n{preview}"
    )


def execute_run_terminal(ctx, command: str) -> str:
    if not command or not command.strip():
        return "Error: empty command"
    workdir = (ctx.tool_calls_dir if ctx else None) or "."
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            env={**os.environ},
            start_new_session=True,
        )
    except Exception as e:
        return f"Error: {str(e)}"

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=_RUN_TERMINAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        stdout, stderr = proc.communicate()

    parts = []
    if timed_out:
        parts.append(f"Error: timed out after {_RUN_TERMINAL_TIMEOUT}s (process killed)")
    parts.append(f"Exit code: {proc.returncode}")
    out = _spill_stream("STDOUT", stdout or "", workdir)
    if out:
        parts.append(out)
    err = _spill_stream("STDERR", stderr or "", workdir)
    if err:
        parts.append(err)
    return "\n".join(parts)
