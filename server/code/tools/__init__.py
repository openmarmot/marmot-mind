import json
from typing import Callable, Dict

from .run_terminal import (
    _RUN_TERMINAL_TOOL,
    execute_run_terminal,
    set_tool_calls_dir,
)
from .web_search import (
    _WEB_SEARCH_TOOL,
    execute_web_search,
    configure as configure_web_search,
    set_brave_api_key,
)
from .speak import _SPEAK_TOOL, execute_speak
from .mind import (
    _SET_FOCUS_TOOL,
    _LOG_OBSERVATION_TOOL,
    _PLAN_WAKE_TOOL,
    execute_set_focus,
    execute_log_observation,
    execute_plan_next_wake,
)

# Public re-exports for server
BASE_TOOLS = [_RUN_TERMINAL_TOOL, _SPEAK_TOOL, _SET_FOCUS_TOOL, _LOG_OBSERVATION_TOOL, _PLAN_WAKE_TOOL]
WEB_SEARCH_TOOL = _WEB_SEARCH_TOOL

# Simple registry for clean dispatch + easy future extension
_TOOL_EXECUTORS: Dict[str, Callable[[dict], str]] = {
    "run_terminal": lambda args: execute_run_terminal(args.get("command", "")),
    "web_search": lambda args: execute_web_search(
        args.get("query", ""), args.get("max_results", 5)
    ),
    "speak": lambda args: execute_speak(args.get("text", "")),
    "set_focus": lambda args: execute_set_focus(args.get("text", "")),
    "log_observation": lambda args: execute_log_observation(args.get("note", "")),
    "plan_next_wake": lambda args: execute_plan_next_wake(
        args.get("delay_seconds", 300), args.get("reason", "")
    ),
}


def execute_tool(tool_call: dict) -> str:
    """Dispatch a tool call (as received from the LLM) to the right executor.
    The executors are pure with respect to server state; dependencies are
    injected via the configure_* functions below.
    """
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except Exception:
        args = {}

    executor = _TOOL_EXECUTORS.get(name)
    if executor:
        return executor(args)
    return f"Error: unknown tool {name}"


# Configuration helpers (called by server at startup)
def configure_tools(*, tool_calls_dir: str | None = None, brave_api_key: str | None = None):
    """One-shot configuration for all tools that need external values."""
    if tool_calls_dir:
        set_tool_calls_dir(tool_calls_dir)
    if brave_api_key is not None:
        configure_web_search(brave_api_key)
