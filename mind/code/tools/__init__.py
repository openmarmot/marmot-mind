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
)
from .chat import (
    _POST_MESSAGE_TOOL,
    execute_post_message,
    set_post_handler,
)
from .mind_tools import (
    _SET_FOCUS_TOOL,
    _LOG_OBSERVATION_TOOL,
    _PLAN_WAKE_TOOL,
    _WRITE_NEXT_STEPS_TOOL,
    _UPDATE_GOALS_TOOL,
    _REMEMBER_TOOL,
    execute_set_focus,
    execute_log_observation,
    execute_plan_next_wake,
    execute_write_next_steps,
    execute_update_goals,
    execute_remember,
)

BASE_TOOLS = [
    _POST_MESSAGE_TOOL,
    _RUN_TERMINAL_TOOL,
    _SET_FOCUS_TOOL,
    _LOG_OBSERVATION_TOOL,
    _PLAN_WAKE_TOOL,
    _WRITE_NEXT_STEPS_TOOL,
    _UPDATE_GOALS_TOOL,
    _REMEMBER_TOOL,
]
WEB_SEARCH_TOOL = _WEB_SEARCH_TOOL

_TOOL_EXECUTORS: Dict[str, Callable[[dict], str]] = {
    "post_message": lambda args: execute_post_message(
        args.get("text", ""), args.get("tags")
    ),
    "run_terminal": lambda args: execute_run_terminal(args.get("command", "")),
    "web_search": lambda args: execute_web_search(
        args.get("query", ""), args.get("max_results", 5)
    ),
    "set_focus": lambda args: execute_set_focus(args.get("text", "")),
    "log_observation": lambda args: execute_log_observation(args.get("note", "")),
    "plan_next_wake": lambda args: execute_plan_next_wake(
        args.get("delay_seconds", 300), args.get("reason", "")
    ),
    "write_next_steps": lambda args: execute_write_next_steps(args.get("steps", "")),
    "update_goals": lambda args: execute_update_goals(args.get("goals", "")),
    "remember": lambda args: execute_remember(args.get("note", "")),
}


def execute_tool(tool_call: dict) -> str:
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


def configure_tools(
    *,
    tool_calls_dir: str | None = None,
    brave_api_key: str | None = None,
    post_handler=None,
):
    if tool_calls_dir:
        set_tool_calls_dir(tool_calls_dir)
    if brave_api_key is not None:
        configure_web_search(brave_api_key)
    if post_handler is not None:
        set_post_handler(post_handler)


def get_tools(web_search_enabled: bool = False) -> list:
    tools = list(BASE_TOOLS)
    if web_search_enabled:
        tools.append(WEB_SEARCH_TOOL)
    return tools
