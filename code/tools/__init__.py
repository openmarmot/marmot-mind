import json
from typing import Callable, Dict

from .run_terminal import _RUN_TERMINAL_TOOL, execute_run_terminal
from .web_search import _WEB_SEARCH_TOOL, execute_web_search
from .chat import _POST_MESSAGE_TOOL, execute_post_message
from .look_at_image import _LOOK_AT_IMAGE_TOOL, execute_look_at_image
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
from .context import ToolContext

BASE_TOOLS = [
    _POST_MESSAGE_TOOL,
    _RUN_TERMINAL_TOOL,
    _LOOK_AT_IMAGE_TOOL,
    _SET_FOCUS_TOOL,
    _LOG_OBSERVATION_TOOL,
    _PLAN_WAKE_TOOL,
    _WRITE_NEXT_STEPS_TOOL,
    _UPDATE_GOALS_TOOL,
    _REMEMBER_TOOL,
]
WEB_SEARCH_TOOL = _WEB_SEARCH_TOOL

_TOOL_EXECUTORS: Dict[str, Callable] = {
    "post_message": lambda ctx, args: execute_post_message(
        ctx, args.get("text", ""), args.get("tags")
    ),
    "run_terminal": lambda ctx, args: execute_run_terminal(ctx, args.get("command", "")),
    "look_at_image": lambda ctx, args: execute_look_at_image(
        ctx, args.get("source", ""), args.get("focus", "")
    ),
    "web_search": lambda ctx, args: execute_web_search(
        ctx, args.get("query", ""), args.get("max_results", 5)
    ),
    "set_focus": lambda ctx, args: execute_set_focus(args.get("text", "")),
    "log_observation": lambda ctx, args: execute_log_observation(args.get("note", "")),
    "plan_next_wake": lambda ctx, args: execute_plan_next_wake(
        args.get("delay_seconds", 300), args.get("reason", "")
    ),
    "write_next_steps": lambda ctx, args: execute_write_next_steps(args.get("steps", "")),
    "update_goals": lambda ctx, args: execute_update_goals(args.get("goals", "")),
    "remember": lambda ctx, args: execute_remember(args.get("note", "")),
}


def execute_tool(ctx: ToolContext, tool_call: dict) -> str:
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except Exception:
        args = {}
    executor = _TOOL_EXECUTORS.get(name)
    if executor:
        return executor(ctx, args)
    return f"Error: unknown tool {name}"


def get_tools(web_search_enabled: bool = False) -> list:
    tools = list(BASE_TOOLS)
    if web_search_enabled:
        tools.append(WEB_SEARCH_TOOL)
    return tools
