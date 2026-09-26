"""Per-think-loop tool state. Minds run concurrently, so nothing here is process-global."""


class ToolContext:
    def __init__(self, tool_calls_dir: str, post_handler, brave_api_key: str | None = None):
        self.tool_calls_dir = tool_calls_dir
        self.post_handler = post_handler
        self.brave_api_key = (brave_api_key or "").strip() or None
        self.pending_images: list[dict] = []

    def take_pending_images(self) -> list[dict]:
        out = self.pending_images
        self.pending_images = []
        return out
