"""Typed conversation message history for kernel optimization sessions.

Provides structured message tracking with role-based typing, enabling
smart compaction and multi-turn context assembly for the LLM agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_RESULT = "tool_result"
    OPTIMIZATION_EVENT = "optimization_event"


@dataclass
class KernelMessage:
    role: str  # from MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata examples:
    # tool_result: {"tool_name": "explain_iteration", "args": {"iteration": 9}}
    # optimization_event: {"event": "kept", "speedup": 2.84, "iteration": 11}
    # assistant: {"tokens_used": 500, "cost": 0.001, "model": "groq/llama-3.3-70b"}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


class ConversationHistory:
    """Typed conversation history with context assembly and compaction."""

    def __init__(self, max_messages: int = 200):
        self._messages: list[KernelMessage] = []
        self._max_messages = max_messages

    def add(self, role: str, content: str, **metadata) -> KernelMessage:
        """Add a message to history."""
        msg = KernelMessage(role=role, content=content, metadata=metadata)
        self._messages.append(msg)
        # Evict oldest messages if over the cap
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]
        return msg

    def add_user(self, content: str) -> KernelMessage:
        return self.add(MessageRole.USER, content)

    def add_assistant(self, content: str, **metadata) -> KernelMessage:
        return self.add(MessageRole.ASSISTANT, content, **metadata)

    def add_tool_result(self, tool_name: str, result: str, args: dict | None = None) -> KernelMessage:
        return self.add(
            MessageRole.TOOL_RESULT,
            result,
            tool_name=tool_name,
            args=args or {},
        )

    def add_optimization_event(self, event: str, **data) -> KernelMessage:
        return self.add(MessageRole.OPTIMIZATION_EVENT, event, **data)

    def add_system(self, content: str) -> KernelMessage:
        return self.add(MessageRole.SYSTEM, content)

    def get_messages(self, last_n: int | None = None) -> list[KernelMessage]:
        if last_n is None:
            return list(self._messages)
        return list(self._messages[-last_n:])

    def get_context_for_llm(self, max_tokens: int = 3000) -> str:
        """Build context string for LLM prompt from conversation history.

        Prioritizes: recent user messages > tool results > optimization events.
        Summarizes old messages when exceeding token budget.
        """
        if not self._messages:
            return ""

        max_chars = max_tokens * 4  # inverse of token estimate

        # Always include the most recent messages (up to last 10)
        recent = self._messages[-10:]

        parts: list[str] = []
        parts.append("--- CONVERSATION HISTORY ---")

        for msg in recent:
            role_label = msg.role.upper()
            line = f"[{role_label}] {msg.content}"

            # Add relevant metadata inline
            if msg.role == MessageRole.TOOL_RESULT:
                tool_name = msg.metadata.get("tool_name", "unknown")
                line = f"[TOOL:{tool_name}] {msg.content}"
            elif msg.role == MessageRole.OPTIMIZATION_EVENT:
                event_data = {k: v for k, v in msg.metadata.items() if k != "event"}
                if event_data:
                    extras = ", ".join(f"{k}={v}" for k, v in event_data.items())
                    line = f"[OPT_EVENT] {msg.content} ({extras})"

            parts.append(line)

        context = "\n".join(parts)

        # Truncate if over budget
        if len(context) > max_chars:
            context = context[-max_chars:]
            # Find the first complete line after truncation
            newline_pos = context.find("\n")
            if newline_pos > 0:
                context = "...(truncated)\n" + context[newline_pos + 1:]

        return context

    def compact(self) -> str:
        """Compact old messages into a summary, keeping recent ones.

        Returns the summary that replaced the old messages.
        """
        if len(self._messages) <= 20:
            return ""

        # Keep the last 10 messages, summarize the rest
        old_messages = self._messages[:-10]
        recent_messages = self._messages[-10:]

        # Build summary from old messages
        user_count = sum(1 for m in old_messages if m.role == MessageRole.USER)
        assistant_count = sum(1 for m in old_messages if m.role == MessageRole.ASSISTANT)
        tool_count = sum(1 for m in old_messages if m.role == MessageRole.TOOL_RESULT)
        opt_events = [m for m in old_messages if m.role == MessageRole.OPTIMIZATION_EVENT]

        summary_parts = [
            f"Compacted {len(old_messages)} older messages: "
            f"{user_count} user, {assistant_count} assistant, "
            f"{tool_count} tool results.",
        ]

        # Summarize optimization events
        if opt_events:
            kept_events = [m for m in opt_events if m.content == "kept"]
            discard_events = [m for m in opt_events if m.content == "discarded"]
            error_events = [m for m in opt_events if m.content == "error"]
            summary_parts.append(
                f"Optimization events: {len(kept_events)} kept, "
                f"{len(discard_events)} discarded, {len(error_events)} errors."
            )
            # Include best speedup from events
            speedups = [
                m.metadata.get("speedup", 0.0)
                for m in kept_events
                if m.metadata.get("speedup")
            ]
            if speedups:
                summary_parts.append(f"Best speedup in compacted history: {max(speedups):.2f}x")

        # Include last few user questions for topical continuity
        user_msgs = [m for m in old_messages if m.role == MessageRole.USER]
        if user_msgs:
            last_topics = [m.content[:80] for m in user_msgs[-3:]]
            summary_parts.append("Recent topics: " + " | ".join(last_topics))

        summary = "\n".join(summary_parts)

        # Replace old messages with a single system summary + recent messages
        summary_msg = KernelMessage(
            role=MessageRole.SYSTEM,
            content=summary,
        )
        self._messages = [summary_msg] + recent_messages

        return summary

    def to_list(self) -> list[dict]:
        """Serialize for persistence."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata,
            }
            for msg in self._messages
        ]

    @classmethod
    def from_list(cls, data: list[dict]) -> "ConversationHistory":
        """Deserialize."""
        history = cls()
        for entry in data:
            msg = KernelMessage(
                role=entry.get("role", MessageRole.USER),
                content=entry.get("content", ""),
                timestamp=entry.get("timestamp", datetime.now(timezone.utc).isoformat()),
                metadata=entry.get("metadata", {}),
            )
            history._messages.append(msg)
        return history

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def token_estimate(self) -> int:
        return sum(_estimate_tokens(m.content) for m in self._messages)
