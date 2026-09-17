from typing import Dict, List, Any


class ConversationManager:
    """
    Simple in-memory conversation manager.

    Each conversation is identified by a session_id.
    """

    def __init__(self):
        self._conversations: Dict[
            str, List[Dict[str, str]]
        ] = {}

    def create_session(self, session_id: str) -> None:
        """Create a conversation session."""

        if session_id not in self._conversations:
            self._conversations[session_id] = []

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add a message to a conversation."""

        self.create_session(session_id)

        self._conversations[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def get_history(
        self,
        session_id: str,
    ) -> List[Dict[str, str]]:
        """Return conversation history."""

        return self._conversations.get(
            session_id,
            [],
        )

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """Delete conversation history."""

        self._conversations.pop(
            session_id,
            None,
        )

    def get_session_count(self) -> int:
        """Return number of active sessions."""

        return len(self._conversations)