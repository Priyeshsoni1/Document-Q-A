
import os
import uuid

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

CHAT_ENDPOINT = f"{API_URL}/api/v1/chat"


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Priyesh Soni | AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Session State
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None


# =========================================================
# Suggested Questions
# =========================================================

SUGGESTED_QUESTIONS = [
    "Who is Priyesh Soni? Tell me about his background.",
    "Tell me everything about Priyesh Soni.",
    "What are Priyesh's technical skills?",
    "Explain Priyesh's AgentForge project in detail.",
    "What technologies has Priyesh used in his AI projects?",
    "What are Priyesh's career goals?",
]


# =========================================================
# Helper Functions
# =========================================================

def get_score(score):
    """Format a retrieval score safely."""

    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def render_citations(citations):
    """Display retrieved document sources."""

    if not citations:
        return

    with st.expander(
        "📚 Retrieved Sources",
        expanded=False,
    ):
        for index, citation in enumerate(
            citations,
            start=1,
        ):
            source = citation.get(
                "source",
                "Unknown source",
            )

            page = citation.get(
                "page",
                "N/A",
            )

            score = get_score(
                citation.get("score")
            )

            st.markdown(
                f"**{index}. {source}**"
            )

            st.caption(
                f"Page: {page} | "
                f"Retrieval score: {score}"
            )

            if index < len(citations):
                st.divider()


def render_message(message):
    """Render a conversation message."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_citations(
                message.get("citations", [])
            )


def ask_question(question):
    """
    Send a question to the existing FastAPI RAG API
    and save the conversation in Streamlit session state.
    """

    question = question.strip()

    if not question:
        return

    # Save and display the user's question.
    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    # Build the existing backend request.
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "top_k": st.session_state.top_k,
    }

    source = st.session_state.source_filter.strip()

    if source:
        payload["source"] = source

    page = st.session_state.page_filter

    if page > 0:
        payload["page"] = page

    # Call FastAPI.
    with st.chat_message("assistant"):
        with st.spinner(
            "Searching Priyesh's knowledge base..."
        ):
            try:
                response = requests.post(
                    CHAT_ENDPOINT,
                    json=payload,
                    timeout=(5, 120),
                )

                response.raise_for_status()

                result = response.json()

                answer = result.get("answer")

                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(
                        "The API response did not contain "
                        "a valid answer."
                    )

                citations = result.get("citations", [])

                if not isinstance(citations, list):
                    citations = []

                st.markdown(answer)
                render_citations(citations)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )

            except requests.Timeout:
                error_message = (
                    "The request timed out. "
                    "Please try again with a simpler question."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except requests.HTTPError as exc:
                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else "Unknown"
                )

                error_message = (
                    f"The RAG API returned HTTP "
                    f"{status_code}. Check the backend logs."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except (requests.RequestException, ValueError) as exc:
                error_message = (
                    "Unable to get an answer from the "
                    "knowledge base. Check the API connection "
                    "and response format."
                )

                st.error(error_message)

                # Detailed error is useful during local development.
                st.caption(str(exc))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.title("⚙️ RAG Configuration")

    st.caption(
        "Priyesh Soni — Personal Knowledge Assistant"
    )

    st.divider()

    st.subheader("Retrieval Settings")

    st.slider(
        "Top-K Retrieval",
        min_value=1,
        max_value=10,
        value=5,
        key="top_k",
        help=(
            "Number of retrieved document chunks "
            "sent to the answering pipeline."
        ),
    )

    st.text_input(
        "Source Filter",
        placeholder="example.pdf",
        key="source_filter",
        help="Restrict retrieval to a document source.",
    )

    st.number_input(
        "Page Filter",
        min_value=0,
        value=0,
        step=1,
        key="page_filter",
        help="Use 0 to search across all pages.",
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.markdown(
        """
        Ask about:

        - 👤 Personal and professional background
        - 💻 Technical skills
        - 🤖 AI and LLM engineering
        - 🔎 RAG architecture
        - 🛠️ Projects and technologies
        - 🎯 Career goals
        """
    )

    st.caption(
        "Answers depend on the documents indexed "
        "by your backend."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        try:
            response = requests.delete(
                f"{API_URL}/api/v1/chat/"
                f"{st.session_state.session_id}",
                timeout=(5, 30),
            )

            response.raise_for_status()

            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.suggested_question = None

            st.rerun()

        except requests.RequestException as exc:
            st.error(
                "Could not clear the backend conversation."
            )
            st.caption(str(exc))

    st.divider()

    st.caption(f"API: {API_URL}")

    st.caption(
        f"Session: "
        f"{st.session_state.session_id[:8]}..."
    )


# =========================================================
# Main Interface
# =========================================================

st.title("🤖 Ask About Priyesh Soni")

st.markdown(
    """
    Your personal AI knowledge assistant, powered by
    **Retrieval-Augmented Generation (RAG)**.

    Ask questions about Priyesh's background, technical skills,
    AI engineering journey, projects, and professional goals.
    Answers are generated from the indexed knowledge base
    and include document citations when available.
    """
)

st.divider()


# =========================================================
# Suggested Questions
# =========================================================

if not st.session_state.messages:

    st.subheader("💡 Start with a question")

    st.write(
        "Select a question below or type your own."
    )

    columns = st.columns(2)

    for index, suggested_question in enumerate(
        SUGGESTED_QUESTIONS
    ):
        with columns[index % 2]:
            if st.button(
                suggested_question,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.suggested_question = (
                    suggested_question
                )
                st.rerun()


# =========================================================
# Display Conversation History
# =========================================================

for message in st.session_state.messages:
    render_message(message)


# =========================================================
# Chat Input
# =========================================================

question = st.chat_input(
    "Ask anything about Priyesh Soni..."
)

# Suggested questions and typed questions use the same
# backend request path.
if st.session_state.suggested_question:
    question = st.session_state.suggested_question
    st.session_state.suggested_question = None

if question:
    ask_question(question)