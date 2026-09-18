import uuid

import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(
        uuid.uuid4()
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("RAG Configuration")

    st.caption(
        "Production Document Q&A System"
    )

    top_k = st.slider(
        "Top-K Retrieval",
        min_value=1,
        max_value=10,
        value=5,
    )

    source_filter = st.text_input(
        "Source Filter",
        placeholder="example.pdf",
    )

    page_filter = st.number_input(
        "Page Filter",
        min_value=0,
        value=0,
        step=1,
    )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):

        try:
            response = requests.delete(
                f"{API_URL}/api/v1/chat/"
                f"{st.session_state.session_id}",
                timeout=30,
            )

            if response.ok:
                st.session_state.messages = []

                st.success(
                    "Conversation cleared."
                )

            else:
                st.error(
                    "Failed to clear conversation."
                )

        except requests.RequestException:
            st.error(
                "Could not connect to FastAPI."
            )

    st.divider()

    st.caption(
        f"Session: "
        f"{st.session_state.session_id[:8]}..."
    )


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.title(
    "📚 Production Document Q&A"
)

st.write(
    "Ask questions about your document knowledge base."
)


# ---------------------------------------------------------
# Display conversation
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("citations")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for citation in message[
                    "citations"
                ]:

                    st.markdown(
                        f"**{citation['source']}**  \n"
                        f"Page: {citation['page']}  \n"
                        f"Score: "
                        f"{citation['score']:.4f}"
                    )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # -----------------------------------------------------
    # Build API request
    # -----------------------------------------------------

    payload = {
        "question": question,
        "session_id": (
            st.session_state.session_id
        ),
        "top_k": top_k,
    }

    if source_filter.strip():

        payload["source"] = (
            source_filter.strip()
        )

    if page_filter > 0:

        payload["page"] = page_filter

    # -----------------------------------------------------
    # Call FastAPI
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/api/v1/chat",
                    json=payload,
                    timeout=120,
                )

                if not response.ok:

                    st.error(
                        "The RAG API returned an error."
                    )

                    st.stop()

                result = response.json()

                answer = result["answer"]

                citations = result.get(
                    "citations",
                    [],
                )

                # -----------------------------------------
                # Answer
                # -----------------------------------------

                st.markdown(answer)

                # -----------------------------------------
                # Sources
                # -----------------------------------------

                if citations:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for citation in citations:

                            st.markdown(
                                f"**{citation['source']}**  \n"
                                f"Page: "
                                f"{citation['page']}  \n"
                                f"Similarity: "
                                f"{citation['score']:.4f}"
                            )

                # -----------------------------------------
                # Save assistant message
                # -----------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )

            except requests.RequestException as exc:

                st.error(
                    "Unable to connect to the "
                    "FastAPI backend."
                )

                st.caption(
                    str(exc)
                )