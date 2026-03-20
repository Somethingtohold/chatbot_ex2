"""
Career Advisor Bot — Streamlit Application
==========================================
An AI-powered career advisor specialising in data science and technology careers.

Features:
  - Conversational interface with full chat history
  - Advanced RAG with MultiQueryRetriever (query translation)
  - 3 domain-specific tools
  - Source attribution panel for every RAG-based response
  - Tool call results displayed inline
  - Rate limiting (max 20 requests per session)
  - Input validation and error handling
"""

import logging
import os
import time
from pathlib import Path

import tiktoken
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MAX_REQUESTS_PER_SESSION = 20
MAX_INPUT_LENGTH = 1000
MODEL_NAME = "gpt-4o-mini"

try:
    SYSTEM_PROMPT = Path("system_prompt.txt").read_text(encoding="utf-8")
except Exception as e:
    logger.error(f"Failed to load system_prompt.txt: {e}")
    SYSTEM_PROMPT = None


# ---------------------------------------------------------------------------
# Cached initialisation (only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialising knowledge base...")
def initialise_agent():
    """
    Build the LangChain agent with all tools.
    Cached so ChromaDB and embeddings are only initialised once.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

    # Import here to avoid circular deps at module load
    from rag import build_retriever_tool, get_retriever, get_vectorstore
    from tools import estimate_salary, fetch_recent_jobs, course_recommender

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.3,
        api_key=api_key,
    )

    vectorstore = get_vectorstore()
    retriever = get_retriever(vectorstore, llm)
    retriever_tool = build_retriever_tool(retriever)

    tools = [
        retriever_tool,
        estimate_salary,
        fetch_recent_jobs,
        course_recommender,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )

    return agent_executor, retriever


def count_tokens(text: str) -> int:
    """Count tokens in a string using the GPT-4o-mini encoding."""
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def retrieve_context(query: str, retriever) -> tuple[str, list]:
    """Run the retriever and return (formatted context string, raw doc list)."""
    from rag import format_retrieved_docs
    try:
        docs = retriever.invoke(query)
        return format_retrieved_docs(docs), docs
    except Exception as exc:
        logger.warning(f"Retrieval failed: {exc}")
        return "", []


def run_agent(query: str, chat_history: list, context: str, agent_executor) -> dict:
    """Invoke the agent and return its full response dict."""
    return agent_executor.invoke({
        "input": query,
        "context": context,
        "chat_history": chat_history,
    })


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_input(text: str) -> tuple[bool, str]:
    """Return (is_valid, error_message)."""
    text = text.strip()
    if not text:
        return False, "Please enter a question."
    if len(text) > MAX_INPUT_LENGTH:
        return False, (f"Input too long ({len(text)} chars). "
                       f"Please keep it under {MAX_INPUT_LENGTH} characters.")
    forbidden = ["<script", "javascript:", "SELECT ", "DROP TABLE"]
    for pattern in forbidden:
        if pattern.lower() in text.lower():
            return False, "Input contains disallowed content."
    return True, ""


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "messages": [],
        "request_count": 0,
        "sources": {},
        "tool_calls": {},
        "last_request_time": 0.0,
        "total_tokens": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_lc_history() -> list:
    """Convert session messages to LangChain message objects."""
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def parse_tool_calls(intermediate_steps: list) -> list[dict]:
    """Extract tool name, input, and output from agent intermediate steps."""
    calls = []
    for action, observation in intermediate_steps:
        calls.append({
            "tool": action.tool,
            "input": action.tool_input,
            "output": str(observation),
        })
    return calls


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
TOOL_ICONS = {
    "career_knowledge_search": "",
    "estimate_salary": "",
    "fetch_recent_jobs": "",
    "course_recommender": "",
}

TOOL_LABELS = {
    "career_knowledge_search": "Knowledge Base Search",
    "estimate_salary": "Salary Estimator",
    "fetch_recent_jobs": "Recent Job Openings",
    "course_recommender": "Course Recommender",
}


def render_tool_calls(calls: list[dict]):
    """Render tool call details in an expander."""
    if not calls:
        return
    with st.expander(f"Tools used ({len(calls)})", expanded=False):
        for call in calls:
            tool_name = call["tool"]
            icon = TOOL_ICONS.get(tool_name, "🔧")
            label = TOOL_LABELS.get(tool_name, tool_name)
            st.markdown(f"**{icon} {label}**")
            if isinstance(call["input"], dict):
                for k, v in call["input"].items():
                    st.markdown(f"- *{k}:* `{v}`")
            else:
                st.markdown(f"- *Input:* `{call['input']}`")
            if tool_name != "career_knowledge_search":
                # Show output for non-retrieval tools (retrieval output shown in Sources)
                st.markdown("**Result:**")
                st.markdown(call["output"])
            st.markdown("---")


def render_sources(docs: list):
    """Render retrieved knowledge base sources in an expander."""
    if not docs:
        return

    # De-duplicate by content preview
    seen = set()
    unique_docs = []
    for doc in docs:
        preview = doc.page_content[:80]
        if preview not in seen:
            seen.add(preview)
            unique_docs.append(doc)

    with st.expander(f"Sources ({len(unique_docs)} passages retrieved)", expanded=False):
        for i, doc in enumerate(unique_docs, 1):
            source = doc.metadata.get("source", "knowledge base")
            source_name = Path(source).stem.replace("_", " ").title()
            st.markdown(f"**[{i}] {source_name}**")
            st.markdown(
                f"<div style='background:#f0f2f6;border-left:3px solid #4CAF50;"
                f"padding:8px 12px;border-radius:4px;font-size:0.87em;'>"
                f"{doc.page_content[:400]}{'...' if len(doc.page_content) > 400 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ── Page config ──────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Career Advisor Bot",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Career Advisor Bot")
        st.markdown("*Specialising in Data Science & Technology Careers*")
        st.markdown("---")

        st.subheader("About")
        st.markdown(
            "This bot uses **Retrieval-Augmented Generation (RAG)** with "
            "**Multi-Query Retrieval** to provide expert career advice grounded "
            "in a curated knowledge base.\n\n"
            "**Tools available:**\n"
            "- Salary estimator\n"
            "- Recent job openings\n"
            "- Course recommender\n"
            "- Knowledge base search"
        )
        st.markdown("---")

        st.subheader("Try asking:")
        example_questions = [
            "What skills do I need to become a Data Scientist?",
            "What salary can I expect as a Senior ML Engineer in Germany?",
            "I know Python, SQL and Excel — what's my gap to become a Data Scientist?",
            "Create a 12-month learning plan to become a Data Engineer",
            "What are the current trends in AI/ML for 2025?",
            "Compare Data Scientist vs ML Engineer career paths",
        ]
        for q in example_questions:
            if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
                st.session_state["prefill_query"] = q

        st.markdown("---")

        remaining = MAX_REQUESTS_PER_SESSION - st.session_state.request_count
        st.metric("Requests remaining", remaining)
        st.metric("Tokens used (session)", f"{st.session_state.total_tokens:,}")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = {}
            st.session_state.tool_calls = {}
            st.session_state.request_count = 0
            st.session_state.total_tokens = 0
            st.rerun()

        st.markdown("---")
        with st.expander("Privacy Notice"):
            st.markdown(
                "- Your inputs are sent to **OpenAI** for processing.\n"
                "- No conversation data is stored on this server.\n"
                "- Usage is subject to [OpenAI's data usage policies](https://openai.com/policies/api-data-usage-policies)."
            )
        st.markdown(
            "<small>Powered by GPT-4o-mini · LangChain · ChromaDB</small>",
            unsafe_allow_html=True,
        )

    # ── Main content ─────────────────────────────────────────────────────
    st.title("Career Advisor Bot")
    st.markdown(
        "Your AI-powered guide for data science and technology career questions. "
        "Ask about roles, skills, salaries, learning paths, and industry trends."
    )

    # Config checks
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("**OpenAI API key not found.** Add `OPENAI_API_KEY=your_key` to your `.env` file.")
        st.stop()
    if not SYSTEM_PROMPT or not SYSTEM_PROMPT.strip():
        st.error("**SYSTEM_PROMPT not configured.** Add `SYSTEM_PROMPT=...` to your `.env` file.")
        st.stop()

    # Load agent (cached)
    try:
        agent_executor, retriever = initialise_agent()
    except Exception as exc:
        st.error(f"**Initialisation failed:** {exc}")
        st.stop()

    # ── Render conversation history ───────────────────────────────────────
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_tool_calls(st.session_state.tool_calls.get(idx, []))
                render_sources(st.session_state.sources.get(idx, []))

    # ── Handle sidebar example button prefill ────────────────────────────
    prefill = st.session_state.pop("prefill_query", None)

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a career question...", key="chat_input")

    # Use prefilled query if set (from sidebar buttons)
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # Validate
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            st.warning(error_msg)
            st.stop()

        # Rate limiting
        if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
            st.error(
                f"You have reached the session limit of {MAX_REQUESTS_PER_SESSION} requests. "
                "Please refresh the page to start a new session."
            )
            st.stop()

        # Simple per-request throttle (min 1 second between requests)
        now = time.time()
        elapsed = now - st.session_state.last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})
        msg_idx = len(st.session_state.messages)  # index of the upcoming assistant message

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Step 1: retrieve context (for source attribution)
                    context, source_docs = retrieve_context(user_input, retriever)

                    # Step 2: run agent
                    chat_history = build_lc_history()[:-1]  # exclude last user msg (already in input)
                    result = run_agent(user_input, chat_history, context, agent_executor)

                    answer = result.get("output", "I could not generate a response.")
                    intermediate_steps = result.get("intermediate_steps", [])
                    tool_calls = parse_tool_calls(intermediate_steps)

                    # Store metadata
                    st.session_state.tool_calls[msg_idx] = tool_calls
                    st.session_state.sources[msg_idx] = source_docs
                    st.session_state.request_count += 1
                    st.session_state.last_request_time = time.time()
                    st.session_state.total_tokens += count_tokens(user_input + context + answer)

                    st.markdown(answer)
                    render_tool_calls(tool_calls)
                    render_sources(source_docs)

                except Exception as exc:
                    logger.error(f"Agent error: {exc}", exc_info=True)
                    answer = (
                        "I encountered an error while processing your request. "
                        "Please try again or rephrase your question."
                    )
                    st.error(answer)
                    tool_calls = []
                    source_docs = []

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
