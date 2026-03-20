"""
Career Advisor Bot — RAG Pipeline
Implements:
  - Document loading from knowledge_base/
  - MultiQueryRetriever for query translation (advanced RAG)
"""
import logging
import os
from pathlib import Path
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Suppress verbose HTTP logs from chromadb / openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
COLLECTION_NAME = "career_advisor_kb"

def _load_documents() -> list[Document]:
    """Load all .txt files from the knowledge_base directory."""
    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")

    loader = DirectoryLoader(
        str(KB_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        silent_errors=True,
    )
    docs = loader.load()

    if not docs:
        raise ValueError("No documents found in knowledge_base/. "
                         "Ensure .txt files are present.")

    logger.info(f"Loaded {len(docs)} document(s) from {KB_DIR}")
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Separators are chosen to respect the markdown-like structure of
    the knowledge base files (headings, paragraphs, sentences).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n---\n\n", "\n---\n", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks "
                f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def _build_vectorstore(chunks: list[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    """Create a new ChromaDB vectorstore from document chunks."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info(f"Built vectorstore with {len(chunks)} chunks → persisted to {CHROMA_DIR}")
    return vectorstore


def _load_vectorstore(embeddings: OpenAIEmbeddings) -> Chroma:
    """Load an existing ChromaDB vectorstore from disk."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def _vectorstore_is_current() -> bool:
    """
    Check whether the persisted vectorstore exists and is not stale.
    We consider it stale if any knowledge base file is newer than the
    chroma_db directory.
    """
    if not CHROMA_DIR.exists():
        return False

    chroma_mtime = CHROMA_DIR.stat().st_mtime
    kb_files = list(KB_DIR.glob("*.txt"))
    if not kb_files:
        return False

    latest_kb_mtime = max(f.stat().st_mtime for f in kb_files)
    return chroma_mtime >= latest_kb_mtime


def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Return a ChromaDB vectorstore — either loaded from disk (if up to date)
    or freshly built from the knowledge base.

    Args:
        force_rebuild: If True, always rebuild from scratch.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not force_rebuild and _vectorstore_is_current():
        logger.info("Loading existing vectorstore from disk.")
        return _load_vectorstore(embeddings)

    logger.info("Building vectorstore from knowledge base documents.")
    docs = _load_documents()
    chunks = _split_documents(docs)
    return _build_vectorstore(chunks, embeddings)


def get_retriever(vectorstore: Chroma, llm: ChatOpenAI) -> MultiQueryRetriever:
    """
    Build a MultiQueryRetriever on top of the vectorstore.

    MultiQueryRetriever uses the LLM to generate *multiple* alternative
    phrasings of the user's query, runs each through the vector store,
    and merges the results — dramatically improving recall compared to
    a single-query retriever.

    Args:
        vectorstore: Initialised ChromaDB vectorstore.
        llm:         LLM used to generate query variants.
    """
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,   # also run the original query
    )
    return retriever


def format_retrieved_docs(docs: list[Document]) -> str:
    """Format a list of retrieved documents into a single context string."""
    if not docs:
        return "No relevant information found in the knowledge base."

    formatted = []
    seen_contents = set()

    for i, doc in enumerate(docs, 1):
        # De-duplicate chunks that may appear across query variants
        content_preview = doc.page_content[:100]
        if content_preview in seen_contents:
            continue
        seen_contents.add(content_preview)

        source = doc.metadata.get("source", "knowledge base")
        source_name = Path(source).stem.replace("_", " ").title()
        formatted.append(f"[Source {i}: {source_name}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def build_retriever_tool(retriever: MultiQueryRetriever) -> Tool:
    """
    Wrap the MultiQueryRetriever as a LangChain Tool so that the agent
    can decide when to query the knowledge base.
    """
    def _retrieve_and_format(query: str) -> str:
        docs = retriever.invoke(query)
        return format_retrieved_docs(docs)

    return Tool(
        name="career_knowledge_search",
        func=_retrieve_and_format,
        description=(
            "Search the career advisor knowledge base for information about "
            "data science and technology roles, required skills, career paths, "
            "salary benchmarks, and industry trends. "
            "Use this tool when the user asks general career questions, "
            "questions about specific roles, or needs background information "
            "that is not covered by the other specialist tools."
        ),
    )
