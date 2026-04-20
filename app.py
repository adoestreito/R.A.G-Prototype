"""FastAPI web UI + LangChain RAG over Hugging Face docs (Ollama + Chroma)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parent / ".chroma"
STATIC_DIR = Path(__file__).resolve().parent / "static"
USER_NOTES_COLLECTION = "user_notes"

PERSONALITY_HINTS: dict[str, str] = {
    "neutral": "Clear and balanced; neither stiff nor silly.",
    "funny": "Witty and playful where it helps understanding; jokes never replace facts.",
    "scientific": "Precise and evidence-oriented; state uncertainty when context is thin.",
    "formal": "Professional, concise, and structured; avoid slang and filler.",
    "casual": "Relaxed and conversational, like a sharp colleague explaining over coffee.",
}

TLDR_SYSTEM_BASE = """You are a technical writer helping developers learn from documentation snippets \
(HTML, CSS, JavaScript, Web APIs, ML tooling, etc., depending on the corpus). \
The user’s tone preference is described below — follow it, but never sacrifice accuracy.

Tone: {personality_hint}

You ONLY respond with documentation-grounded takeaways from the provided context snippets \
(and any labeled user notes, which the user added intentionally — treat them as trusted local context). \
Do not invent facts. If the snippets are insufficient, say so in the summary lines.

Output EXACTLY one block in this format and nothing else (no other text before or after):

:::SUMMARY
Line 1: Direct answer in one short sentence.
Line 2: Most useful detail, step, or caveat in one short sentence.
Line 3: Optional quip or “heads-up” in one short sentence (or a honest “not enough in the snippets” note).

:::

Rules for the three lines: plain text only inside the block (no markdown, bullets, or links). \
The user will get official doc URLs separately; you do not need to repeat them."""

CHAT_SYSTEM = """You are a technical assistant answering from retrieved documentation snippets \
and any labeled user notes (trusted local context the user saved on topic tiles).

Tone: {personality_hint}

Answer thoroughly and clearly. You may use markdown (headings, bullet lists, short code fences) when it helps. \
Ground claims in the provided context. If something is not supported by the context, say so explicitly.

Do not fabricate API behavior, browser support, or quotes. User notes are supplementary — still prefer official \
docs for normative behavior when both apply.

End with a short **Sources** section listing the documentation URLs from the snippets (not raw file paths unless \
that is all you have)."""


_SUMMARY_START = ":::SUMMARY"
_SUMMARY_END = ":::"


def source_to_doc_url(source: str) -> str:
    s = (source or "").strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return f"https://github.com/{s}"


def _split_summary_from_answer(raw: str) -> tuple[list[str], str]:
    """Extract three-line :::SUMMARY ... ::: block; return (lines, body_without_block)."""
    text = (raw or "").strip()
    if _SUMMARY_START not in text:
        return _fallback_summary_lines(text), text
    try:
        i0 = text.index(_SUMMARY_START) + len(_SUMMARY_START)
        i1 = text.index(_SUMMARY_END, i0)
        block = text[i0:i1].strip()
        after = text[i1 + len(_SUMMARY_END) :].strip()
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        lines = lines[:3]
        if not lines:
            return _fallback_summary_lines(after or text), after or text
        while len(lines) < 3:
            lines.append("")
        return lines[:3], after
    except ValueError:
        pass
    return _fallback_summary_lines(text), text


def _unique_doc_links_ordered(docs: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        u = d.metadata.get("doc_url") or source_to_doc_url(
            d.metadata.get("source", "")
        )
        u = (u or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _fallback_summary_lines(answer: str) -> list[str]:
    """Up to 3 short lines from first sentences if the model skipped the summary block."""
    if not answer.strip():
        return ["", "", ""]
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 160:
            p = p[:157].rstrip() + "…"
        out.append(p)
        if len(out) == 3:
            break
    while len(out) < 3:
        out.append("")
    return out[:3]


def format_context(docs: list) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, 1):
        url = d.metadata.get("doc_url") or source_to_doc_url(
            d.metadata.get("source", "")
        )
        parts.append(
            f"--- Snippet {i} ---\nDocumentation link: {url}\n{d.page_content}"
        )
    return "\n\n".join(parts)


def format_user_notes_context(docs: list) -> str:
    if not docs:
        return ""
    parts: list[str] = []
    for i, d in enumerate(docs, 1):
        title = (d.metadata.get("title") or "My note").strip()
        src = (d.metadata.get("source") or "").strip()
        parts.append(
            f"--- User note {i} (topic: {title}) ---\n"
            f"Tile source key: {src}\n{d.page_content}"
        )
    return "\n\n".join(parts)


def _note_stable_id(source_key: str) -> str:
    h = hashlib.sha256(source_key.strip().encode("utf-8")).hexdigest()[:32]
    return f"note:{h}"


@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=embedding_model)


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        temperature=float(os.getenv("OLLAMA_CHAT_TEMPERATURE", "0.35")),
    )


def _doc_store() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings(),
    )


def _notes_store() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings(),
        collection_name=USER_NOTES_COLLECTION,
    )


def _doc_retriever():
    return _doc_store().as_retriever(
        search_kwargs={"k": int(os.getenv("RETRIEVAL_K", "5"))}
    )


def _notes_retriever():
    return _notes_store().as_retriever(
        search_kwargs={"k": int(os.getenv("NOTES_RETRIEVAL_K", "4"))}
    )


def _merge_context(question: str) -> tuple[str, list]:
    """Return (formatted context, all docs for link extraction)."""
    doc_r = _doc_retriever()
    note_r = _notes_retriever()
    doc_docs = doc_r.invoke(question)
    note_docs = note_r.invoke(question)
    blocks: list[str] = []
    nd = format_user_notes_context(note_docs)
    if nd:
        blocks.append(nd)
    dd = format_context(doc_docs)
    if dd:
        blocks.append(dd)
    merged_docs = list(note_docs) + list(doc_docs)
    return "\n\n".join(blocks), merged_docs


app = FastAPI(title="Docs RAG Chat")

Personality = Literal["neutral", "funny", "scientific", "formal", "casual"]
ResponseMode = Literal["tldr", "chat"]


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    response_mode: ResponseMode = "tldr"
    personality: Personality = "neutral"


class ChatOut(BaseModel):
    summary_lines: list[str]
    links: list[str]
    source_count: int
    answer: Optional[str] = None


class NoteIn(BaseModel):
    source: str = Field(..., min_length=1, max_length=2000)
    title: str = Field("", max_length=500)
    text: str = Field("", max_length=8000)
    doc_url: str = Field("", max_length=2000)


class NoteOut(BaseModel):
    source: str
    title: str
    text: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/suggestions")
def suggestions() -> list:
    path = STATIC_DIR / "suggestions.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


@app.get("/api/notes", response_model=list[NoteOut])
def list_notes() -> list[NoteOut]:
    if not CHROMA_DIR.exists():
        return []
    try:
        store = _notes_store()
        bag = store.get(include=["documents", "metadatas"])
    except Exception:
        return []
    docs = bag.get("documents") or []
    metas = bag.get("metadatas") or []
    out: list[NoteOut] = []
    for i, text in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        if not isinstance(meta, dict):
            meta = {}
        src = (meta.get("source") or "").strip()
        if not src:
            continue
        out.append(
            NoteOut(
                source=src,
                title=(meta.get("title") or "").strip(),
                text=text or "",
            )
        )
    return out


@app.post("/api/notes", response_model=NoteOut)
def save_note(body: NoteIn) -> NoteOut:
    if not CHROMA_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Vector index missing at {CHROMA_DIR}. Run: python ingest.py",
        ) from None
    source = body.source.strip()
    title = body.title.strip()
    text = body.text.strip()
    doc_url = (body.doc_url or "").strip()
    nid = _note_stable_id(source)
    store = _notes_store()
    try:
        store.delete(ids=[nid])
    except Exception:
        pass
    if not text:
        return NoteOut(source=source, title=title, text="")
    page = f"Topic: {title or source}\nUser notes:\n{text}"
    meta = {
        "source": source,
        "title": title or "Note",
        "kind": "user_note",
        "doc_url": doc_url,
    }
    store.add_texts(texts=[page], metadatas=[meta], ids=[nid])
    return NoteOut(source=source, title=title, text=text)


@app.post("/api/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    if not CHROMA_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Vector index missing at {CHROMA_DIR}. Run: python ingest.py",
        ) from None
    question = body.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty message")
    ctx, merged_docs = _merge_context(question)
    if not ctx.strip():
        raise HTTPException(
            status_code=503,
            detail="No documents retrieved; re-run ingest or check the index.",
        )
    personality_hint = PERSONALITY_HINTS.get(
        body.personality, PERSONALITY_HINTS["neutral"]
    )
    llm = _llm()
    if body.response_mode == "chat":
        system = CHAT_SYSTEM.format(personality_hint=personality_hint)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Context from retrieved documentation and user notes:\n\n{context}\n\nUser question: {question}",
                ),
            ]
        )
    else:
        system = TLDR_SYSTEM_BASE.format(personality_hint=personality_hint)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Context from retrieved documentation and user notes:\n\n{context}\n\nUser question: {question}",
                ),
            ]
        )
    msg = prompt.format_messages(context=ctx, question=question)
    try:
        ai = llm.invoke(msg)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error: {e!s}. Is the model pulled and Ollama running?",
        ) from e
    content = getattr(ai, "content", None) or str(ai)
    links = _unique_doc_links_ordered(merged_docs)
    if body.response_mode == "tldr":
        summary_lines, _rest = _split_summary_from_answer(content)
        return ChatOut(
            summary_lines=summary_lines,
            links=links,
            source_count=len(links),
            answer=None,
        )
    return ChatOut(
        summary_lines=["", "", ""],
        links=links,
        source_count=len(links),
        answer=content.strip(),
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
