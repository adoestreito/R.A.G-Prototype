"""Build Chroma index + starter tiles from a Hugging Face dataset or local MDN content."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parent / ".chroma"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_HF_DATASET_ID = "m-ric/huggingface_doc"


def source_to_doc_url(source: str) -> str:
    s = (source or "").strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return f"https://github.com/{s}"


def _title_from_filename(filename: str) -> str:
    base = re.sub(r"\.(mdx?|ipynb|md)$", "", filename, flags=re.I)
    base = base.replace("_", " ").replace("-", " ")
    parts = [p for p in base.split() if p]
    return " ".join(p.capitalize() for p in parts) if parts else filename


def _meta_from_source(source: str) -> tuple[str, str, str]:
    """title, repo_label (org/repo), doc_url."""
    doc_url = source_to_doc_url(source)
    parts = source.split("/")
    org, repo = "", ""
    filename = parts[-1] if parts else ""
    if len(parts) >= 4 and "blob" in parts:
        bi = parts.index("blob")
        org, repo = parts[0], parts[1]
        filename = parts[-1] if bi < len(parts) - 1 else filename

    fn_base = re.sub(r"\.(mdx?|ipynb|md)$", "", filename, flags=re.I)
    if fn_base.lower() == "index" and len(parts) >= 2:
        parent_slug = parts[-2]
        title = _title_from_filename(f"{parent_slug}.md")
    else:
        title = _title_from_filename(filename)

    repo_label = f"{org}/{repo}" if org and repo else (parts[0] if parts else "docs")
    return title, repo_label, doc_url


def _mdn_doc_area_from_source(source: str) -> str:
    m = re.search(r"/files/en-us/(.+?)/[^/]+\.md$", source, re.I)
    if m:
        return m.group(1)
    return "mdn/content"


def _query_for_tile(title: str, repo_label: str, brand: str) -> str:
    if brand == "mdn":
        return (
            f"What does MDN document about {title}? "
            f'(From “{repo_label}” in mdn/content.)'
        )
    slug = repo_label.split("/")[-1].replace("-", " ")
    return (
        f"What do the Hugging Face docs explain about {title}? "
        f'(From the "{slug}" documentation.)'
    )


def _slug_from_source_url(source: str) -> str:
    """Folder slug for index.* pages, else file stem; lowercase for matching."""
    parts = [p for p in source.split("/") if p]
    if not parts:
        return ""
    filename = parts[-1]
    stem = re.sub(r"\.(mdx?|ipynb|md)$", "", filename, flags=re.I)
    if stem.lower() == "index" and len(parts) >= 2:
        return parts[-2].lower()
    return stem.lower()


def _sources_look_like_mdn_http_headers(sources: set[str]) -> bool:
    return any("/reference/headers/" in s or s.endswith("/reference/headers") for s in sources)


# Default “important first” order when ingesting MDN …/web/http/reference/headers (no SUGGESTION_PRIORITY set).
_DEFAULT_MDN_HEADER_SLUGS: tuple[str, ...] = (
    "accept",
    "accept-language",
    "accept-encoding",
    "authorization",
    "cache-control",
    "content-type",
    "content-length",
    "cookie",
    "set-cookie",
    "origin",
    "referer",
    "user-agent",
    "host",
    "if-modified-since",
    "if-none-match",
    "etag",
    "expires",
    "access-control-allow-origin",
    "access-control-request-method",
    "vary",
    "www-authenticate",
)


def _normalize_slug_token(token: str) -> str:
    return token.strip().lower().replace("_", "-")


def _pick_tile_sources(
    unique_sorted: list[str], count: int, strategy: str, priority_slugs: list[str]
) -> list[str]:
    if count <= 0 or not unique_sorted:
        return []
    if strategy == "random":
        k = min(count, len(unique_sorted))
        seed = os.getenv("SUGGESTION_RANDOM_SEED")
        rng = random.Random(int(seed)) if seed else random.Random()
        pool = list(unique_sorted)
        picked = rng.sample(pool, k=k) if k < len(pool) else pool
        rng.shuffle(picked)
        return picked[:count]

    if strategy == "priority" and priority_slugs:
        slug_to_sources: dict[str, list[str]] = {}
        for s in unique_sorted:
            slug = _normalize_slug_token(_slug_from_source_url(s))
            slug_to_sources.setdefault(slug, []).append(s)

        pool_slugs = set(slug_to_sources.keys())
        warned: set[str] = set()
        for raw in priority_slugs:
            t = _normalize_slug_token(raw)
            if t and t not in pool_slugs and t not in warned:
                warned.add(t)
                print(
                    f"ingest: SUGGESTION_PRIORITY slug {t!r} has no indexed page — "
                    "skipped (missing path, MDN_MAX_FILES sampling, or page body < 80 chars after frontmatter)."
                )

        chosen: list[str] = []
        used: set[str] = set()
        for token in priority_slugs:
            t = _normalize_slug_token(token)
            if not t:
                continue
            if t in slug_to_sources:
                for cand in slug_to_sources[t]:
                    if cand not in used:
                        chosen.append(cand)
                        used.add(cand)
                        break
                continue
            for s in unique_sorted:
                if s in used:
                    continue
                slug = _normalize_slug_token(_slug_from_source_url(s))
                if t == slug or slug.startswith(t + "-") or slug.endswith("-" + t):
                    chosen.append(s)
                    used.add(s)
                    break

        for s in unique_sorted:
            if len(chosen) >= count:
                break
            if s not in used:
                chosen.append(s)
                used.add(s)
        return chosen[:count]

    # alphabetical (default): stable, skim-friendly order
    return unique_sorted[:count]


def build_suggestions_from_sources(
    sources: set[str], count: int, brand: str
) -> list[dict[str, str]]:
    unique = sorted({s.strip() for s in sources if s.strip()})
    if not unique:
        return []

    strategy = os.getenv("SUGGESTION_STRATEGY", "alphabetical").strip().lower()
    if strategy not in ("alphabetical", "priority", "random"):
        strategy = "alphabetical"

    raw_priority = os.getenv("SUGGESTION_PRIORITY", "")
    priority_slugs = [
        _normalize_slug_token(p) for p in raw_priority.split(",") if _normalize_slug_token(p)
    ]

    if strategy == "priority" and not priority_slugs:
        if brand == "mdn" and _sources_look_like_mdn_http_headers(sources):
            priority_slugs = list(_DEFAULT_MDN_HEADER_SLUGS)
            print(
                "SUGGESTION_STRATEGY=priority with empty SUGGESTION_PRIORITY: "
                "using built-in important HTTP header order for reference/headers."
            )
        else:
            strategy = "alphabetical"

    picked = _pick_tile_sources(unique, count, strategy, priority_slugs)
    out: list[dict[str, str]] = []
    for src in picked:
        title, repo_label, doc_url = _meta_from_source(src)
        subtitle = _mdn_doc_area_from_source(src) if brand == "mdn" else repo_label
        out.append(
            {
                "title": title,
                "subtitle": subtitle,
                "query": _query_for_tile(title, subtitle, brand),
                "source": src,
                "doc_url": doc_url,
            }
        )
    return out


def _strip_mdn_frontmatter(text: str) -> str:
    t = text.lstrip("\ufeff")
    stripped = re.sub(
        r"^---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n",
        "",
        t,
        count=1,
        flags=re.DOTALL,
    )
    return stripped if stripped != t else t


def _collect_mdn_markdown_paths(
    content_root: Path, path_prefix: str | None = None
) -> list[Path]:
    root = content_root.resolve()
    files_base = root / "files"
    if not files_base.is_dir():
        raise SystemExit(
            f"Expected a “files” directory under MDN_CONTENT_ROOT.\n"
            f"Clone https://github.com/mdn/content and set MDN_CONTENT_ROOT to that folder."
        )

    if path_prefix:
        rel = path_prefix.strip().strip("/").replace("\\", "/")
        base = (root / rel).resolve()
        try:
            base.relative_to(root)
        except ValueError:
            raise SystemExit(
                f"MDN_PATH_PREFIX must stay inside MDN_CONTENT_ROOT (got {path_prefix!r})."
            ) from None
        if not base.is_dir():
            raise SystemExit(
                f"MDN_PATH_PREFIX is not a directory:\n  {base}\n"
                f"Use a path relative to the repo root, e.g. files/en-us/web/http/reference/headers\n"
                f"(see https://github.com/mdn/content/tree/main/files/en-us/web/http/reference/headers)"
            )
        return sorted(p for p in base.rglob("*.md") if p.is_file())

    en = files_base / "en-us"
    if en.is_dir():
        locale_dirs = [en]
    else:
        locale_dirs = [
            d
            for d in files_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    paths: list[Path] = []
    for loc in locale_dirs:
        paths.extend(p for p in loc.rglob("*.md") if p.is_file())
    return paths


def _mdn_github_source(content_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(content_root).as_posix()
    return f"mdn/content/blob/main/{rel}"


def _run_ingest_mdn(
    text_splitter: RecursiveCharacterTextSplitter, n_tiles: int
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    raw = os.getenv("MDN_CONTENT_ROOT", "").strip()
    if not raw:
        raise SystemExit(
            "Set MDN_CONTENT_ROOT to your clone of https://github.com/mdn/content "
            "(repository root, the folder that contains “files/”)."
        )
    content_root = Path(raw).expanduser().resolve()
    if not content_root.is_dir():
        raise SystemExit(f"MDN_CONTENT_ROOT is not a directory: {content_root}")

    prefix_env = os.getenv("MDN_PATH_PREFIX", "").strip()
    path_prefix = prefix_env or None
    if path_prefix:
        print(f"MDN ingest subtree: {path_prefix}")

    paths = _collect_mdn_markdown_paths(content_root, path_prefix)
    if not paths:
        hint = f" under {path_prefix}" if path_prefix else ""
        raise SystemExit(f"No .md files found{hint}. Check MDN_CONTENT_ROOT and MDN_PATH_PREFIX.")

    max_files = int(os.getenv("MDN_MAX_FILES", "2500"))
    total_pages = len(paths)
    if max_files > 0 and total_pages > max_files:
        seed = os.getenv("MDN_RANDOM_SEED")
        rng = random.Random(int(seed)) if seed else random.Random()
        paths = rng.sample(paths, max_files)
        print(f"Sampling {max_files} of {total_pages} MDN pages (MDN_MAX_FILES).")

    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    page_sources: set[str] = set()

    for file_path in paths:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        body = _strip_mdn_frontmatter(text).strip()
        if len(body) < 80:
            continue
        src = _mdn_github_source(content_root, file_path)
        doc_url = source_to_doc_url(src)
        page_sources.add(src)
        for ch in text_splitter.split_text(body):
            documents.append(ch)
            metadatas.append({"source": src, "doc_url": doc_url})

    if not documents:
        raise SystemExit("No MDN content produced chunks — check MDN_CONTENT_ROOT.")

    suggestions = build_suggestions_from_sources(page_sources, n_tiles, "mdn")
    return documents, metadatas, suggestions


def _run_ingest_hf(
    text_splitter: RecursiveCharacterTextSplitter, n_tiles: int
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    dataset_id = os.getenv("HF_DATASET_ID", DEFAULT_HF_DATASET_ID)
    print(f"Loading Hugging Face dataset {dataset_id} …")
    ds = load_dataset(dataset_id, split="train")

    page_sources: set[str] = {
        (row.get("source") or "").strip()
        for row in ds
        if (row.get("source") or "").strip()
    }
    suggestions = build_suggestions_from_sources(page_sources, n_tiles, "hf")

    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        src = row.get("source") or ""
        doc_url = source_to_doc_url(src)
        for ch in text_splitter.split_text(text):
            documents.append(ch)
            metadatas.append({"source": src, "doc_url": doc_url})

    return documents, metadatas, suggestions


def main() -> None:
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    mode = os.getenv("INGEST_SOURCE", "hf").strip().lower()
    n_tiles = int(os.getenv("SUGGESTION_TILES", "12"))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if mode == "mdn":
        print("Ingest source: MDN (local clone of mdn/content)")
        documents, metadatas, suggestions = _run_ingest_mdn(text_splitter, n_tiles)
    else:
        if mode != "hf":
            print(f"Unknown INGEST_SOURCE={mode!r}, using hf.")
        documents, metadatas, suggestions = _run_ingest_hf(text_splitter, n_tiles)

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    sugg_path = STATIC_DIR / "suggestions.json"
    sugg_path.write_text(json.dumps(suggestions, indent=2), encoding="utf-8")
    print(f"Wrote {len(suggestions)} starter tiles → {sugg_path}")

    print(f"Embedding {len(documents)} chunks with {embedding_model} …")
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_texts(
        texts=documents,
        embedding=emb,
        metadatas=metadatas,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Done. Index at {CHROMA_DIR}")
    if mode == "mdn":
        print("Links point at GitHub (markdown source), same style as the HF doc index.")


if __name__ == "__main__":
    main()
