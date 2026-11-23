"""Indexing intelligence: turn a raw code snippet into validated, structured
metadata and a contextual embedding string. Everything here is defensive so a
single bad LLM response never corrupts the index.
"""
import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("xplorer.indexer")

# Rough char budgets (≈4 chars/token).
_MAX_SNIPPET_CHARS = 12000      # above this we chunk+summarize instead of one-shot
_CHUNK_CHARS = 6000
_CHUNK_OVERLAP = 400


class SnippetAnalysis(BaseModel):
    summary: str = Field(default="")
    description: str = Field(default="")
    purpose: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)


METADATA_SYSTEM_PROMPT = """You are an expert AI code analyst. Analyze a code snippet and return a structured JSON object.

Directives:
- Base everything only on the provided code and its file/import context.
- summary: one concise paragraph a developer can read.
- description: one dense sentence optimized for search (key nouns, verbs, technologies).
- purpose: a short high-level statement (e.g. 'User authentication endpoint').
- tags: relevant technical tags (e.g. 'api-endpoint', 'data-model', 'authentication', 'async', 'file-io', 'database-query', 'orm').
- key_entities: important function/class/variable names or concepts in the code.
Return ONLY the JSON object."""

SUMMARY_SYSTEM_PROMPT = """You are an expert code analyst. Provide a concise, one-paragraph summary of the provided code.
Focus on its primary purpose, inputs, and what it returns or its main side effect. Start directly, no preamble."""

COMBINE_SUMMARY_PROMPT = """You are an expert code analyst. You are given several partial summaries of different
chunks of a single file. Combine them into one concise, coherent paragraph describing the whole file's purpose and
responsibilities. Start directly, no preamble."""


def _extract_json(text):
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Grab the outermost { ... }.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _fallback_analysis(file_path, item_name, code, static_tags):
    return {
        "summary": "Automatic analysis was unavailable; this entry indexes the raw code snippet.",
        "description": f"Code from {file_path} containing {item_name}.",
        "purpose": "Unknown (analysis fallback).",
        "tags": sorted(set(static_tags or [])),
        "key_entities": [item_name] if item_name else [],
    }


def analyze_snippet(registry, file_path, item_name, code, model,
                    import_header="", class_ctx="", static_tags=None):
    """Produce validated SnippetAnalysis metadata with a repair loop, then a
    code-derived fallback. Returns a plain dict."""
    static_tags = static_tags or []

    # Very large snippets: summarize first so we don't overflow context.
    analysis_code = code
    pre_summary = ""
    if len(code) > _MAX_SNIPPET_CHARS:
        pre_summary = summarize_long_text(registry, code, model, label=f"{file_path}::{item_name}")

    context_bits = [f"File: `{file_path}`"]
    if class_ctx:
        context_bits.append(f"Enclosing scope: `{class_ctx}`")
    if import_header:
        context_bits.append(f"File imports:\n{import_header}")
    if pre_summary:
        context_bits.append(f"High-level summary of this (large) snippet:\n{pre_summary}")
    context = "\n".join(context_bits)

    shown_code = analysis_code if len(analysis_code) <= _MAX_SNIPPET_CHARS else analysis_code[:_MAX_SNIPPET_CHARS]
    user_prompt = f"{context}\n\nCode snippet:\n```\n{shown_code}\n```"

    messages = [
        {"role": "system", "content": METADATA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for attempt in range(3):
        raw = ""
        try:
            raw = registry.chat(messages, model=model, schema=SnippetAnalysis, think=False)
            data = _extract_json(raw)
            analysis = SnippetAnalysis(**data)
            result = analysis.model_dump()
            # Merge static tags so regex-detected actions always survive.
            result["tags"] = sorted(set((result.get("tags") or []) + list(static_tags)))
            if not result.get("key_entities") and item_name:
                result["key_entities"] = [item_name]
            return result
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
            last_err = e
            # Repair turn: show the model its mistake and ask again.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (f"That response could not be parsed/validated ({e}). "
                            "Return ONLY a valid JSON object with keys summary, description, "
                            "purpose, tags (array), key_entities (array)."),
            })
        except Exception as e:  # provider/transport error
            last_err = e
            break

    logger.warning(f"Snippet analysis fell back for {file_path}::{item_name}: {last_err}")
    return _fallback_analysis(file_path, item_name, code, static_tags)


def _chunk(text, size=_CHUNK_CHARS, overlap=_CHUNK_OVERLAP):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def summarize_long_text(registry, text, model, label=""):
    """Summarize text that may exceed the context window: summarize each chunk,
    then combine. Small text is summarized in a single call."""
    text = text or ""
    if len(text) <= _MAX_SNIPPET_CHARS:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this code{(' from ' + label) if label else ''}:\n\n```\n{text}\n```"},
        ]
        try:
            return registry.chat(messages, model=model, think=False)
        except Exception as e:
            logger.warning(f"Summarize failed for {label}: {e}")
            return "Summary unavailable."

    partials = []
    for i, chunk in enumerate(_chunk(text)):
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize chunk {i + 1} of {label or 'a file'}:\n\n```\n{chunk}\n```"},
        ]
        try:
            partials.append(registry.chat(messages, model=model, think=False))
        except Exception as e:
            logger.warning(f"Chunk summarize failed for {label} chunk {i}: {e}")

    if not partials:
        return "Summary unavailable."
    combine_messages = [
        {"role": "system", "content": COMBINE_SUMMARY_PROMPT},
        {"role": "user", "content": "Partial summaries:\n\n" + "\n\n".join(f"- {p}" for p in partials)},
    ]
    try:
        return registry.chat(combine_messages, model=model, think=False)
    except Exception as e:
        logger.warning(f"Combine summarize failed for {label}: {e}")
        return " ".join(partials)[:2000]


def build_embedding_text(em, file_path, description, code):
    """Contextual string used for embedding. Provider-specific prefixes
    (search_document:, Voyage input_type) are applied inside the registry, so
    they are intentionally not added here."""
    desc = (description or "").strip()
    header = file_path if not desc else f"{file_path}\n{desc}"
    text = f"{header}\n\n{code}"
    # Keep within the model's budget (~4 chars/token).
    budget = max(1000, em.max_tokens * 4)
    if len(text) > budget:
        text = text[:budget]
    return text
