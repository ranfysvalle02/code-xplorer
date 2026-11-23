"""Multi-step chat agent. Instead of one-shot prompting, a question flows through
plan -> retrieve -> grade -> expand -> (sufficiency loop) -> draft -> verify.
Each stage emits a trace event so the UI can show what the agent is doing, and
every LLM call degrades gracefully if parsing fails.
"""
import json
import logging
import re
from typing import List

from pydantic import BaseModel, Field

from . import db as dbmod

logger = logging.getLogger("xplorer.agent")


class Plan(BaseModel):
    rewritten_query: str = Field(default="")
    classification: str = Field(default="complex")  # "simple" | "complex"
    sub_queries: List[str] = Field(default_factory=list)


class Grading(BaseModel):
    relevant_indices: List[int] = Field(default_factory=list)


class Sufficiency(BaseModel):
    sufficient: bool = Field(default=True)
    missing: str = Field(default="")
    follow_up_query: str = Field(default="")


class Verification(BaseModel):
    grounded: bool = Field(default=True)
    issues: List[str] = Field(default_factory=list)


ANSWER_SYSTEM_PROMPT = """You are an expert AI code analyst answering questions about a codebase.

Core directives:
- Base your entire answer ONLY on the provided code snippets, summaries, and conversation history.
- Cite sources inline using the form `path::item` (e.g. `app.py::scan`). Cite every file/function you rely on.
- If the answer isn't in the provided context, say so plainly. Do not invent APIs or behavior.
- Be concise and concrete. Use the conversation history for follow-up context."""


def _safe_json_chat(registry, messages, schema, model, default):
    """Call the model expecting structured output; return a validated object or
    the provided default on any failure."""
    try:
        raw = registry.chat(messages, model=model, schema=schema, think=False)
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        if not text.startswith("{"):
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1:
                text = text[s:e + 1]
        return schema(**json.loads(text))
    except Exception as e:
        logger.warning(f"Structured call failed ({schema.__name__}): {e}")
        return default


def _hit_key(h):
    return f"{h.get('file_path')}::{h.get('item_name')}"


def _hit_block(h):
    summary = ""
    la = h.get("llm_analysis") or {}
    if la.get("summary"):
        summary = f"\n# summary: {la['summary']}"
    return f"--- {_hit_key(h)} ---{summary}\n{h.get('code', '')}\n"


class ChatAgent:
    def __init__(self, registry, coll, config):
        self.registry = registry
        self.coll = coll
        self.config = config

    # -------------------------------------------------------------- retrieval
    def _search(self, codebase_id, query, em, limit=8):
        try:
            qvec = self.registry.embed([query], em, kind="query")[0]
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []
        try:
            return dbmod.hybrid_search(self.coll, codebase_id, query, qvec, em, limit=limit)
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            return []

    # ------------------------------------------------------------------- plan
    def _plan(self, question, history):
        hist = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in (history or [])[-4:])
        messages = [
            {"role": "system", "content": (
                "You plan how to answer a question about a codebase. Return JSON with: "
                "rewritten_query (a standalone search query resolving pronouns from history), "
                "classification ('simple' for a direct lookup about one thing, 'complex' for "
                "multi-part/architectural questions), and sub_queries (1-4 focused search "
                "queries covering the question).")},
            {"role": "user", "content": f"Conversation so far:\n{hist}\n\nQuestion: {question}"},
        ]
        default = Plan(rewritten_query=question, classification="complex", sub_queries=[question])
        plan = _safe_json_chat(self.registry, messages, Plan, self.registry.active_fast_llm, default)
        if not plan.rewritten_query:
            plan.rewritten_query = question
        if not plan.sub_queries:
            plan.sub_queries = [plan.rewritten_query]
        plan.sub_queries = plan.sub_queries[:4]
        return plan

    # ------------------------------------------------------------------ grade
    def _grade(self, question, hits, keep=8):
        if not hits:
            return []
        reranked = self.registry.rerank(question, [h.get("code", "") for h in hits], top_k=keep)
        if reranked is not None:
            return [hits[i] for i, _ in reranked[:keep]]

        # LLM relevance grading fallback.
        catalog = "\n".join(
            f"[{i}] {_hit_key(h)} :: {(h.get('llm_analysis') or {}).get('description', '')[:160]}"
            for i, h in enumerate(hits))
        messages = [
            {"role": "system", "content": (
                "Given a question and numbered candidate code snippets, return JSON "
                "{\"relevant_indices\": [..]} listing only the indices useful for answering, "
                "most relevant first.")},
            {"role": "user", "content": f"Question: {question}\n\nCandidates:\n{catalog}"},
        ]
        default = Grading(relevant_indices=list(range(min(keep, len(hits)))))
        grading = _safe_json_chat(self.registry, messages, Grading, self.registry.active_fast_llm, default)
        picked = [hits[i] for i in grading.relevant_indices if 0 <= i < len(hits)]
        return (picked or hits)[:keep]

    # ----------------------------------------------------------------- expand
    def _expand(self, codebase_id, hits, em, existing_keys):
        entities = []
        for h in hits[:4]:
            entities.extend((h.get("llm_analysis") or {}).get("key_entities", []) or [])
        extra = []
        for ent in list(dict.fromkeys(entities))[:4]:
            for h in self._search(codebase_id, ent, em, limit=3):
                if _hit_key(h) not in existing_keys:
                    existing_keys.add(_hit_key(h))
                    extra.append(h)
        return extra

    # ------------------------------------------------------------ sufficiency
    def _sufficiency(self, question, hits):
        catalog = "\n".join(f"- {_hit_key(h)}: {(h.get('llm_analysis') or {}).get('description', '')[:160]}"
                            for h in hits)
        messages = [
            {"role": "system", "content": (
                "Decide if the retrieved snippets are enough to answer the question. Return JSON "
                "{\"sufficient\": bool, \"missing\": str, \"follow_up_query\": str}. If sufficient, "
                "follow_up_query may be empty.")},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved:\n{catalog}"},
        ]
        return _safe_json_chat(self.registry, messages, Sufficiency,
                               self.registry.active_fast_llm, Sufficiency(sufficient=True))

    # ------------------------------------------------------------------ pack
    def _pack_context(self, preselected, hits):
        budget = max(6000, self.config.num_ctx_cap * 3)
        parts = []
        used = 0
        paths = set()

        for block in preselected or []:
            text = f"--- SELECTED: {block.get('source', 'context')} ---\n{block.get('content', '')}\n"
            if used + len(text) > budget:
                continue
            parts.append(text)
            used += len(text)
            if block.get("path"):
                paths.add(block["path"])

        for h in hits:
            block = _hit_block(h)
            if used + len(block) > budget:
                continue
            parts.append(block)
            used += len(block)
            if h.get("file_path"):
                paths.add(h["file_path"])

        return "".join(parts), paths

    # ----------------------------------------------------------------- verify
    def _verify(self, question, answer, context, context_paths):
        # Deterministic citation check: any `path::item` cited must be present.
        cited = set(re.findall(r"`([^`\n]+?)::[^`\n]+`", answer))
        cited |= set(re.findall(r"`([\w./\-]+\.\w+)`", answer))
        missing = [c for c in cited if c not in context_paths and c not in context]
        citation_ok = len(missing) == 0

        # Grounding check via fast model.
        messages = [
            {"role": "system", "content": (
                "Check whether an answer is fully supported by the provided context. Return JSON "
                "{\"grounded\": bool, \"issues\": [..]} where issues are claims not supported by "
                "the context.")},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context[:12000]}\n\nAnswer:\n{answer}"},
        ]
        verification = _safe_json_chat(self.registry, messages, Verification,
                                       self.registry.active_fast_llm, Verification(grounded=True))
        if missing:
            verification.issues = list(verification.issues) + [f"Cited source not in context: {m}" for m in missing]
        verification.grounded = verification.grounded and citation_ok
        return verification

    # -------------------------------------------------------------------- run
    def run(self, codebase_id, question, history=None, preselected=None):
        """Generator yielding trace events, then a final answer event."""
        history = history or []
        preselected = preselected or []
        em = self.registry.active_embedding()
        max_iter = max(1, self.config.agent_max_iterations)

        # 1. Plan
        plan = self._plan(question, history)
        yield {"type": "step", "stage": "plan", "detail": {
            "rewritten": plan.rewritten_query,
            "classification": plan.classification,
            "sub_queries": plan.sub_queries,
        }}
        simple = plan.classification == "simple"

        # 2. Retrieve (+ optional expand / sufficiency loop)
        hits = []
        keys = set()
        queries = list(dict.fromkeys([plan.rewritten_query] + plan.sub_queries))
        iterations = 0
        while True:
            iterations += 1
            new_count = 0
            for q in queries:
                for h in self._search(codebase_id, q, em, limit=8):
                    k = _hit_key(h)
                    if k not in keys:
                        keys.add(k)
                        hits.append(h)
                        new_count += 1
            yield {"type": "step", "stage": "retrieve", "detail": {
                "queries": queries, "new_hits": new_count, "total_hits": len(hits)}}

            # 3. Grade
            graded = self._grade(plan.rewritten_query, hits, keep=10 if not simple else 6)
            yield {"type": "step", "stage": "grade", "detail": {
                "kept": [_hit_key(h) for h in graded]}}
            hits = graded

            if simple:
                break

            # 4. Expand
            extra = self._expand(codebase_id, hits, em, keys)
            if extra:
                hits.extend(extra)
                yield {"type": "step", "stage": "expand", "detail": {
                    "added": [_hit_key(h) for h in extra]}}

            # 5. Sufficiency
            if iterations >= max_iter:
                break
            suff = self._sufficiency(plan.rewritten_query, hits)
            yield {"type": "step", "stage": "sufficiency", "detail": {
                "sufficient": suff.sufficient, "missing": suff.missing}}
            if suff.sufficient or not suff.follow_up_query:
                break
            queries = [suff.follow_up_query]

        # 6. Draft
        context, context_paths = self._pack_context(preselected, hits)
        if not context.strip():
            yield {"type": "answer",
                   "answer": "I couldn't find anything relevant in the indexed codebase to answer that. "
                             "Try scanning the repository, or select specific files/snippets as context.",
                   "trace": {"hits": []}}
            return

        draft_messages = [
            {"role": "system", "content": f"{ANSWER_SYSTEM_PROMPT}\n\n## CODEBASE CONTEXT\n\n{context}"},
        ]
        draft_messages.extend(history)
        draft_messages.append({"role": "user", "content": question})
        try:
            answer = self.registry.chat(draft_messages, model=self.registry.active_llm, think=False)
        except Exception as e:
            yield {"type": "answer", "answer": f"[Error generating answer: {e}]", "trace": {}}
            return
        yield {"type": "step", "stage": "draft", "detail": {"chars": len(answer)}}

        # 7. Verify (+ one revision)
        if not simple:
            verification = self._verify(question, answer, context, context_paths)
            yield {"type": "step", "stage": "verify", "detail": {
                "grounded": verification.grounded, "issues": verification.issues[:5]}}
            if not verification.grounded and verification.issues:
                revise_messages = [
                    {"role": "system", "content": f"{ANSWER_SYSTEM_PROMPT}\n\n## CODEBASE CONTEXT\n\n{context}"},
                ]
                revise_messages.extend(history)
                revise_messages.append({"role": "user", "content": question})
                revise_messages.append({"role": "assistant", "content": answer})
                revise_messages.append({"role": "user", "content": (
                    "Revise your answer to fix these grounding issues, using ONLY the context above. "
                    "Remove or correct unsupported claims:\n- " + "\n- ".join(verification.issues[:6]))})
                try:
                    answer = self.registry.chat(revise_messages, model=self.registry.active_llm, think=False)
                    yield {"type": "step", "stage": "revise", "detail": {"chars": len(answer)}}
                except Exception as e:
                    logger.warning(f"Revision failed: {e}")

        yield {"type": "answer", "answer": answer, "trace": {
            "hits": [_hit_key(h) for h in hits],
            "classification": plan.classification,
        }}

    def answer(self, codebase_id, question, history=None, preselected=None):
        """Non-streaming convenience wrapper: returns (answer, trace)."""
        final = {"answer": "", "trace": {}}
        steps = []
        for event in self.run(codebase_id, question, history, preselected):
            if event.get("type") == "answer":
                final = event
            else:
                steps.append(event)
        final.setdefault("trace", {})
        final["trace"]["steps"] = steps
        return final["answer"], final["trace"]
