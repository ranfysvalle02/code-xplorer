"""Auditable analysis pipelines.

A pipeline is a JSON-defined list of small, inspectable steps. Each step declares
its type, the model it uses (user-selectable), its instructions, and which prior
steps feed it. Deterministic steps (`git`, `search`, `transform`) produce the
*objective artifacts* -- real commits, diffs, and retrieved code -- that LLM
steps (`extract`, `report`) must build on. Every step records its model, prompt,
inputs, raw output, parsed output, timing, and status, so the whole run is an
editable, re-runnable audit trail instead of a black box.

Design echoes the "box the model in" philosophy: extract don't summarize, anchor
on objective artifacts (here, git + grounded code search), move determinism
upstream, and generate the narrative last on a short leash.
"""
import datetime as dt
import json
import logging
import re
import subprocess
import time
import uuid

from pydantic import BaseModel, Field

from . import db as dbmod

logger = logging.getLogger("xplorer.pipeline")

STEP_TYPES = {"git", "search", "extract", "transform", "report"}

_REV_RE = re.compile(r'^[\w./~^@-]+$')
_PATCH_CAP = 12000
_INPUT_CAP = 16000


# --------------------------------------------------------------------- schema
class StepSpec(BaseModel):
    id: str
    title: str = ""
    type: str = "extract"
    model: str | None = None
    instructions: str = ""
    inputs: list[str] = Field(default_factory=list)
    output: str = "json"          # "json" | "text"
    params: dict = Field(default_factory=dict)


class PipelineSpec(BaseModel):
    name: str = "Untitled pipeline"
    description: str = ""
    steps: list[StepSpec] = Field(default_factory=list)


def default_model_for(step_type, registry):
    if step_type == "report":
        return registry.active_llm
    if step_type == "extract":
        return registry.active_fast_llm
    return None  # git / search / transform are deterministic


# ----------------------------------------------------------------------- git
def git_available(repo_path):
    if not repo_path:
        return False
    try:
        out = subprocess.run(["git", "-C", repo_path, "rev-parse", "--is-inside-work-tree"],
                             capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and out.stdout.strip() == "true"
    except Exception:
        return False


def _git(repo_path, args, timeout=30):
    return subprocess.run(["git", "-C", repo_path] + args,
                          capture_output=True, text=True, timeout=timeout)


def run_git(repo_path, op, params):
    """Run a read-only git operation and return structured JSON + a text rendering
    for downstream LLM steps. These are the objective artifacts."""
    op = (op or "log").lower()
    params = params or {}

    if op == "log":
        limit = min(int(params.get("limit", 20) or 20), 200)
        res = _git(repo_path, ["log", f"-n{limit}", "--date=short",
                               "--pretty=format:%H%x1f%an%x1f%ad%x1f%s"])
        commits = []
        for line in res.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append({"sha": parts[0][:10], "author": parts[1],
                                "date": parts[2], "subject": parts[3]})
        text = "\n".join(f"{c['date']} {c['sha']} {c['author']}: {c['subject']}" for c in commits)
        return {"op": "log", "count": len(commits), "commits": commits}, text

    if op == "diff":
        last = params.get("last")
        base = params.get("base")
        head = params.get("head", "HEAD")
        if last:
            rev = f"HEAD~{int(last)}..HEAD"
        elif base:
            rev = f"{base}..{head}"
        else:
            rev = "HEAD~1..HEAD"
        for token in re.split(r'\.\.+', rev):
            if token and not _REV_RE.match(token):
                raise ValueError(f"Unsafe git revision: {token}")
        stat = _git(repo_path, ["diff", "--stat", rev]).stdout.strip()
        patch = _git(repo_path, ["diff", rev]).stdout
        files = []
        for line in _git(repo_path, ["diff", "--numstat", rev]).stdout.splitlines():
            cols = line.split("\t")
            if len(cols) == 3:
                files.append({"added": cols[0], "removed": cols[1], "file": cols[2]})
        truncated = len(patch) > _PATCH_CAP
        patch_text = patch[:_PATCH_CAP]
        structured = {"op": "diff", "range": rev, "files": files,
                      "stat": stat, "patch_truncated": truncated}
        text = f"Diff {rev}\n{stat}\n\n{patch_text}"
        return structured, text

    if op == "churn":
        limit = min(int(params.get("limit", 200) or 200), 2000)
        top = min(int(params.get("top", 15) or 15), 100)
        res = _git(repo_path, ["log", f"-n{limit}", "--name-only", "--pretty=format:"])
        counts = {}
        for line in res.stdout.splitlines():
            f = line.strip()
            if f:
                counts[f] = counts.get(f, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
        hotspots = [{"file": f, "changes": n} for f, n in ranked]
        text = "\n".join(f"{h['changes']:>4}  {h['file']}" for h in hotspots)
        return {"op": "churn", "hotspots": hotspots}, text

    if op == "contributors":
        limit = min(int(params.get("limit", 2000) or 2000), 20000)
        res = _git(repo_path, ["log", f"-n{limit}", "--pretty=format:%an"])
        counts = {}
        for name in res.stdout.splitlines():
            name = name.strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:50]
        contributors = [{"author": a, "commits": n} for a, n in ranked]
        text = "\n".join(f"{c['commits']:>5}  {c['author']}" for c in contributors)
        return {"op": "contributors", "contributors": contributors}, text

    raise ValueError(f"Unknown git op: {op}")


# ------------------------------------------------------------------- planning
APPROACHES = {
    "thorough": ("Decompose the ask into several small, individually verifiable steps. "
                 "Prefer multiple narrow `search` steps each feeding a focused `extract`, "
                 "then a final `report`. Favor auditability over brevity."),
    "fast": ("Keep it minimal: one or two `search` steps, one `extract`, one `report`."),
    "git-forensic": ("Lead with `git` steps (log, diff, churn) so the analysis is anchored "
                     "in what actually changed, then add `search` + `extract` + `report`."),
    "evidence-first": ("Maximize grounding: pair every `extract` with exactly one `search` or `git` "
                       "step, require verbatim-quote evidence in every extracted fact, and add a "
                       "`transform` step to count/rank facts deterministically before the `report`."),
}

APPROACH_LABELS = {
    "thorough": "Thorough — many small, verifiable steps",
    "fast": "Fast — minimal steps, quick answer",
    "git-forensic": "Git forensic — anchor on real commits & diffs",
    "evidence-first": "Evidence-first — quotes + deterministic counts",
}


def approach_catalog(has_git=True):
    """Approaches the planner can take, for the UI. Each produces a different
    decomposition of the same ask, so users can compare pipelines, not prompts."""
    out = []
    for key, guidance in APPROACHES.items():
        if key == "git-forensic" and not has_git:
            continue
        out.append({"id": key, "label": APPROACH_LABELS.get(key, key), "guidance": guidance})
    return out

_PLANNER_SYSTEM = """You design auditable analysis pipelines for a code-exploration tool.

A pipeline is JSON: {"name","description","steps":[...]}. Each step is:
{
  "id": short snake_case id (unique),
  "title": human label,
  "type": one of "git" | "search" | "extract" | "transform" | "report",
  "model": null (use defaults),
  "instructions": for extract/report, what the model must do,
  "inputs": array of prior step ids this step consumes (and/or the literal "ask"),
  "output": "json" or "text",
  "params": type-specific knobs
}

Step types:
- git (deterministic; only if the repo is a git repo): params.op in {"log","diff","churn","contributors"};
  diff supports params.last (N commits) or params.base/params.head. Produces real commits/diffs.
- search (deterministic): params.query (string) runs grounded hybrid code search; returns matching snippets.
- extract (LLM): reads its inputs and returns a JSON array/object of small, typed, evidence-bearing facts.
  Each fact SHOULD cite provenance (file path or commit sha) it came from. Do not summarize into prose.
- transform (deterministic): params.op in {"top_n","count_by","flatten","passthrough"}, params.field, params.n.
- report (LLM, output "text"): writes the final Markdown, citing the file paths / commits from prior steps.

Rules:
- Ground every LLM step on a prior git/search step via inputs. Never let an extract/report invent facts.
- The LAST step should be a single "report".
- Keep it to at most 7 steps. Return ONLY the JSON object."""


def plan_pipeline(registry, ask, approach="thorough", has_git=True, model=None):
    """Ask an LLM (user-selectable `model`) to break the ask into a JSON pipeline.
    Different `approach` values yield different decompositions of the same ask."""
    guidance = APPROACHES.get(approach, APPROACHES["thorough"])
    git_note = ("The target repo IS a git repository; you may use `git` steps."
                if has_git else "The target repo is NOT a git repository; do NOT use `git` steps.")
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": f"{git_note}\nApproach: {guidance}\n\nAsk: {ask}\n\nDesign the pipeline JSON."},
    ]
    try:
        raw = registry.chat(messages, model=model or registry.active_fast_llm,
                            schema=PipelineSpec, think=False)
        spec = _parse_spec(raw)
        if spec and spec.steps:
            _normalize_spec(spec)
            return spec.model_dump()
    except Exception as e:
        logger.warning(f"Pipeline planning failed, using fallback template: {e}")
    # Fallback: a sensible default template.
    return (git_forensics_template() if has_git else generic_template(ask))


def _parse_spec(raw):
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    if not text.startswith("{"):
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            text = text[s:e + 1]
    return PipelineSpec(**json.loads(text))


def _dedupe_ids(spec):
    seen = set()
    for i, step in enumerate(spec.steps):
        if not step.id or step.id in seen:
            step.id = f"step_{i + 1}"
        seen.add(step.id)


_DEFAULT_EXTRACT = ("Extract a JSON array of small, typed facts relevant to the ask from the inputs. "
                    "Each fact must include: {claim, file (or commit sha), evidence (a short verbatim "
                    "quote from the input)}. Only include facts grounded in the inputs.")
_DEFAULT_REPORT = ("Write a concise, well-structured Markdown report answering the ask, using ONLY the "
                   "inputs. Cite the file paths / commit shas provided. Do not invent anything.")


def _normalize_spec(spec):
    """Fill gaps weaker planner models leave so the pipeline still runs well."""
    _dedupe_ids(spec)
    for step in spec.steps:
        if step.type not in STEP_TYPES:
            step.type = "extract"
        if step.type == "extract" and not step.instructions.strip():
            step.instructions = _DEFAULT_EXTRACT
        if step.type == "report":
            step.output = "text"
            if not step.instructions.strip():
                step.instructions = _DEFAULT_REPORT


# ------------------------------------------------------------------ templates
def security_template():
    return {
        "name": "Security review",
        "description": "Find and report likely security issues, grounded in real code.",
        "steps": [
            {"id": "find_auth", "title": "Search auth & secrets", "type": "search", "output": "json",
             "params": {"query": "authentication authorization login password token secret api key", "limit": 10}},
            {"id": "find_input", "title": "Search input handling", "type": "search", "output": "json",
             "params": {"query": "request input validation sql query injection eval deserialize subprocess shell", "limit": 10}},
            {"id": "findings", "title": "Extract findings", "type": "extract", "output": "json",
             "inputs": ["find_auth", "find_input"],
             "instructions": ("From the retrieved snippets, extract concrete security findings. Return a JSON array "
                              "of objects: {severity: 'high'|'medium'|'low', category, file, evidence (a short verbatim "
                              "code quote from the snippet), rationale}. Only include findings you can tie to a specific file and quote.")},
            {"id": "report", "title": "Security report", "type": "report", "output": "text",
             "inputs": ["findings"],
             "instructions": ("Write a concise Markdown security report grouped by severity. For every finding, cite the "
                              "file path and include the evidence quote. Do not introduce issues not present in the findings input.")},
        ],
    }


def architecture_template():
    return {
        "name": "Architecture overview",
        "description": "Explain the system's main components and how they fit together.",
        "steps": [
            {"id": "entry", "title": "Search entry points", "type": "search", "output": "json",
             "params": {"query": "main entry point server app routes startup configuration bootstrap", "limit": 10}},
            {"id": "data", "title": "Search data & integrations", "type": "search", "output": "json",
             "params": {"query": "database model schema client external API queue cache", "limit": 10}},
            {"id": "components", "title": "Extract components", "type": "extract", "output": "json",
             "inputs": ["entry", "data"],
             "instructions": ("Extract the main components. Return a JSON array of {name, responsibility, files (array of "
                              "paths from the snippets), key_entities}. Ground every component in a file you were shown.")},
            {"id": "report", "title": "Architecture report", "type": "report", "output": "text",
             "inputs": ["components"],
             "instructions": ("Write a Markdown architecture overview: a short intro, then a section per component citing its "
                              "files, then a 'How it fits together' paragraph. Only use the components provided.")},
        ],
    }


def git_forensics_template():
    return {
        "name": "Change forensics (git)",
        "description": "What changed recently, where the risk is, grounded in real commits and diffs.",
        "steps": [
            {"id": "log", "title": "Recent commits", "type": "git", "output": "json",
             "params": {"op": "log", "limit": 20}},
            {"id": "churn", "title": "Change hotspots", "type": "git", "output": "json",
             "params": {"op": "churn", "limit": 200, "top": 15}},
            {"id": "diff", "title": "Recent diff", "type": "git", "output": "json",
             "params": {"op": "diff", "last": 5}},
            {"id": "changes", "title": "Extract changes & risk", "type": "extract", "output": "json",
             "inputs": ["log", "churn", "diff"],
             "instructions": ("From the real commits, hotspots, and diff, extract a JSON array of {change, files, "
                              "commit (sha if known), risk: 'high'|'medium'|'low', evidence (quote from the diff or commit subject)}. "
                              "Only describe changes present in the inputs.")},
            {"id": "report", "title": "Change report", "type": "report", "output": "text",
             "inputs": ["changes", "churn"],
             "instructions": ("Write a Markdown report: 'What changed recently', 'Hotspots to watch' (from churn), and "
                              "'Risks'. Cite commit shas and file paths from the inputs. Do not invent changes.")},
        ],
    }


def generic_template(ask=""):
    return {
        "name": "Grounded Q&A",
        "description": "Search the codebase, extract grounded facts, and report.",
        "steps": [
            {"id": "search", "title": "Search codebase", "type": "search", "output": "json",
             "params": {"query": ask or "overview", "limit": 10}},
            {"id": "facts", "title": "Extract facts", "type": "extract", "output": "json",
             "inputs": ["ask", "search"],
             "instructions": ("Extract a JSON array of small, typed facts that help answer the ask. Each: "
                              "{claim, file, evidence (verbatim quote from a snippet)}. Ground every fact in a snippet.")},
            {"id": "report", "title": "Report", "type": "report", "output": "text",
             "inputs": ["ask", "facts"],
             "instructions": "Answer the ask in Markdown, citing file paths. Use only the provided facts."},
        ],
    }


def builtin_templates(has_git=True):
    templates = [security_template(), architecture_template(), generic_template()]
    if has_git:
        templates.insert(0, git_forensics_template())
    return templates


# ------------------------------------------------------------------- runner
def _clip(text, cap=_INPUT_CAP):
    text = text if isinstance(text, str) else json.dumps(text, default=str, indent=2)
    return text if len(text) <= cap else text[:cap] + "\n... [truncated]"


def _extract_json(raw):
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opn, cls in (("[", "]"), ("{", "}")):
            s, e = text.find(opn), text.rfind(cls)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(text[s:e + 1])
                except json.JSONDecodeError:
                    continue
    raise json.JSONDecodeError("No JSON found", text, 0)


def _stale_dependents(spec, step_id):
    """Step ids after `step_id` (in spec order) that transitively depend on it.

    Steps with no explicit inputs implicitly consume everything computed before
    them (see `_resolve_inputs`), so they are treated as dependents too.
    """
    ids = [s.id for s in spec.steps]
    if step_id not in ids:
        return []
    target_i = ids.index(step_id)
    stale = set()
    for i, s in enumerate(spec.steps):
        if i <= target_i:
            continue
        deps = [r for r in (s.inputs or []) if r != "ask"]
        if not deps or step_id in deps or any(d in stale for d in deps):
            stale.add(s.id)
    return [sid for sid in ids if sid in stale]


class PipelineRunner:
    def __init__(self, registry, coll, codebase_id, repo_path=None, has_git=False,
                 runs_collection=None):
        self.registry = registry
        self.coll = coll
        self.codebase_id = codebase_id
        self.repo_path = repo_path
        self.has_git = has_git
        self.runs_collection = runs_collection

    def run(self, spec_dict, ask=""):
        try:
            spec = PipelineSpec(**spec_dict)
        except Exception as e:
            yield {"type": "error", "error": f"Invalid pipeline spec: {e}"}
            return
        _normalize_spec(spec)

        run_id = str(uuid.uuid4())
        yield {"type": "pipeline_start", "run_id": run_id, "name": spec.name,
               "steps": [{"id": s.id, "title": s.title, "type": s.type,
                          "model": s.model or default_model_for(s.type, self.registry)}
                         for s in spec.steps]}

        results = {}                       # step id -> parsed output
        text_results = {"__ask__": ask or ""}  # step id -> text rendering for LLM input
        trail = []
        final_output = ""

        for step in spec.steps:
            yield {"type": "step_start", "id": step.id, "title": step.title, "type_": step.type}
            started = time.time()
            used_model = step.model or default_model_for(step.type, self.registry)
            audit = {
                "id": step.id, "title": step.title, "type": step.type,
                "model": used_model, "status": "ok",
                "started_at": dt.datetime.utcnow().isoformat() + "Z",
                "input_preview": None, "prompt": None,
                "output": None, "text_output": None, "raw_output": None, "error": None,
            }
            try:
                out, text_out, prompt = self._exec(step, results, text_results)
                results[step.id] = out
                text_results[step.id] = text_out
                audit["output"] = out
                audit["text_output"] = _clip(text_out)
                audit["prompt"] = prompt
                audit["input_preview"] = _clip(self._resolve_inputs(step, text_results), 2000)
                if step.type == "report":
                    final_output = out if isinstance(out, str) else text_out
            except Exception as e:
                audit["status"] = "error"
                audit["error"] = str(e)
                logger.warning(f"Pipeline step '{step.id}' failed: {e}")
            audit["duration_ms"] = int((time.time() - started) * 1000)
            trail.append(audit)
            yield {"type": "step_done", "audit": audit}

        status = "error" if any(a["status"] == "error" for a in trail) else "ok"
        if not final_output and trail:
            last = trail[-1]
            final_output = last.get("output") if isinstance(last.get("output"), str) else _clip(last.get("output"), 4000)

        run_doc = {
            "_id": run_id, "codebase_id": self.codebase_id, "name": spec.name,
            "description": spec.description, "spec": spec.model_dump(),
            "trail": trail, "status": status, "final_output": final_output,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        }
        if self.runs_collection is not None:
            try:
                self.runs_collection.replace_one({"_id": run_id}, run_doc, upsert=True)
            except Exception as e:
                logger.warning(f"Could not persist pipeline run: {e}")

        yield {"type": "pipeline_done", "run_id": run_id, "status": status,
               "final": final_output}

    def run_step(self, spec_dict, step_id, ask="", run_id=None, prior_trail=None):
        """Re-run a single step against the cached outputs of its upstream steps,
        stitch the new audit entry into the saved run's trail, and mark
        downstream dependents stale (they keep their old output until re-run)."""
        try:
            spec = PipelineSpec(**spec_dict)
        except Exception as e:
            yield {"type": "error", "error": f"Invalid pipeline spec: {e}"}
            return
        _normalize_spec(spec)

        step = next((s for s in spec.steps if s.id == step_id), None)
        if step is None:
            yield {"type": "error", "error": f"Step '{step_id}' not found in spec."}
            return

        # Prefer the persisted trail; fall back to one supplied by the client.
        trail = None
        run_doc = None
        if run_id and self.runs_collection is not None:
            run_doc = self.runs_collection.find_one({"_id": run_id})
            if run_doc:
                trail = run_doc.get("trail")
        if trail is None:
            trail = prior_trail or []
        trail_by_id = {a.get("id"): a for a in trail if isinstance(a, dict)}

        # Seed upstream context from cached audit entries (spec order).
        results = {}
        text_results = {"__ask__": ask or ""}
        for s in spec.steps:
            if s.id == step_id:
                break
            a = trail_by_id.get(s.id)
            if not a or a.get("output") is None:
                continue
            results[s.id] = a.get("output")
            text_results[s.id] = a.get("text_output") or _clip(a.get("output"))

        yield {"type": "step_start", "id": step.id, "title": step.title, "type_": step.type}
        started = time.time()
        used_model = step.model or default_model_for(step.type, self.registry)
        audit = {
            "id": step.id, "title": step.title, "type": step.type,
            "model": used_model, "status": "ok",
            "started_at": dt.datetime.utcnow().isoformat() + "Z",
            "input_preview": None, "prompt": None,
            "output": None, "text_output": None, "raw_output": None, "error": None,
        }
        try:
            out, text_out, prompt = self._exec(step, results, text_results)
            audit["output"] = out
            audit["text_output"] = _clip(text_out)
            audit["prompt"] = prompt
            audit["input_preview"] = _clip(self._resolve_inputs(step, text_results), 2000)
        except Exception as e:
            audit["status"] = "error"
            audit["error"] = str(e)
            logger.warning(f"Pipeline step '{step.id}' re-run failed: {e}")
        audit["duration_ms"] = int((time.time() - started) * 1000)
        yield {"type": "step_done", "audit": audit}

        # Stitch: replace the entry in the trail, flag dependents as stale.
        stale_ids = _stale_dependents(spec, step_id)
        trail_by_id[step.id] = audit
        for sid in stale_ids:
            entry = trail_by_id.get(sid)
            if entry and entry.get("status") == "ok":
                entry["status"] = "stale"
        new_trail = [trail_by_id[s.id] for s in spec.steps if s.id in trail_by_id]

        status = "error" if any(a.get("status") == "error" for a in new_trail) else "ok"
        final_output = ""
        for s in spec.steps:
            a = trail_by_id.get(s.id)
            if a and s.type == "report" and a.get("status") in ("ok", "stale") and a.get("output"):
                final_output = a["output"] if isinstance(a["output"], str) else (a.get("text_output") or "")

        if run_doc is not None and self.runs_collection is not None:
            try:
                self.runs_collection.update_one(
                    {"_id": run_id},
                    {"$set": {"trail": new_trail, "status": status,
                              "final_output": final_output,
                              "updated_at": dt.datetime.utcnow().isoformat() + "Z"}})
            except Exception as e:
                logger.warning(f"Could not persist re-run step: {e}")

        yield {"type": "step_run_done", "run_id": run_id, "status": status,
               "stale": stale_ids, "final": final_output}


    # --------------------------------------------------------------- helpers
    def _resolve_inputs(self, step, text_results):
        parts = []
        for ref in (step.inputs or []):
            if ref == "ask":
                parts.append(f"## ASK\n{text_results.get('__ask__', '')}")
            elif ref in text_results:
                parts.append(f"## FROM STEP `{ref}`\n{text_results[ref]}")
        if not parts:
            # No explicit inputs: feed the ask + everything computed so far.
            if "__ask__" in text_results:
                parts.append(f"## ASK\n{text_results['__ask__']}")
            for k, v in text_results.items():
                if k != "__ask__":
                    parts.append(f"## FROM STEP `{k}`\n{v}")
        return _clip("\n\n".join(parts))

    def _exec(self, step, results, text_results):
        if step.type == "git":
            return self._exec_git(step)
        if step.type == "search":
            return self._exec_search(step)
        if step.type == "extract":
            return self._exec_extract(step, text_results)
        if step.type == "transform":
            return self._exec_transform(step, results)
        if step.type == "report":
            return self._exec_report(step, text_results)
        raise ValueError(f"Unknown step type: {step.type}")

    def _exec_git(self, step):
        if not self.has_git or not self.repo_path:
            return {"available": False, "note": "Not a git repository."}, "Not a git repository.", None
        structured, text = run_git(self.repo_path, step.params.get("op", "log"), step.params)
        return structured, text, None

    def _exec_search(self, step):
        query = step.params.get("query") or step.title or "overview"
        limit = min(int(step.params.get("limit", 10) or 10), 25)
        em = self.registry.active_embedding()
        qvec = self.registry.embed([query], em, kind="query")[0]
        hits = dbmod.hybrid_search(self.coll, self.codebase_id, query, qvec, em, limit=limit)
        trimmed = []
        text_lines = [f"Query: {query}"]
        for h in hits:
            la = h.get("llm_analysis") or {}
            code = (h.get("code") or "")[:1200]
            trimmed.append({"file_path": h.get("file_path"), "item_name": h.get("item_name"),
                            "summary": la.get("summary", ""), "score": h.get("score"),
                            "code": code})
            text_lines.append(f"\n--- {h.get('file_path')}::{h.get('item_name')} ---\n"
                              f"{la.get('summary', '')}\n{code}")
        return {"query": query, "hits": trimmed}, "\n".join(text_lines), None

    def _exec_extract(self, step, text_results):
        inputs_text = self._resolve_inputs(step, text_results)
        prompt = (f"{step.instructions}\n\n"
                  f"Use ONLY the inputs below. Ground every item in a file path or commit present in them.\n\n"
                  f"{inputs_text}")
        messages = [
            {"role": "system", "content": "You extract small, typed, evidence-bearing JSON facts. "
                                          "Return ONLY JSON (an array or object). Never invent facts not in the inputs."},
            {"role": "user", "content": prompt},
        ]
        raw = self.registry.chat(messages, model=step.model or self.registry.active_fast_llm,
                                 schema=None, think=False)
        try:
            parsed = _extract_json(raw)
        except Exception:
            parsed = {"_unparsed": raw[:4000]}
        return parsed, _clip(parsed), prompt

    def _exec_transform(self, step, results):
        op = step.params.get("op", "passthrough")
        field = step.params.get("field")
        n = int(step.params.get("n", 10) or 10)
        src = []
        for ref in (step.inputs or []):
            v = results.get(ref)
            if isinstance(v, list):
                src.extend(v)
            elif isinstance(v, dict) and isinstance(v.get("hits"), list):
                src.extend(v["hits"])
            elif v is not None:
                src.append(v)
        if op == "count_by" and field:
            counts = {}
            for item in src:
                key = str((item or {}).get(field, "unknown")) if isinstance(item, dict) else str(item)
                counts[key] = counts.get(key, 0) + 1
            out = [{"key": k, "count": c} for k, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
        elif op == "top_n":
            out = src[:n]
        elif op == "flatten":
            out = src
        else:
            out = src
        return out, _clip(out), None

    def _exec_report(self, step, text_results):
        inputs_text = self._resolve_inputs(step, text_results)
        prompt = (f"{step.instructions}\n\nWrite well-structured Markdown. Cite file paths and commit shas "
                  f"from the inputs. Do not introduce facts not present below.\n\n{inputs_text}")
        messages = [
            {"role": "system", "content": "You are a precise technical writer. You synthesize a final Markdown "
                                          "document strictly from the provided structured inputs, citing their provenance."},
            {"role": "user", "content": prompt},
        ]
        text = self.registry.chat(messages, model=step.model or self.registry.active_llm, think=False)
        return text, text, prompt

    def seed_ask(self, text_results, ask):
        text_results["__ask__"] = ask or ""
