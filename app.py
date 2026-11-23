#!/usr/bin/env python3
"""code-xplorer: local-first code analysis, hybrid search, and a multi-step chat agent.

Flask app that wires together the xplorer package:
  - xplorer.config    runtime configuration (local-first defaults)
  - xplorer.models    model registry (Ollama default, Voyage/Azure optional)
  - xplorer.parsing   structure extraction + full-coverage line-window fallback
  - xplorer.db        Mongo setup, per-model vector indexes, hybrid search
  - xplorer.indexer   validated metadata extraction + chunked summarization
  - xplorer.agent     plan -> retrieve -> grade -> expand -> draft -> verify
"""
import os
import logging
import json
import re
import subprocess
import tempfile
import shutil
import atexit
import threading
import concurrent.futures
import uuid
import hashlib
import time

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from xplorer.config import load_config
from xplorer.models import ModelRegistry
from xplorer import parsing
from xplorer import db as dbmod
from xplorer import indexer as idx
from xplorer import pipeline as pipe
from xplorer.agent import ChatAgent

# --- Initialization & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

app = Flask(__name__)
config = load_config()
registry = ModelRegistry(config)

# --- Global state ---
SCANNED_FILES_STRUCTURE = {}
WHOLE_FILE_SUMMARY_CACHE = {}
mongo_client = None
db = None
code_collection = None
codebase_state_collection = None
pipeline_runs_collection = None
SCAN_SESSIONS = {}

indexing_lock = threading.Lock()
_index_create_lock = threading.Lock()
_ensured_vector_indexes = set()


# --- Core helpers ---------------------------------------------------------
def generate_codebase_id(path_or_url: str) -> str:
    normalized_input = path_or_url.strip()
    repo_info = parse_github_input(normalized_input)
    if repo_info:
        identifier = repo_info['url'].lower()
        if identifier.endswith('.git'):
            identifier = identifier[:-4]
    else:
        identifier = os.path.abspath(normalized_input)
    return hashlib.sha256(identifier.encode()).hexdigest()


def parse_github_input(input_string):
    input_string = input_string.strip()
    https_pattern = re.compile(r'^(?:https?://)?(?:www\.)?github\.com/([\w\-.]+)/([\w\-.]+?)(?:\.git)?/?$')
    match = https_pattern.match(input_string)
    if match:
        owner, repo = match.groups()
        return {"url": f"https://github.com/{owner}/{repo}.git", "name": repo}

    ssh_pattern = re.compile(r'^git@github\.com:([\w\-.]+)/([\w\-.]+?)\.git$')
    match = ssh_pattern.match(input_string)
    if match:
        owner, repo = match.groups()
        return {"url": f"https://github.com/{owner}/{repo}.git", "name": repo}

    shorthand_pattern = re.compile(r'^([\w\-.]+)/([\w\-.]+)$')
    match = shorthand_pattern.match(input_string)
    if match and '.' not in match.group(1):
        owner, repo = match.groups()
        return {"url": f"https://github.com/{owner}/{repo}.git", "name": repo}
    return None


def clone_repo_to_tempdir(repo_url):
    temp_dir = tempfile.mkdtemp()
    logging.info(f"Cloning {repo_url} into temporary directory: {temp_dir}")
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, temp_dir],
            check=True, capture_output=True, text=True
        )
        logging.info("Repository cloned successfully.")
        return temp_dir
    except FileNotFoundError:
        logging.error("'git' command not found. Please ensure Git is installed and in your PATH.")
        shutil.rmtree(temp_dir)
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone repository: {e.stderr}")
        shutil.rmtree(temp_dir)
        raise


def scan_directory(path, session_id):
    """Scan a directory and populate the session with file data + action tags."""
    scanned_files_data = {}
    logging.info(f"[{session_id}] Starting directory scan at: {os.path.abspath(path)}")

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in parsing.IGNORE_DIRS]
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in parsing.IGNORE_EXTS or ext in parsing.BINARY_EXTS:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logging.warning(f"[{session_id}] Could not read file {file_path}: {e}")
                continue
            if not content.strip():
                continue
            relative_path = os.path.relpath(file_path, path)
            _, file_tags = parsing.get_structure_or_windows(relative_path, content)
            scanned_files_data[relative_path] = {
                "content": content,
                "char_count": len(content),
                "tags": file_tags,
            }

    logging.info(f"[{session_id}] Scan complete. Found {len(scanned_files_data)} relevant files.")
    SCAN_SESSIONS[session_id]['scanned_files_data'] = scanned_files_data


def _enclosing_scope(structure, item):
    """Return the name of the class/scope that encloses a function item, if any."""
    start = item.get('start_line')
    end = item.get('end_line')
    best = None
    for other in structure:
        if other is item or other.get('type') != 'class':
            continue
        o_start = other.get('start_line')
        o_end = other.get('end_line')
        if o_start is None:
            continue
        if o_start <= start and (o_end is None or (end or start) <= o_end):
            if best is None or o_start > best.get('start_line'):
                best = other
    return best['name'] if best else ""


# --- Vector index management ---------------------------------------------
def ensure_active_vector_index(em, wait=False):
    """Make sure a vectorSearch index exists for the given embedding model,
    sized to its real (probed) dimensions. Returns (index_name, dims)."""
    if code_collection is None:
        return None, None
    with _index_create_lock:
        dims = registry.probe_dims(em)
        if em.id not in _ensured_vector_indexes:
            dbmod.ensure_vector_index(code_collection, em, dims)
            _ensured_vector_indexes.add(em.id)
    if wait:
        dbmod.wait_for_index_ready(code_collection, em.index_name)
    return em.index_name, dims


# --- Indexing -------------------------------------------------------------
def _index_one_snippet(snippet_data):
    """Analyze + embed a single snippet with the active embedding model."""
    session_id = snippet_data['session_id']
    codebase_id = snippet_data['codebase_id']
    em = snippet_data['embedding_model']
    with indexing_lock:
        SCAN_SESSIONS[session_id]['indexing_status']["progress"] += 1

    file_path = snippet_data['file_path']
    item_name = snippet_data['item_name']
    code_to_process = snippet_data['code']
    static_tags = snippet_data.get('tags', [])
    import_header = snippet_data.get('import_header', '')
    class_ctx = snippet_data.get('class_ctx', '')
    unique_id = f"{codebase_id}-{file_path}::{item_name}"

    if not code_to_process.strip():
        return "skipped", snippet_data

    try:
        analysis = idx.analyze_snippet(
            registry, file_path, item_name, code_to_process,
            model=registry.active_fast_llm,
            import_header=import_header, class_ctx=class_ctx, static_tags=static_tags,
        )
        embed_text = idx.build_embedding_text(em, file_path, analysis.get("description", ""), code_to_process)
        embedding = registry.embed([embed_text], em, kind="doc")[0]

        combined_tags = sorted(set(static_tags + (analysis.get('tags') or [])))
        document = {
            "codebase_id": codebase_id,
            "session_id": session_id,
            "file_path": file_path,
            "item_name": item_name,
            "code": code_to_process,
            "llm_analysis": analysis,
            "tags": combined_tags,
        }
        code_collection.update_one(
            {"_id": unique_id},
            {"$set": {**document, em.field_path: embedding}},
            upsert=True,
        )
        return "ok", snippet_data
    except Exception as e:
        logging.error(f"Failed to index snippet {unique_id}: {e}")
        return "fail", snippet_data


def _collect_snippets(files_data, session_id, codebase_id, em):
    snippets = []
    for file_path, data in files_data.items():
        content = data["content"]
        structure, _ = parsing.get_structure_or_windows(file_path, content)
        file_ext = os.path.splitext(file_path)[1].lower()
        import_header = parsing.get_import_header(content, file_path)

        for item in structure:
            code_block = parsing.extract_code_block(
                content, item['start_line'], item.get('end_line'), file_ext)
            if not code_block.strip():
                continue
            snippets.append({
                "file_path": file_path,
                "item_name": item['name'],
                "code": code_block,
                "session_id": session_id,
                "codebase_id": codebase_id,
                "tags": item.get('tags', []),
                "import_header": import_header,
                "class_ctx": _enclosing_scope(structure, item),
                "embedding_model": em,
            })
    return snippets


def process_and_index_snippets(files_data, session_id, codebase_id):
    """Find all snippets and index them in parallel, with provider-aware
    concurrency and a one-shot retry of failures."""
    if code_collection is None:
        msg = "Cannot index: MongoDB is not configured."
        logging.error(f"[{session_id}] {msg}")
        SCAN_SESSIONS[session_id]['status'] = 'error'
        SCAN_SESSIONS[session_id]['error_message'] = msg
        return

    em = registry.active_embedding()
    if not registry.embedding_available(em):
        msg = (f"Embedding model '{em.id}' is not available. "
               f"Pull it (Ollama) or set VOYAGE_API_KEY, then rescan.")
        logging.error(f"[{session_id}] {msg}")
        SCAN_SESSIONS[session_id]['indexing_status'] = {
            "running": False, "progress": 0, "total": 0, "message": msg}
        SCAN_SESSIONS[session_id]['status'] = 'error'
        SCAN_SESSIONS[session_id]['error_message'] = msg
        return

    # Ensure the vector index exists (sized to the real embedding dimensions).
    try:
        ensure_active_vector_index(em)
    except Exception as e:
        logging.warning(f"[{session_id}] Could not ensure vector index yet: {e}")

    snippets = _collect_snippets(files_data, session_id, codebase_id, em)
    if not snippets:
        SCAN_SESSIONS[session_id]['indexing_status'] = {
            "running": False, "progress": 0, "total": 0, "message": "No changes to index"}
        return

    workers = 4 if em.provider == "voyage" else 3
    SCAN_SESSIONS[session_id]['indexing_status'] = {
        "running": True, "progress": 0, "total": len(snippets),
        "message": f"Indexing with {em.label}...", "model": em.id}
    logging.info(f"[{session_id}] Indexing {len(snippets)} snippets with {em.id} ({workers} workers).")

    failed = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_index_one_snippet, s): s for s in snippets}
            for future in concurrent.futures.as_completed(futures):
                try:
                    status, snip = future.result()
                    if status == "fail":
                        failed.append(snip)
                except Exception as exc:
                    logging.error(f'[{session_id}] Indexing task raised: {exc}')

        # Retry failures once, serially.
        still_failed = 0
        if failed:
            logging.info(f"[{session_id}] Retrying {len(failed)} failed snippets.")
            for snip in failed:
                with indexing_lock:
                    SCAN_SESSIONS[session_id]['indexing_status']["progress"] = max(
                        0, SCAN_SESSIONS[session_id]['indexing_status']["progress"] - 1)
                status, _ = _index_one_snippet(snip)
                if status == "fail":
                    still_failed += 1

        if still_failed:
            SCAN_SESSIONS[session_id]['indexing_status']["message"] = (
                f"Indexing complete with {still_failed} snippet(s) failed.")
        else:
            SCAN_SESSIONS[session_id]['indexing_status']["message"] = "Indexing complete"
        logging.info(f"[{session_id}] Background indexing complete ({still_failed} failures).")
    except Exception as e:
        logging.error(f"[{session_id}] Error during parallel indexing: {e}")
        SCAN_SESSIONS[session_id]['indexing_status']["message"] = f"Error during indexing: {e}"
    finally:
        SCAN_SESSIONS[session_id]['indexing_status']["running"] = False


def delete_indexed_files(codebase_id, file_paths):
    if not file_paths or code_collection is None:
        return
    try:
        code_collection.delete_many({"codebase_id": codebase_id, "file_path": {"$in": file_paths}})
        logging.info(f"Deleted snippets for {len(file_paths)} files from codebase {codebase_id}")
    except Exception as e:
        logging.error(f"Failed to delete indexed files for codebase {codebase_id}: {e}")


def background_scan_and_index(session_id, path_or_url):
    """Clone/find repo, detect changes vs stored state, surgically update index."""
    scan_path = None
    is_temp_dir = False
    session = SCAN_SESSIONS.get(session_id)
    if not session:
        return

    try:
        codebase_id = generate_codebase_id(path_or_url)
        session['codebase_id'] = codebase_id
        logging.info(f"[{session_id}] Processing codebase ID: {codebase_id}")

        repo_info = parse_github_input(path_or_url)
        session['source'] = path_or_url
        session['git_url'] = repo_info["url"] if repo_info else None
        if repo_info:
            session['status'] = 'cloning'
            session['message'] = f"Cloning {repo_info['name']}..."
            scan_path = clone_repo_to_tempdir(repo_info["url"])
            is_temp_dir = True
            session['local_path'] = None
        elif os.path.isdir(path_or_url):
            scan_path = path_or_url
            session['local_path'] = os.path.abspath(scan_path)
        else:
            raise ValueError(f"Input '{path_or_url}' is not a valid directory or GitHub repository.")

        session['display_name'] = repo_info["name"] if repo_info else os.path.basename(os.path.abspath(scan_path))
        session['status'] = 'scanning'
        session['message'] = 'Detecting file changes...'

        previous_state = {
            doc['file_path']: doc['content_hash']
            for doc in codebase_state_collection.find({"codebase_id": codebase_id})
        }
        logging.info(f"[{session_id}] Found {len(previous_state)} files in last known state.")

        scan_directory(scan_path, session_id)
        files_data_for_indexing = session.get('scanned_files_data', {})

        current_state = {
            path: hashlib.sha256(data['content'].encode()).hexdigest()
            for path, data in files_data_for_indexing.items()
        }

        previous_paths = set(previous_state.keys())
        current_paths = set(current_state.keys())
        new_files = list(current_paths - previous_paths)
        deleted_files = list(previous_paths - current_paths)
        modified_files = [p for p in previous_paths & current_paths if previous_state[p] != current_state[p]]

        # If the active embedding model has no vectors yet for this codebase,
        # (re)index everything so switching models always produces a usable index.
        active_slug = registry.active_embedding().slug
        needs_full = False
        if code_collection is not None and not (new_files or modified_files):
            try:
                has_any = code_collection.find_one(
                    {"codebase_id": codebase_id, f"embeddings.{active_slug}": {"$exists": True}})
                if has_any is None and code_collection.find_one({"codebase_id": codebase_id}) is None:
                    needs_full = True
            except Exception:
                pass
        if needs_full:
            new_files = list(current_paths)

        logging.info(f"[{session_id}] Changes: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted.")
        session['change_summary'] = {"new": len(new_files), "modified": len(modified_files), "deleted": len(deleted_files)}
        session['scan_complete'] = True

        files_to_index_paths = new_files + modified_files
        if not files_to_index_paths and not deleted_files:
            session['status'] = 'complete'
            session['message'] = 'Codebase is already up-to-date.'
            return

        if deleted_files:
            delete_indexed_files(codebase_id, deleted_files)

        if files_to_index_paths:
            files_to_index_data = {p: files_data_for_indexing[p] for p in files_to_index_paths}
            process_and_index_snippets(files_to_index_data, session_id, codebase_id)

        logging.info(f"[{session_id}] Updating codebase state in database...")
        codebase_state_collection.delete_many({"codebase_id": codebase_id})
        if current_state:
            docs_to_insert = [
                {"codebase_id": codebase_id, "file_path": path, "content_hash": hash_val}
                for path, hash_val in current_state.items()
            ]
            codebase_state_collection.insert_many(docs_to_insert)

        while session.get('indexing_status', {}).get('running', False):
            time.sleep(1)

        session['status'] = 'complete'
        session['message'] = 'All processes complete.'

    except Exception as e:
        logging.error(f"[{session_id}] Error in background worker: {e}")
        session['status'] = 'error'
        session['error_message'] = str(e)
    finally:
        if is_temp_dir and scan_path and os.path.isdir(scan_path):
            logging.info(f"[{session_id}] Cleaning up temporary directory: {scan_path}")
            shutil.rmtree(scan_path)


def backfill_embeddings(session_id, codebase_id, em):
    """Embed-only backfill: add vectors for `em` to every snippet of a codebase
    that doesn't have them yet. The LLM analysis is never re-run."""
    session = SCAN_SESSIONS.get(session_id)
    if session is None or code_collection is None:
        return
    try:
        if not registry.embedding_available(em):
            msg = f"Embedding model '{em.id}' is not available."
            session['status'] = 'error'
            session['error_message'] = msg
            session['indexing_status'] = {"running": False, "progress": 0, "total": 0, "message": msg}
            return

        ensure_active_vector_index(em)
        query = {"codebase_id": codebase_id, em.field_path: {"$exists": False}}
        docs = list(code_collection.find(query, {"_id": 1, "file_path": 1, "code": 1, "llm_analysis": 1}))
        total = len(docs)
        session['indexing_status'] = {
            "running": True, "progress": 0, "total": total,
            "message": f"Re-embedding with {em.label}...", "model": em.id}

        if total == 0:
            session['indexing_status'] = {"running": False, "progress": 0, "total": 0,
                                          "message": "Already embedded."}
        else:
            batch = max(1, em.batch_size)
            for i in range(0, total, batch):
                chunk = docs[i:i + batch]
                texts = [idx.build_embedding_text(
                    em, d.get("file_path", ""), (d.get("llm_analysis") or {}).get("description", ""),
                    d.get("code", "")) for d in chunk]
                try:
                    vecs = registry.embed(texts, em, kind="doc")
                    for d, v in zip(chunk, vecs):
                        code_collection.update_one({"_id": d["_id"]}, {"$set": {em.field_path: v}})
                except Exception as e:
                    logging.error(f"[{session_id}] Backfill batch failed: {e}")
                with indexing_lock:
                    session['indexing_status']["progress"] = min(total, i + len(chunk))
            session['indexing_status']["running"] = False
            session['indexing_status']["message"] = "Re-embedding complete"

        # Make sure the new index is queryable before we report done.
        dbmod.wait_for_index_ready(code_collection, em.index_name, timeout=120)
        session['status'] = 'complete'
        session['message'] = f"Now searching with {em.label}."
    except Exception as e:
        logging.error(f"[{session_id}] Backfill error: {e}")
        session['status'] = 'error'
        session['error_message'] = str(e)
        if 'indexing_status' in session:
            session['indexing_status']["running"] = False


# --- Preselected context --------------------------------------------------
def _structure_for(session_id, file_path, content):
    cache_key = f"{session_id}::{file_path}"
    structure = SCANNED_FILES_STRUCTURE.get(cache_key)
    if structure is None:
        structure, _ = parsing.get_structure_or_windows(file_path, content)
        SCANNED_FILES_STRUCTURE[cache_key] = structure
    return structure


def build_preselected(session_id, data):
    """Turn the UI's manual selections into agent context blocks."""
    session = SCAN_SESSIONS.get(session_id, {})
    scanned = session.get('scanned_files_data', {})
    blocks = []

    for snippet in data.get('custom_snippets', []) or []:
        blocks.append({"source": snippet.get('source', 'custom snippet'),
                       "content": snippet.get('content', ''),
                       "path": snippet.get('path')})

    for summary in data.get('selected_summaries', []) or []:
        desc = f"summary of {summary.get('name') or summary.get('path')}"
        blocks.append({"source": desc, "content": summary.get('content', ''),
                       "path": summary.get('path')})

    for file_path, items in (data.get('selected_items', {}) or {}).items():
        if file_path not in scanned:
            continue
        content = scanned[file_path]["content"]
        if "__all__" in items:
            blocks.append({"source": f"FILE: {file_path}", "content": content, "path": file_path})
            continue
        structure = _structure_for(session_id, file_path, content)
        item_map = {s['name']: s for s in structure}
        file_ext = os.path.splitext(file_path)[1].lower()
        for item_name in items:
            if item_name in item_map:
                item_data = item_map[item_name]
                block = parsing.extract_code_block(
                    content, item_data['start_line'], item_data.get('end_line'), file_ext)
                blocks.append({"source": f"{file_path} ({item_data.get('type')}: {item_name})",
                               "content": block, "path": file_path})
    return blocks


# --- Pipeline repo resolution ---------------------------------------------
def resolve_repo_for_session(session):
    """Return (repo_path, has_git, cleanup_fn) for pipeline git steps.

    - Local directories are used in place (no copy).
    - GitHub sources are shallow-cloned to a temp dir for the run (with enough
      depth for log/diff) and cleaned up afterwards.
    """
    local_path = session.get('local_path')
    if local_path and os.path.isdir(local_path):
        return local_path, pipe.git_available(local_path), None

    git_url = session.get('git_url')
    if git_url:
        temp_dir = tempfile.mkdtemp(prefix="xpl-pipe-")
        try:
            subprocess.run(['git', 'clone', '--depth', '80', git_url, temp_dir],
                           check=True, capture_output=True, text=True)
            return temp_dir, True, (lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        except Exception as e:
            logging.warning(f"Pipeline clone failed for {git_url}: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, False, None

    return None, False, None


# --- Flask Routes ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan_status/<session_id>')
def get_scan_status(session_id):
    session = SCAN_SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    response_data = {
        "status": session.get('status'),
        "message": session.get('message'),
        "error": session.get('error_message'),
        "displayName": session.get('display_name'),
        "indexingStatus": session.get('indexing_status', {}),
        "changeSummary": session.get('change_summary'),
    }

    if session.get('scan_complete') and not session.get('files_sent'):
        files_data = session.get('scanned_files_data', {})
        files_for_frontend = [
            {"path": path, "char_count": data["char_count"], "content": data["content"], "tags": data["tags"]}
            for path, data in files_data.items()
        ]
        files_for_frontend.sort(key=lambda x: x['path'])
        response_data['files'] = files_for_frontend
        session['files_sent'] = True

    return jsonify(response_data)


@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    path_or_url = data.get('path', '.').strip() or '.'

    session_id = str(uuid.uuid4())
    SCAN_SESSIONS[session_id] = {
        "status": "starting", "message": "Initializing...",
        "scan_complete": False, "files_sent": False,
        "scanned_files_data": {}, "indexing_status": {},
        "error_message": None, "display_name": "", "codebase_id": None,
    }

    if len(SCAN_SESSIONS) > 10:
        oldest_session = next(iter(SCAN_SESSIONS))
        del SCAN_SESSIONS[oldest_session]

    thread = threading.Thread(target=background_scan_and_index, args=(session_id, path_or_url))
    thread.start()
    return jsonify({"sessionId": session_id}), 202


@app.route('/structure', methods=['POST'])
def get_structure():
    data = request.get_json()
    file_path = data.get('path')
    session_id = data.get('sessionId')
    if not all([file_path, session_id]):
        return jsonify({"error": "File path and session ID are required."}), 400

    session = SCAN_SESSIONS.get(session_id)
    if not session or not session.get('scanned_files_data'):
        return jsonify({"error": "Session not found or scan not complete."}), 404

    scanned_files_data = session['scanned_files_data']
    cache_key = f"{session_id}::{file_path}"
    if cache_key in SCANNED_FILES_STRUCTURE:
        return jsonify(SCANNED_FILES_STRUCTURE[cache_key])

    if file_path in scanned_files_data:
        content = scanned_files_data[file_path]["content"]
        structure, _ = parsing.get_structure_or_windows(file_path, content)
        SCANNED_FILES_STRUCTURE[cache_key] = structure
        return jsonify(structure)

    return jsonify({"error": "File not found in scanned data for this session."}), 404


@app.route('/extract_snippet', methods=['POST'])
def extract_snippet():
    data = request.get_json()
    file_path = data.get('path')
    item_name = data.get('name')
    session_id = data.get('sessionId')
    if not all([file_path, item_name, session_id]):
        return jsonify({"error": "File path, item name, and session ID are required."}), 400

    session = SCAN_SESSIONS.get(session_id)
    if not session or 'scanned_files_data' not in session:
        return jsonify({"error": "Session data not found."}), 404

    scanned_files_data = session['scanned_files_data']
    if file_path not in scanned_files_data:
        return jsonify({"error": "File not found in scanned data."}), 404

    content = scanned_files_data[file_path]["content"]
    structure = _structure_for(session_id, file_path, content)
    item_data = next((item for item in structure if item['name'] == item_name), None)

    if item_data:
        file_ext = os.path.splitext(file_path)[1].lower()
        block_content = parsing.extract_code_block(
            content, item_data['start_line'], item_data.get('end_line'), file_ext)
        return jsonify({"content": block_content})
    return jsonify({"error": f"Item '{item_name}' not found in '{file_path}'"}), 404


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    session_id = data.get('sessionId')
    file_path = data.get('path')
    item_name = data.get('name')
    if not all([session_id, file_path]):
        return jsonify({"error": "Session ID and file path are required."}), 400

    session = SCAN_SESSIONS.get(session_id)
    if not session or 'scanned_files_data' not in session:
        return jsonify({"error": "Session data not found."}), 404

    codebase_id = session.get('codebase_id')
    scanned_files_data = session['scanned_files_data']
    if file_path not in scanned_files_data:
        return jsonify({"error": "File not found in scanned data."}), 404

    if item_name:
        if code_collection is None:
            return jsonify({"error": "Database not available to fetch snippet summary."}), 503
        document = code_collection.find_one(
            {"codebase_id": codebase_id, "file_path": file_path, "item_name": item_name})
        if document and "llm_analysis" in document and "summary" in document["llm_analysis"]:
            return jsonify({"answer": document["llm_analysis"]["summary"]})
        return jsonify({"answer": "Summary not found in the index. It may still be processing."})

    cache_key = f"{session_id}::{file_path}"
    if cache_key in WHOLE_FILE_SUMMARY_CACHE:
        return jsonify({"answer": WHOLE_FILE_SUMMARY_CACHE[cache_key]})

    content = scanned_files_data[file_path]["content"]
    try:
        answer = idx.summarize_long_text(registry, content, model=registry.active_fast_llm, label=file_path)
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {e}"}), 500

    if "unavailable" not in answer.lower():
        WHOLE_FILE_SUMMARY_CACHE[cache_key] = answer
    return jsonify({"answer": answer})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get('sessionId')
    if not session_id or session_id not in SCAN_SESSIONS:
        return jsonify({"error": "Invalid or expired session. Please scan a repository again."}), 400

    user_question = data.get('question')
    if not user_question:
        return jsonify({"error": "No user question found in the request."}), 400

    codebase_id = SCAN_SESSIONS[session_id].get('codebase_id')
    history = data.get('history', [])
    preselected = build_preselected(session_id, data)

    agent = ChatAgent(registry, code_collection, config)
    try:
        answer, trace = agent.answer(codebase_id, user_question, history=history, preselected=preselected)
    except Exception as e:
        logging.error(f"Chat agent failed: {e}")
        return jsonify({"error": f"Chat failed: {e}"}), 500
    return jsonify({"answer": answer, "trace": trace})


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    session_id = data.get('sessionId')
    if not session_id or session_id not in SCAN_SESSIONS:
        return jsonify({"error": "Invalid or expired session. Please scan a repository again."}), 400

    user_question = data.get('question')
    if not user_question:
        return jsonify({"error": "No user question found in the request."}), 400

    codebase_id = SCAN_SESSIONS[session_id].get('codebase_id')
    history = data.get('history', [])
    preselected = build_preselected(session_id, data)
    agent = ChatAgent(registry, code_collection, config)

    def generate():
        try:
            for event in agent.run(codebase_id, user_question, history=history, preselected=preselected):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logging.error(f"Chat stream failed: {e}")
            yield f"data: {json.dumps({'type': 'answer', 'answer': f'[Error: {e}]', 'trace': {}})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/search', methods=['POST'])
def search():
    if code_collection is None:
        return jsonify({"error": "Search is not available. Check MongoDB configuration."}), 500

    data = request.get_json()
    query = data.get('query')
    session_id = data.get('sessionId')
    if not query or not session_id:
        return jsonify({"error": "Search query and sessionId are required."}), 400
    if session_id not in SCAN_SESSIONS:
        return jsonify({"error": "Invalid or expired session. Please scan a repository again."}), 400

    codebase_id = SCAN_SESSIONS[session_id].get('codebase_id')
    if not codebase_id:
        return jsonify({"error": "Could not determine codebase for this session."}), 400

    em = registry.active_embedding()
    if not registry.embedding_available(em):
        return jsonify({"error": f"Embedding model '{em.id}' is unavailable."}), 503

    try:
        query_embedding = registry.embed([query], em, kind="query")[0]
        results = dbmod.hybrid_search(code_collection, codebase_id, query, query_embedding, em, limit=10)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Search error: {e}")
        return jsonify({"error": f"An error occurred during search: {e}"}), 500


# --- Pipeline routes ------------------------------------------------------
def _session_has_git(session):
    if session.get('local_path') and os.path.isdir(session['local_path']):
        return pipe.git_available(session['local_path'])
    return bool(session.get('git_url'))


@app.route('/pipelines/templates')
def pipelines_templates():
    session_id = request.args.get('sessionId')
    session = SCAN_SESSIONS.get(session_id, {})
    has_git = _session_has_git(session)
    return jsonify({
        "has_git": has_git,
        "templates": pipe.builtin_templates(has_git),
        "approaches": pipe.approach_catalog(has_git),
    })


@app.route('/pipelines/plan', methods=['POST'])
def pipelines_plan():
    data = request.get_json() or {}
    ask = (data.get('ask') or '').strip()
    approach = data.get('approach', 'thorough')
    model = (data.get('model') or '').strip() or None
    if not ask:
        return jsonify({"error": "An 'ask' is required."}), 400
    session = SCAN_SESSIONS.get(data.get('sessionId'), {})
    has_git = _session_has_git(session)
    try:
        spec = pipe.plan_pipeline(registry, ask, approach=approach, has_git=has_git, model=model)
        return jsonify({"spec": spec, "approaches": pipe.approach_catalog(has_git)})
    except Exception as e:
        logging.error(f"Pipeline planning failed: {e}")
        return jsonify({"error": f"Planning failed: {e}"}), 500


@app.route('/pipelines/run', methods=['POST'])
def pipelines_run():
    data = request.get_json() or {}
    session_id = data.get('sessionId')
    spec = data.get('spec')
    ask = data.get('ask', '')
    if not session_id or session_id not in SCAN_SESSIONS:
        return jsonify({"error": "Invalid or expired session. Please scan a repository again."}), 400
    if not spec or not spec.get('steps'):
        return jsonify({"error": "A pipeline spec with at least one step is required."}), 400

    session = SCAN_SESSIONS[session_id]
    codebase_id = session.get('codebase_id')
    repo_path, has_git, cleanup = resolve_repo_for_session(session)

    runner = pipe.PipelineRunner(
        registry, code_collection, codebase_id,
        repo_path=repo_path, has_git=has_git,
        runs_collection=pipeline_runs_collection)

    def generate():
        try:
            for event in runner.run(spec, ask=ask):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:
            logging.error(f"Pipeline run failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            if cleanup:
                cleanup()

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/pipelines/run_step', methods=['POST'])
def pipelines_run_step():
    data = request.get_json() or {}
    session_id = data.get('sessionId')
    spec = data.get('spec')
    step_id = data.get('stepId')
    ask = data.get('ask', '')
    run_id = data.get('runId')
    prior_trail = data.get('trail')
    if not session_id or session_id not in SCAN_SESSIONS:
        return jsonify({"error": "Invalid or expired session. Please scan a repository again."}), 400
    if not spec or not spec.get('steps'):
        return jsonify({"error": "A pipeline spec with at least one step is required."}), 400
    if not step_id:
        return jsonify({"error": "A 'stepId' is required."}), 400

    session = SCAN_SESSIONS[session_id]
    codebase_id = session.get('codebase_id')
    target = next((s for s in spec.get('steps', []) if s.get('id') == step_id), None)
    # Only pay for repo resolution (possible clone) when the step needs git.
    if target and target.get('type') == 'git':
        repo_path, has_git, cleanup = resolve_repo_for_session(session)
    else:
        repo_path, has_git, cleanup = None, False, None

    runner = pipe.PipelineRunner(
        registry, code_collection, codebase_id,
        repo_path=repo_path, has_git=has_git,
        runs_collection=pipeline_runs_collection)

    def generate():
        try:
            for event in runner.run_step(spec, step_id, ask=ask, run_id=run_id,
                                         prior_trail=prior_trail):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:
            logging.error(f"Pipeline step re-run failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            if cleanup:
                cleanup()

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/pipelines/runs')
def pipelines_runs():
    if pipeline_runs_collection is None:
        return jsonify({"runs": []})
    session = SCAN_SESSIONS.get(request.args.get('sessionId'), {})
    codebase_id = session.get('codebase_id')
    query = {"codebase_id": codebase_id} if codebase_id else {}
    runs = list(pipeline_runs_collection.find(
        query, {"trail": 0, "spec": 0}).sort("created_at", -1).limit(25))
    return jsonify({"runs": runs})


@app.route('/pipelines/runs/<run_id>')
def pipelines_run_detail(run_id):
    if pipeline_runs_collection is None:
        return jsonify({"error": "No run store available."}), 404
    run = pipeline_runs_collection.find_one({"_id": run_id})
    if not run:
        return jsonify({"error": "Run not found."}), 404
    return jsonify(run)


# --- Model / health routes ------------------------------------------------
@app.route('/health')
def health():
    data = registry.health()
    data["mongo"] = {"up": False, "indexes": []}
    if code_collection is not None:
        try:
            mongo_client.admin.command('ping')
            data["mongo"]["up"] = True
            data["mongo"]["indexes"] = dbmod.index_states(code_collection)
        except Exception as e:
            data["mongo"]["error"] = str(e)
    return jsonify(data)


@app.route('/models')
def models():
    em_active = registry.active_embedding()
    embedding_models = [m.to_dict(available=registry.embedding_available(m))
                        for m in registry.embedding_catalog()]
    return jsonify({
        "embedding_models": embedding_models,
        "chat_models": registry.chat_model_catalog(),
        "active": {
            "embedding": em_active.id,
            "llm": registry.active_llm,
            "fast_llm": registry.active_fast_llm,
        },
    })


@app.route('/models/select', methods=['POST'])
def models_select():
    data = request.get_json() or {}
    session_id = data.get('sessionId')
    changed_embedding = False

    if data.get('llm'):
        registry.active_llm = data['llm']
    if data.get('fast_llm'):
        registry.active_fast_llm = data['fast_llm']
    if data.get('embedding_id') and data['embedding_id'] != registry.active_embedding_id:
        em = registry.get_embedding(data['embedding_id'])
        if em is None:
            return jsonify({"error": f"Unknown embedding model: {data['embedding_id']}"}), 400
        if not registry.embedding_available(em):
            return jsonify({"error": f"Embedding model '{em.id}' is not available yet. Pull it first."}), 400
        registry.active_embedding_id = data['embedding_id']
        changed_embedding = True

    response = {"active": {
        "embedding": registry.active_embedding_id,
        "llm": registry.active_llm,
        "fast_llm": registry.active_fast_llm,
    }, "reembedding": False}

    # If the embedding model changed and we have a codebase, backfill its vectors.
    if changed_embedding and session_id and session_id in SCAN_SESSIONS:
        codebase_id = SCAN_SESSIONS[session_id].get('codebase_id')
        if codebase_id:
            em = registry.active_embedding()
            session = SCAN_SESSIONS[session_id]
            session['status'] = 'scanning'
            session['message'] = f"Re-embedding with {em.label}..."
            session['files_sent'] = True  # don't resend file tree
            session['indexing_status'] = {"running": True, "progress": 0, "total": 0,
                                          "message": "Starting re-embed..."}
            threading.Thread(target=backfill_embeddings, args=(session_id, codebase_id, em)).start()
            response["reembedding"] = True

    return jsonify(response)


@app.route('/models/pull', methods=['POST'])
def models_pull():
    data = request.get_json() or {}
    model_name = (data.get('model') or '').strip()
    if not model_name:
        return jsonify({"error": "A model name is required."}), 400
    # Accept "ollama/<name>" ids too.
    if model_name.startswith("ollama/"):
        model_name = model_name[len("ollama/"):]

    def generate():
        for progress in registry.pull_stream(model_name):
            yield f"data: {json.dumps(progress)}\n\n"
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# --- Startup --------------------------------------------------------------
def init_backend():
    global mongo_client, db, code_collection, codebase_state_collection, pipeline_runs_collection
    try:
        if not config.mdb_uri:
            raise ValueError("MDB_URI is not set.")
        mongo_client = MongoClient(config.mdb_uri, serverSelectionTimeoutMS=30000)
        mongo_client.admin.command('ping')
        logging.info("MongoDB client connected successfully.")
        db, code_collection, codebase_state_collection = dbmod.setup_database_and_collection(
            mongo_client, config.db_name, config.collection_name)
        if code_collection is not None:
            dbmod.ensure_text_index(code_collection)
            pipeline_runs_collection = db[f"{config.collection_name}_pipeline_runs"]
            try:
                pipeline_runs_collection.create_index("codebase_id")
                pipeline_runs_collection.create_index("created_at")
            except Exception:
                pass
            atexit.register(lambda: mongo_client.close())
        else:
            logging.error("Failed to set up MongoDB collections. Search will be disabled.")
    except (ConnectionFailure, ValueError, OperationFailure) as e:
        logging.error(f"Error initializing MongoDB client: {e}")
        mongo_client = db = code_collection = codebase_state_collection = pipeline_runs_collection = None

    # Resolve the active embedding model (auto-fallback to an available local one).
    try:
        active_id, auto = registry.resolve_startup_embedding()
        if auto:
            logging.warning(f"Configured embedding model unavailable; using '{active_id}' instead.")
        logging.info(f"Active embedding model: {registry.active_embedding_id}")
    except Exception as e:
        logging.warning(f"Could not resolve embedding model at startup: {e}")


init_backend()


if __name__ == '__main__':
    host = config.flask_host
    port = config.flask_port
    print("--- code-xplorer (local-first) ---")
    print(f"Embedding: {registry.active_embedding_id} | LLM: {registry.active_llm} | Fast: {registry.active_fast_llm}")
    print(f"Ollama: {config.ollama_host} | Voyage: {'on' if config.voyage_enabled else 'off'} | Azure: {'on' if config.azure_enabled else 'off'}")
    print(f"Starting server at http://{host}:{port}")
    print("Tip: `docker compose up -d` for MongoDB, and `ollama serve` for local models.")
    print("----------------------------------")
    app.run(host=host, port=port, debug=False, threaded=True)
