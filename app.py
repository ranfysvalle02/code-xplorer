#!/usr/bin/env python3
import os
import sys
import logging
import json
import re
import subprocess
import tempfile
import shutil
import ast
import atexit
import threading
import concurrent.futures
import uuid
from dotenv import load_dotenv
from openai import AzureOpenAI
from flask import Flask, request, jsonify, render_template
import voyageai
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# --- Initialization & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
IGNORE_DIRS = {'.git', '.github', '__pycache__', 'node_modules', 'venv', '.vscode', 'build', 'dist', 'docs', '.idea'}
IGNORE_EXTS = {'.pyc', '.log', '.env', '.DS_Store', '.tmp', '.swo', '.swp'}
BINARY_EXTS = {'.png', '.jpg', '.jpeg', '.gif', 'ico', '.zip', '.gz', '.pdf', '.exe', '.dll', '.so', '.webp', '.svg'}

# --- Global variables ---
SCANNED_FILES_DATA = {}
SCANNED_FILES_STRUCTURE = {}
WHOLE_FILE_SUMMARY_CACHE = {}  # Session-based cache for whole-file summaries
SCAN_BASE_PATH = None
mongo_client, db, code_collection = None, None, None  # For MongoDB state

# --- State for Background Indexing ---
INDEXING_STATUS = {"running": False, "progress": 0, "total": 0, "message": "Not started"}
indexing_lock = threading.Lock()

# --- Azure OpenAI Client Configuration ---
try:
    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        timeout=30.0,
    )
    DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), DEPLOYMENT]):
        raise ValueError("One or more required Azure environment variables are missing.")
    logging.info(f"Azure OpenAI client configured successfully for deployment: {DEPLOYMENT}")
except Exception as e:
    logging.error(f"Error initializing Azure OpenAI client: {e}")
    client = None

# --- Voyage AI Client Configuration ---
try:
    voyage_client = voyageai.Client(
        api_key=os.getenv("VOYAGE_API_KEY"),
        timeout=30.0,
    )
    VOYAGE_EMBEDDING_MODEL = "voyage-code-2"
    VOYAGE_EMBEDDING_DIMENSIONS = 1536
    logging.info(f"Voyage AI client configured successfully for model: {VOYAGE_EMBEDDING_MODEL}")
except Exception as e:
    logging.error(f"Error initializing Voyage AI client: {e}. Please set VOYAGE_API_KEY.")
    voyage_client = None

# --- MongoDB Setup Functions ---
def setup_database_and_collection(client, db_name, coll_name):
    """Gets the database and creates the collection if it doesn't exist."""
    try:
        db = client[db_name]
        if coll_name not in db.list_collection_names():
            logging.warning(f"Collection '{coll_name}' not found. Creating it now...")
            db.create_collection(coll_name)
            logging.info(f"Successfully created collection '{coll_name}' in database '{db_name}'.")
        else:
            logging.info(f"Successfully connected to existing collection '{coll_name}'.")
        collection = db[coll_name]
        return db, collection
    except OperationFailure as e:
        logging.error(f"MongoDB operation failed during setup: {e.details}")
        logging.error("Please ensure the user has the correct permissions (e.g., readWrite, dbAdmin).")
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during database setup: {e}")
        return None, None

# --- Prompt Templates ---
CODE_ANALYSIS_SYSTEM_PROMPT = """You are an expert AI code analyst. Your task is to answer questions about a codebase.

**Core Directives:**
- **Strictly Grounded:** Base your entire response *only* on the code snippets, summaries, and conversation history provided.
- **Use Summaries:** You may be provided with AI-generated summaries of code. Use these for high-level understanding but always prioritize the raw code for specific details when available.
- **Cite Sources:** When referencing code, always mention the file path and the specific function or class name.
- **Acknowledge Limits:** If the answer isn't in the context, state that clearly. Do not infer, guess, or use external knowledge.
- **Be a Chatbot:** Use the conversation history to understand the context of follow-up questions."""

CODE_SUMMARY_SYSTEM_PROMPT = """You are an expert code analyst. Your task is to provide a concise, one-paragraph summary of the provided code snippet.
Focus on its primary purpose, inputs, and what it returns or its main side effect. Do not describe the implementation details line-by-line. Start the summary directly, without any preamble."""

CODE_INDEXING_SYSTEM_PROMPT = """You are a search indexing expert. Your task is to create a concise, one-sentence description of the provided code snippet's primary function.
Focus on extracting key nouns, verbs, and technical terms that would be useful for a search query. This description is for an internal search index, not for human display.
Example: 'A Flask route that handles user authentication via POST requests using JWT for authorization.'
Start the description directly, without any preamble."""


# --- Core Backend Functions ---

def get_llm_response(client, messages, model_deployment, is_json=False):
    """
    Calls the chat completions API.
    """
    if not client:
        return {"answer": "[Error: OpenAI client not configured]"}
    try:
        response_format = {"type": "json_object"} if is_json else None
        response = client.chat.completions.create(
            model=model_deployment,
            messages=messages,
            response_format=response_format
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in get_llm_response: {e}")
        return {"answer": f"[Error calling LLM: {e}]"}


def parse_github_input(input_string):
    """
    Parses various GitHub URL/string formats and returns a dict with a clone URL and repo name.
    """
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
    """Clones a public GitHub repository to a temporary directory."""
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
        logging.error("'git' command not found. Please ensure Git is installed and in your system PATH.")
        shutil.rmtree(temp_dir)
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone repository: {e.stderr}")
        shutil.rmtree(temp_dir)
        raise


def scan_directory(path='.'):
    """Scans a directory and returns a dictionary of relative_path: {content, char_count}."""
    global SCANNED_FILES_DATA, SCANNED_FILES_STRUCTURE, WHOLE_FILE_SUMMARY_CACHE
    SCANNED_FILES_DATA, SCANNED_FILES_STRUCTURE, WHOLE_FILE_SUMMARY_CACHE = {}, {}, {}
    logging.info(f"Starting directory scan at: {os.path.abspath(path)}")

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext not in IGNORE_EXTS and ext not in BINARY_EXTS:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        relative_path = os.path.relpath(file_path, path)
                        SCANNED_FILES_DATA[relative_path] = {
                            "content": content,
                            "char_count": len(content)
                        }
                except Exception as e:
                    logging.warning(f"Could not read file {file_path}: {e}")

    logging.info(f"Scan complete. Found {len(SCANNED_FILES_DATA)} relevant files.")
    return SCANNED_FILES_DATA

class PythonCodeParser(ast.NodeVisitor):
    def __init__(self):
        self.structure = []

    def visit_FunctionDef(self, node):
        self.structure.append({
            'name': node.name,
            'type': 'function',
            'start_line': node.lineno - 1,
            'end_line': node.end_lineno -1
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.structure.append({
            'name': node.name,
            'type': 'class',
            'start_line': node.lineno - 1,
            'end_line': node.end_lineno - 1
        })
        self.generic_visit(node)

def parse_code_structure(file_path, content):
    """Parses code content to find high-level structures like classes and functions."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.py':
        try:
            tree = ast.parse(content)
            parser = PythonCodeParser()
            parser.visit(tree)
            return sorted(parser.structure, key=lambda x: x['start_line'])
        except SyntaxError as e:
            logging.warning(f"AST parsing failed for {file_path}: {e}. Falling back to regex.")
            pass

    structure = []
    lines = content.splitlines()

    patterns = {
        '.py': [
            ('class', re.compile(r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)')),
            ('function', re.compile(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('))
        ],
        '.js': [
            ('class', re.compile(r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)')),
            ('function', re.compile(r'^\s*(?:async\s+)?function\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')),
            ('function', re.compile(r'^\s*(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>')),
            ('function', re.compile(r'^\s*export\s+(?:async\s+)?function\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\('))
        ],
    }
    lang_patterns = patterns.get(ext, [])
    if not lang_patterns:
        return []

    for i, line in enumerate(lines):
        for type, pattern in lang_patterns:
            match = pattern.match(line)
            if match:
                structure.append({'name': match.group(1), 'type': type, 'start_line': i})
                break
    return structure


def extract_code_block(content, start_line, end_line=None, file_ext='.js'):
    """Extracts a full code block (function/class) based on AST end_line or indentation/braces."""
    lines = content.splitlines()

    if end_line is not None and end_line < len(lines):
        return '\n'.join(lines[start_line : end_line + 1])

    block_lines = []
    if file_ext == '.py':
        if start_line >= len(lines): return ""
        initial_indent_str = re.match(r'^(\s*)', lines[start_line]).group(1)
        block_lines.append(lines[start_line])
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            current_indent_str = re.match(r'^(\s*)', line).group(1)
            if len(current_indent_str) > len(initial_indent_str):
                block_lines.append(line)
            else:
                break
    else:
        if start_line >= len(lines): return ""
        brace_count = 0
        in_block = False
        block_started = False
        for i in range(start_line, len(lines)):
            line = lines[i]
            if not block_started:
                block_lines.append(line)

            if '{' in line:
                in_block = True
                block_started = True

            if not in_block:
                continue

            if i > start_line and block_started and line not in block_lines:
                 block_lines.append(line)

            brace_count += line.count('{')
            brace_count -= line.count('}')

            if brace_count <= 0 and in_block:
                break

    return '\n'.join(block_lines)


def get_voyage_embedding(text: str) -> list[float]:
    """Generate vector embedding for a text string using Voyage AI."""
    if not voyage_client:
        raise Exception("Voyage AI client is not configured.")
    return voyage_client.embed(texts=[text], model=VOYAGE_EMBEDDING_MODEL).embeddings[0]


def create_search_indexes():
    """Checks for and creates the required Atlas Search text and vector indexes."""
    if code_collection is None:
        logging.error("Cannot create search indexes: MongoDB collection not available.")
        return

    TEXT_INDEX_NAME = "default_text_index"
    VECTOR_INDEX_NAME = "default_vector_index"

    try:
        existing_indexes = [idx['name'] for idx in code_collection.list_search_indexes()]

        if TEXT_INDEX_NAME not in existing_indexes:
            logging.info(f"Creating Atlas Search text index: '{TEXT_INDEX_NAME}'...")
            text_index_model = {
                "name": TEXT_INDEX_NAME,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "item_name": {"type": "string", "analyzer": "lucene.standard"},
                            "description": {"type": "string", "analyzer": "lucene.standard"},
                            "summary": {"type": "string", "analyzer": "lucene.standard"},
                            "code": {"type": "string", "analyzer": "lucene.standard"},
                            "file_path": {"type": "stringKeyword"},
                            "session_id": {"type": "stringKeyword"}
                        }
                    }
                }
            }
            code_collection.create_search_index(model=text_index_model)
            logging.info("Text index creation initiated. It may take a few minutes to become ready.")
        else:
            logging.info(f"Text index '{TEXT_INDEX_NAME}' already exists.")

        if VECTOR_INDEX_NAME not in existing_indexes:
            logging.info(f"Creating Atlas Search vector index: '{VECTOR_INDEX_NAME}'...")
            vector_index_model = {
                "name": VECTOR_INDEX_NAME,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                           "embedding": {
                                "type": "knnVector",
                                "dimensions": VOYAGE_EMBEDDING_DIMENSIONS,
                                "similarity": "cosine"
                            },
                           "session_id": {
                                "type": "token"
                           }
                        }
                    }
                }
            }
            code_collection.create_search_index(model=vector_index_model)
            logging.info("Vector index creation initiated. It may take a few minutes to become ready.")
        else:
            logging.info(f"Vector index '{VECTOR_INDEX_NAME}' already exists.")

    except OperationFailure as e:
        logging.error(f"An error occurred with MongoDB operations: {e.details}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during index creation: {e}")


# --- Parallel Processing Functions ---

def _index_one_snippet(snippet_data):
    """Worker function to process and index a single code snippet with its summary."""
    global INDEXING_STATUS
    file_path = snippet_data['file_path']
    item_name = snippet_data['item_name']
    code_to_process = snippet_data['code']
    session_id = snippet_data['session_id']
    unique_id = f"{session_id}-{file_path}::{item_name}"

    try:
        if not code_to_process.strip():
            return "Skipped empty snippet"

        # 1. Generate description for search index
        indexing_prompt = f"Describe this code:\n\n```\n{code_to_process}\n```"
        indexing_messages = [
            {"role": "system", "content": CODE_INDEXING_SYSTEM_PROMPT},
            {"role": "user", "content": indexing_prompt}
        ]
        desc_response = get_llm_response(client, indexing_messages, DEPLOYMENT)
        description_for_index = desc_response['answer']
        if "[Error" in description_for_index:
              raise Exception(f"LLM error for description: {description_for_index}")

        # 2. Generate summary for user display
        summary_prompt = f"Summarize this code:\n\n```\n{code_to_process}\n```"
        summary_messages = [
            {"role": "system", "content": CODE_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": summary_prompt}
        ]
        summary_response = get_llm_response(client, summary_messages, DEPLOYMENT)
        user_facing_summary = summary_response['answer']
        if "[Error" in user_facing_summary:
              raise Exception(f"LLM error for summary: {user_facing_summary}")

        # 3. Generate vector embedding
        embedding = get_voyage_embedding(code_to_process)

        # 4. Prepare and save the document
        document = {
            "session_id": session_id,
            "file_path": file_path,
            "item_name": item_name,
            "code": code_to_process,
            "description": description_for_index,
            "summary": user_facing_summary,
            "embedding": embedding,
        }
        code_collection.update_one(
            {"_id": unique_id},
            {"$set": document},
            upsert=True
        )
        return f"Successfully indexed"
    except Exception as e:
        logging.error(f"Failed to index snippet {unique_id}: {e}")
        return f"Failed to index"
    finally:
        with indexing_lock:
            INDEXING_STATUS["progress"] += 1


def process_and_index_snippets(files_data, session_id):
    """
    Finds all code snippets in the scanned files and indexes them in parallel for a given session.
    """
    global INDEXING_STATUS
    if code_collection is None or client is None or voyage_client is None:
        logging.error("Cannot start indexing: one or more clients (Mongo, OpenAI, Voyage) are not configured.")
        return

    snippets_to_process = []
    for file_path, data in files_data.items():
        content = data["content"]
        structure = parse_code_structure(file_path, content)
        file_ext = os.path.splitext(file_path)[1].lower()

        for item in structure:
            code_block = extract_code_block(content, item['start_line'], item.get('end_line'), file_ext)
            snippets_to_process.append({
                "file_path": file_path,
                "item_name": item['name'],
                "code": code_block,
                "session_id": session_id
            })

    if not snippets_to_process:
        logging.warning("No code snippets (functions/classes) found to index.")
        INDEXING_STATUS = {"running": False, "progress": 0, "total": 0, "message": "No snippets found"}
        return

    INDEXING_STATUS = {
        "running": True,
        "progress": 0,
        "total": len(snippets_to_process),
        "message": "Indexing in progress..."
    }
    logging.info(f"[{session_id}] Found {INDEXING_STATUS['total']} snippets. Starting parallel indexing...")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_index_one_snippet, snippet) for snippet in snippets_to_process]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logging.error(f'A task generated an exception: {exc}')

        logging.info(f"[{session_id}] Background indexing complete.")
        INDEXING_STATUS["message"] = "Indexing complete"
    except Exception as e:
        logging.error(f"[{session_id}] An error occurred during parallel indexing: {e}")
        INDEXING_STATUS["message"] = f"Error during indexing: {e}"
    finally:
        INDEXING_STATUS["running"] = False


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    """Returns the current status of the background indexing job."""
    return jsonify(INDEXING_STATUS)


@app.route('/scan', methods=['POST'])
def scan():
    """Handles scanning and starts background indexing."""
    global SCAN_BASE_PATH, INDEXING_STATUS
    data = request.get_json()
    path_or_url = data.get('path', '.').strip() or '.'
    scan_path, display_name = None, ""

    try:
        session_id = str(uuid.uuid4())
        INDEXING_STATUS = {"running": False, "progress": 0, "total": 0, "message": "Starting scan..."}

        repo_info = parse_github_input(path_or_url)
        if repo_info:
            scan_path = clone_repo_to_tempdir(repo_info["url"])
            display_name = repo_info["name"]
        elif os.path.isdir(path_or_url):
            scan_path = path_or_url
            display_name = os.path.basename(os.path.abspath(path_or_url))
        else:
            return jsonify({"error": f"Input '{path_or_url}' is not a valid directory or a recognized GitHub repository format."}), 400

        files_found_data = scan_directory(scan_path)
        SCAN_BASE_PATH = scan_path

        if not files_found_data:
            return jsonify({"error": "Scan complete, but no relevant code files were found."}), 404

        indexing_thread = threading.Thread(target=process_and_index_snippets, args=(files_found_data.copy(), session_id))
        indexing_thread.start()

        files_for_frontend = [
            {"path": path, "char_count": data["char_count"], "content": data["content"]}
            for path, data in files_found_data.items()
        ]
        files_for_frontend.sort(key=lambda x: x['path'])

        return jsonify({
            "sessionId": session_id,
            "displayName": display_name,
            "files": files_for_frontend,
        })
    except Exception as e:
        logging.error(f"An unexpected error occurred during scanning: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/structure', methods=['POST'])
def get_structure():
    """Parses a file and returns its structure (functions, classes). Caches results."""
    data = request.get_json()
    file_path = data.get('path')
    if not file_path:
        return jsonify({"error": "File path is required."}), 400

    if file_path in SCANNED_FILES_STRUCTURE:
        return jsonify(SCANNED_FILES_STRUCTURE[file_path])
    if file_path in SCANNED_FILES_DATA:
        content = SCANNED_FILES_DATA[file_path]["content"]
        structure = parse_code_structure(file_path, content)
        SCANNED_FILES_STRUCTURE[file_path] = structure
        return jsonify(structure)

    return jsonify({"error": "File not found in scanned data."}), 404

@app.route('/extract_snippet', methods=['POST'])
def extract_snippet():
    """Extracts a single code snippet (function/class) from a file."""
    data = request.get_json()
    file_path = data.get('path')
    item_name = data.get('name')

    if not file_path or not item_name:
        return jsonify({"error": "File path and item name are required."}), 400
    if file_path not in SCANNED_FILES_DATA:
        return jsonify({"error": "File not found in scanned data."}), 404

    content = SCANNED_FILES_DATA[file_path]["content"]
    structure = SCANNED_FILES_STRUCTURE.get(file_path) or parse_code_structure(file_path, content)
    if file_path not in SCANNED_FILES_STRUCTURE:
        SCANNED_FILES_STRUCTURE[file_path] = structure

    item_data = next((item for item in structure if item['name'] == item_name), None)

    if item_data:
        file_ext = os.path.splitext(file_path)[1].lower()
        block_content = extract_code_block(content, item_data['start_line'], item_data.get('end_line'), file_ext)
        return jsonify({"content": block_content})
    else:
        return jsonify({"error": f"Item '{item_name}' not found in '{file_path}'"}), 404


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Gets a summary. For snippets, it fetches from MongoDB.
    For whole files, it generates on-demand and caches in memory.
    """
    if not client:
        return jsonify({"error": "Azure OpenAI client not configured."}), 500

    data = request.get_json()
    session_id = data.get('sessionId')
    file_path = data.get('path')
    item_name = data.get('name')

    if not all([session_id, file_path]):
        return jsonify({"error": "Session ID and file path are required."}), 400
    if file_path not in SCANNED_FILES_DATA:
        return jsonify({"error": "File not found in scanned data."}), 404

    if item_name:
        if code_collection is None:
            return jsonify({"error": "Database not available to fetch snippet summary."}), 503

        document = code_collection.find_one({
            "session_id": session_id,
            "file_path": file_path,
            "item_name": item_name
        })
        if document and "summary" in document:
            return jsonify({"answer": document["summary"]})
        else:
            return jsonify({"error": f"Summary for '{item_name}' not found. It may still be indexing."}), 404

    else:
        cache_key = f"{session_id}::{file_path}"
        if cache_key in WHOLE_FILE_SUMMARY_CACHE:
            return jsonify({"answer": WHOLE_FILE_SUMMARY_CACHE[cache_key]})

        content = SCANNED_FILES_DATA[file_path]["content"]
        context_name = f"the file `{file_path}`"
        summary_prompt = f"Please summarize {context_name}:\n\n```\n{content}\n```"
        summary_messages = [
            {"role": "system", "content": CODE_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": summary_prompt}
        ]
        ai_response = get_llm_response(client, summary_messages, DEPLOYMENT)

        if "Error" not in ai_response["answer"]:
             WHOLE_FILE_SUMMARY_CACHE[cache_key] = ai_response["answer"]

        return jsonify(ai_response)


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat, builds context, and gets a response from the LLM."""
    if not client:
        return jsonify({"error": "Azure OpenAI client is not configured. Check server .env file."}), 500
    if not SCANNED_FILES_DATA:
        return jsonify({"error": "Please scan a directory or repository first."}), 400

    data = request.get_json()
    user_question = data.get('question')
    selected_items = data.get('selected_items', {})
    selected_summaries = data.get('selected_summaries', [])
    custom_snippets = data.get('custom_snippets', [])
    chat_history = data.get('history', [])

    if not user_question:
        return jsonify({"error": "No user question found in the request."}), 400
    if not selected_items and not selected_summaries and not custom_snippets:
        return jsonify({"error": "No files, snippets, summaries, or custom snippets were selected for context."}), 400

    summary_context_parts = []
    if selected_summaries:
        summary_context_parts.append("## CONTEXT FROM AI-GENERATED SUMMARIES\n")
        for summary in selected_summaries:
            item_desc = f"file `{summary['path']}`"
            if summary.get('name'):
                item_desc = f"item `{summary['name']}` in file `{summary['path']}`"
            summary_context_parts.append(f"- **Summary for {item_desc}:** {summary['content']}\n")

    code_context_parts = []

    if custom_snippets:
        for snippet in custom_snippets:
            code_context_parts.append(
                f"--- CUSTOM SNIPPET FROM: {snippet['source']} ---\n\n{snippet['content']}\n\n"
            )

    for file_path, items in selected_items.items():
        if file_path not in SCANNED_FILES_DATA:
            continue

        content = SCANNED_FILES_DATA[file_path]["content"]
        if "__all__" in items:
            code_context_parts.append(f"--- FILE: {file_path} ---\n\n{content}\n\n")
        else:
            structure = SCANNED_FILES_STRUCTURE.get(file_path) or parse_code_structure(file_path, content)
            SCANNED_FILES_STRUCTURE[file_path] = structure
            item_map = {s['name']: s for s in structure}
            file_ext = os.path.splitext(file_path)[1].lower()
            for item_name in items:
                if item_name in item_map:
                    item_data = item_map[item_name]
                    block_content = extract_code_block(content, item_data['start_line'], item_data.get('end_line'), file_ext)
                    code_context_parts.append(
                        f"--- SNIPPET FROM: {file_path} ({item_data['type']}: {item_name}) ---\n\n{block_content}\n\n"
                    )

    if not code_context_parts and not summary_context_parts:
        return jsonify({"error": "The selected items could not be found or extracted."}), 400

    summary_context = "".join(summary_context_parts)
    code_context = "\n## CODEBASE CONTEXT\n\n" + "".join(code_context_parts) if code_context_parts else ""

    system_prompt = f"{CODE_ANALYSIS_SYSTEM_PROMPT}\n\n{summary_context}{code_context}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_question})

    ai_response = get_llm_response(client, messages, DEPLOYMENT)
    return jsonify(ai_response)

@app.route('/search', methods=['POST'])
def search():
    """Performs hybrid search on the indexed code snippets, scoped to the session_id."""
    if not voyage_client or code_collection is None:
        return jsonify({"error": "Search is not available. Check Voyage/MongoDB configuration."}), 500

    data = request.get_json()
    query = data.get('query')
    session_id = data.get('sessionId')

    if not query or not session_id:
        return jsonify({"error": "Search query and sessionId are required."}), 400

    try:
        query_embedding = get_voyage_embedding(query)

        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vectorPipeline": [
                                {
                                    "$vectorSearch": {
                                        "index": "default_vector_index",
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": 100,
                                        "limit": 10,
                                        # FIX: $vectorSearch's filter uses standard MQL, not Atlas Search syntax.
                                        "filter": {
                                            "session_id": {
                                                "$eq": session_id
                                            }
                                        }
                                    }
                                }
                            ],
                            "fullTextPipeline": [
                                {
                                    "$search": {
                                        "index": "default_text_index",
                                        # The 'compound' operator correctly uses Atlas Search 'term' syntax for its filter.
                                        "compound": {
                                            "must": [{
                                                "text": {
                                                    "query": query,
                                                    "path": ["item_name", "description", "summary", "code"]
                                                }
                                            }],
                                            "filter": [{
                                                "term": {
                                                    "path": "session_id",
                                                    "query": session_id
                                                }
                                            }]
                                        }
                                    }
                                },
                                { "$limit": 10 }
                            ]
                        }
                    },
                    "combination": {
                        "weights": {
                            "vectorPipeline": 0.7,
                            "fullTextPipeline": 0.3
                        }
                    },
                    "scoreDetails": True
                }
            },
            {
                "$project": {
                    "_id": 0, "file_path": 1, "item_name": 1,
                    "description": 1, "summary": 1, "code": 1,
                    "scoreDetails": { "$meta": "scoreDetails" }
                }
            },
            { "$limit": 10 }
        ]

        results = list(code_collection.aggregate(pipeline))
        return jsonify(results)

    except OperationFailure as e:
        logging.error(f"An error occurred during search: {e.details}")
        if "Unrecognized pipeline stage name: '$rankFusion'" in str(e.details):
             return jsonify({"error": "Search functionality requires MongoDB 8.1 or higher. Please upgrade your cluster."}), 500
        return jsonify({"error": f"An error occurred during search: {e.details}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during search: {e}")
        return jsonify({"error": f"An unexpected error occurred during search: {e}"}), 500

if __name__ == '__main__':
    try:
        MONGO_URI = os.getenv("MDB_URI")
        DB_NAME = os.getenv("DB_NAME", "code_search_db")
        COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_snippets")
        if not MONGO_URI:
            raise ValueError("MDB_URI environment variable is not set.")

        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=30000
        )
        mongo_client.admin.command('ping')
        logging.info("MongoDB client connected successfully.")

        db, code_collection = setup_database_and_collection(mongo_client, DB_NAME, COLLECTION_NAME)

        if code_collection is not None:
            create_search_indexes()
            atexit.register(lambda: mongo_client.close())
        else:
            logging.error("Failed to set up MongoDB collection. Search functionality will be disabled.")

    except (ConnectionFailure, ValueError) as e:
        logging.error(f"Error initializing MongoDB client: {e}")
        mongo_client, db, code_collection = None, None, None

    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    print("--- CodeWhisperer ---")
    print(f"🚀 Starting server at http://{host}:{port}")
    print("🔧 Make sure your .env file is configured correctly and Git is installed.")
    print("👉 Open the URL in your browser to start analyzing code!")
    print("--------------------------------")
    app.run(host=host, port=port, debug=False)