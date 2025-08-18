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

# --- EXPANDED: Action Tagging Definitions ---
ACTION_DEFINITIONS = {
    'data-storage': {
        'imports': {
            'sqlalchemy', 'django.db', 'pymongo', 'psycopg2', 'mysql.connector',
            'sqlite3', 'redis', 'cassandra', 'motor', 'asyncpg', 'peewee',
            'mongoose', 'sequelize', 'prisma', 'typeorm', 'knex', 'slonik', 'pg',
            'records', 'aiosqlite'
        },
        'regex': re.compile(
            r'\.Model|\.Base|\.Schema|DeclarativeBase|BaseModel|DataTypes\.|new Sequelize|PrismaClient|createConnection',
            re.IGNORECASE
        ),
        'base_classes': {'Model', 'Base', 'DeclarativeBase', 'BaseModel'}
    },
    'network-request': {
        'imports': {'requests', 'aiohttp', 'httpx', 'urllib3', 'fetch', 'axios', 'got', 'superagent', 'fastapi', 'flask'},
        'regex': re.compile(
            r'\.get\(|\.post\(|\.put\(|\.delete\(|fetch\(|axios\.|urllib3\.PoolManager|'
            r'app\.route\(|@app\.get\(|@app\.post\(|express\.Router\(|\.get\("/api',
            re.IGNORECASE
        )
    },
    'file-io': {
        'imports': {'os', 'shutil', 'pathlib', 'fs'},
        'regex': re.compile(
            r'open\(|readFile|writeFile|readFileSync|writeFileSync|pathlib\.|'
            r'os\.path|fs\.promises',
            re.IGNORECASE
        )
    }
}


# --- Global variables ---
SCANNED_FILES_STRUCTURE = {}
WHOLE_FILE_SUMMARY_CACHE = {}
mongo_client, db, code_collection = None, None, None
SCAN_SESSIONS = {}

# --- State for Background Indexing ---
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


def scan_directory(path, session_id):
    """Scans a directory and populates the session with file data, including action tags."""
    scanned_files_data = {}
    logging.info(f"[{session_id}] Starting directory scan at: {os.path.abspath(path)}")

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
                        
                        _, file_tags = parse_code_structure(relative_path, content)

                        scanned_files_data[relative_path] = {
                            "content": content,
                            "char_count": len(content),
                            "tags": file_tags
                        }
                except Exception as e:
                    logging.warning(f"[{session_id}] Could not read file {file_path}: {e}")

    logging.info(f"[{session_id}] Scan complete. Found {len(scanned_files_data)} relevant files.")
    SCAN_SESSIONS[session_id]['scanned_files_data'] = scanned_files_data


# --- NEW: Robust Code Parsing Engine ---

class PythonCodeParser(ast.NodeVisitor):
    """A more robust AST parser for Python."""
    def __init__(self):
        self.structure = []

    def visit_FunctionDef(self, node):
        self.structure.append({'name': node.name, 'type': 'function', 'start_line': node.lineno, 'end_line': node.end_lineno})
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.structure.append({'name': node.name, 'type': 'class', 'start_line': node.lineno, 'end_line': node.end_lineno})
        self.generic_visit(node)

def _parse_python_structure(content):
    """Parses Python code using AST."""
    try:
        tree = ast.parse(content)
        parser = PythonCodeParser()
        parser.visit(tree)
        return sorted(parser.structure, key=lambda x: x['start_line'])
    except SyntaxError:
        return [] # Return empty list if AST parsing fails

def _parse_js_ts_structure(content):
    """Parses JS/TS code using a comprehensive set of regex patterns."""
    structure = []
    # This single regex captures a wide array of function and class declarations
    pattern = re.compile(
        r'^(?:export\s+(?:default\s+)?|const|let|var)?'  # Optional keywords
        r'(?:async\s+)?'  # Optional async
        r'(?:function\s*\*?\s*([\w$]+)\s*\(|'  # Named function
        r'([\w$]+)\s*=\s*(?:async\s*)?\(|'  # Arrow function assigned to var
        r'class\s+([\w$]+)(?:\s+extends\s+[\w$.]+)?\s*\{|'  # Class declaration
        r'interface\s+([\w$]+)\s*\{|' # TypeScript Interface
        r'type\s+([\w$]+)\s*=\s*\{)' # TypeScript Type Alias
        r')', re.MULTILINE)

    for match in pattern.finditer(content):
        # Find the first non-None group to get the name
        name = next((g for g in match.groups() if g is not None), None)
        if not name:
            continue
            
        line_number = content.count('\n', 0, match.start()) + 1
        
        # Determine type based on the matched text
        full_match_text = match.group(0)
        item_type = 'function'
        if 'class' in full_match_text:
            item_type = 'class'
        elif 'interface' in full_match_text:
            item_type = 'interface'
        elif 'type' in full_match_text:
             item_type = 'interface' # Treat TS types like interfaces for UI purposes

        structure.append({'name': name, 'type': item_type, 'start_line': line_number})
        
    return sorted(structure, key=lambda x: x['start_line'])

def get_imports_from_content(content):
    """Extracts imported modules from file content using regex."""
    imports = set()
    # Handles import {..} from '..', import .. from '..', require('..')
    import_pattern = re.compile(r'import(?:.*?from\s*)?[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\)')
    matches = import_pattern.findall(content)
    for match in matches:
        # match is a tuple of groups, e.g., ('react', '') or ('', 'express')
        module_name = next((g for g in match if g), None)
        if module_name:
            # Handle relative paths vs library names
            if not module_name.startswith(('.', '/')):
                imports.add(module_name.split('/')[0]) # Get base package name
    return imports

def analyze_content_for_tags(content, imports):
    """Analyzes a block of code for action tags."""
    tags = set()
    for tag, definition in ACTION_DEFINITIONS.items():
        if imports.intersection(definition['imports']):
            tags.add(tag)
        if definition['regex'].search(content):
            tags.add(tag)
    return sorted(list(tags))


def parse_code_structure(file_path, content):
    """
    Main dispatcher for parsing. Selects parser based on file extension, then assigns tags.
    """
    ext = os.path.splitext(file_path)[1].lower()
    structure = []

    if ext == '.py':
        structure = _parse_python_structure(content)
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        structure = _parse_js_ts_structure(content)

    all_imports = get_imports_from_content(content)
    file_level_tags = set()

    for item in structure:
        item_code = extract_code_block(content, item['start_line'], item.get('end_line'), ext)
        item_tags = analyze_content_for_tags(item_code, all_imports)
        item['tags'] = item_tags
        file_level_tags.update(item_tags)

    if not file_level_tags: # If no structure, analyze whole file
        file_level_tags.update(analyze_content_for_tags(content, all_imports))

    return structure, sorted(list(file_level_tags))


def extract_code_block(content, start_line, end_line=None, file_ext='.js'):
    """Extracts a full code block using end_line from AST or brace/indentation counting."""
    lines = content.splitlines()
    start_line_idx = start_line - 1

    if start_line_idx >= len(lines):
        return ""

    if end_line is not None and end_line <= len(lines):
        return '\n'.join(lines[start_line_idx : end_line])

    # Fallback logic if end_line is not provided
    if file_ext == '.py':
        initial_indent = len(lines[start_line_idx]) - len(lines[start_line_idx].lstrip(' '))
        block_lines = [lines[start_line_idx]]
        for i in range(start_line_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                block_lines.append(line)
                continue
            current_indent = len(line) - len(line.lstrip(' '))
            if current_indent > initial_indent:
                block_lines.append(line)
            else:
                break
        return '\n'.join(block_lines)
    else: # Brace-based languages
        block_lines = []
        brace_count = 0
        in_block = False
        for i in range(start_line_idx, len(lines)):
            line = lines[i]
            block_lines.append(line)
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count > 0:
                in_block = True
            if in_block and brace_count <= 0:
                break
            # Handle single-line arrow functions without braces
            if i == start_line_idx and '=>' in line and not line.strip().endswith('{'):
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
                        "dynamic": True,
                        "fields": { "tags": {"type": "string", "analyzer": "lucene.keyword"}}}
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
                           "session_id": {"type": "token"},
                           "tags": {"type": "token"}
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
    session_id = snippet_data['session_id']
    with indexing_lock:
        SCAN_SESSIONS[session_id]['indexing_status']["progress"] += 1

    file_path = snippet_data['file_path']
    item_name = snippet_data['item_name']
    code_to_process = snippet_data['code']
    tags = snippet_data.get('tags', [])
    unique_id = f"{session_id}-{file_path}::{item_name}"

    try:
        if not code_to_process.strip():
            return "Skipped empty snippet"

        # 1. Generate description
        indexing_prompt = f"Describe this code:\n\n```\n{code_to_process}\n```"
        indexing_messages = [{"role": "system", "content": CODE_INDEXING_SYSTEM_PROMPT}, {"role": "user", "content": indexing_prompt}]
        desc_response = get_llm_response(client, indexing_messages, DEPLOYMENT)
        description_for_index = desc_response['answer']
        if "[Error" in description_for_index:
              raise Exception(f"LLM error for description: {description_for_index}")

        # 2. Generate summary
        summary_prompt = f"Summarize this code:\n\n```\n{code_to_process}\n```"
        summary_messages = [{"role": "system", "content": CODE_SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": summary_prompt}]
        summary_response = get_llm_response(client, summary_messages, DEPLOYMENT)
        user_facing_summary = summary_response['answer']
        if "[Error" in user_facing_summary:
              raise Exception(f"LLM error for summary: {user_facing_summary}")

        # 3. Generate embedding
        embedding = get_voyage_embedding(code_to_process)

        # 4. Save document
        document = {
            "session_id": session_id, "file_path": file_path, "item_name": item_name,
            "code": code_to_process, "description": description_for_index,
            "summary": user_facing_summary, "embedding": embedding, "tags": tags
        }
        code_collection.update_one({"_id": unique_id}, {"$set": document}, upsert=True)
        return f"Successfully indexed"
    except Exception as e:
        logging.error(f"Failed to index snippet {unique_id}: {e}")
        return f"Failed to index"

def process_and_index_snippets(files_data, session_id):
    """
    Finds all code snippets in the scanned files and indexes them in parallel for a given session.
    """
    if code_collection is None or client is None or voyage_client is None:
        msg = "Cannot start indexing: one or more clients (Mongo, OpenAI, Voyage) are not configured."
        logging.error(f"[{session_id}] {msg}")
        SCAN_SESSIONS[session_id]['status'] = 'error'
        SCAN_SESSIONS[session_id]['error_message'] = msg
        return

    snippets_to_process = []
    for file_path, data in files_data.items():
        content = data["content"]
        structure, _ = parse_code_structure(file_path, content)
        file_ext = os.path.splitext(file_path)[1].lower()

        for item in structure:
            code_block = extract_code_block(content, item['start_line'], item.get('end_line'), file_ext)
            if not code_block.strip(): continue # Skip empty blocks
            
            snippets_to_process.append({
                "file_path": file_path, "item_name": item['name'], "code": code_block,
                "session_id": session_id, "tags": item.get('tags', [])
            })

    if not snippets_to_process:
        logging.warning(f"[{session_id}] No code snippets (functions/classes) found to index.")
        SCAN_SESSIONS[session_id]['indexing_status'] = {"running": False, "progress": 0, "total": 0, "message": "No snippets found"}
        return

    SCAN_SESSIONS[session_id]['indexing_status'] = {
        "running": True, "progress": 0, "total": len(snippets_to_process), "message": "Indexing in progress..."
    }
    logging.info(f"[{session_id}] Found {len(snippets_to_process)} snippets. Starting parallel indexing...")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_index_one_snippet, snippet) for snippet in snippets_to_process]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logging.error(f'[{session_id}] A task generated an exception: {exc}')

        logging.info(f"[{session_id}] Background indexing complete.")
        SCAN_SESSIONS[session_id]['indexing_status']["message"] = "Indexing complete"
    except Exception as e:
        logging.error(f"[{session_id}] An error occurred during parallel indexing: {e}")
        SCAN_SESSIONS[session_id]['indexing_status']["message"] = f"Error during indexing: {e}"
    finally:
        SCAN_SESSIONS[session_id]['indexing_status']["running"] = False


def background_scan_and_index(session_id, path_or_url):
    """The main background worker function."""
    scan_path = None
    is_temp_dir = False
    session = SCAN_SESSIONS.get(session_id)
    if not session:
        logging.error(f"Session {session_id} not found for background worker.")
        return

    try:
        repo_info = parse_github_input(path_or_url)
        if repo_info:
            session['status'] = 'cloning'
            session['message'] = f"Cloning {repo_info['name']}..."
            scan_path = clone_repo_to_tempdir(repo_info["url"])
            session['display_name'] = repo_info["name"]
            is_temp_dir = True
        elif os.path.isdir(path_or_url):
            scan_path = path_or_url
            session['display_name'] = os.path.basename(os.path.abspath(path_or_url))
        else:
            raise ValueError(f"Input '{path_or_url}' is not a valid directory or a recognized GitHub repository format.")

        session['status'] = 'scanning'
        session['message'] = 'Scanning files...'
        scan_directory(scan_path, session_id)

        if not session.get('scanned_files_data'):
             raise FileNotFoundError("Scan complete, but no relevant code files were found.")

        session['status'] = 'indexing'
        session['message'] = 'Scan complete. Starting indexing...'
        session['scan_complete'] = True

        process_and_index_snippets(session['scanned_files_data'].copy(), session_id)

        session['status'] = 'complete'
        session['message'] = 'All processes complete.'
        logging.info(f"[{session_id}] background worker finished successfully.")

    except Exception as e:
        logging.error(f"[{session_id}] Error in background worker: {e}")
        session['status'] = 'error'
        session['error_message'] = str(e)
    finally:
        if is_temp_dir and scan_path and os.path.isdir(scan_path):
            logging.info(f"[{session_id}] Cleaning up temporary directory: {scan_path}")
            shutil.rmtree(scan_path)


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan_status/<session_id>')
def get_scan_status(session_id):
    """Returns the current status of a scan and indexing job."""
    session = SCAN_SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    response_data = {
        "status": session.get('status'),
        "message": session.get('message'),
        "error": session.get('error_message'),
        "displayName": session.get('display_name'),
        "indexingStatus": session.get('indexing_status', {})
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
    """Handles scanning and starts background indexing."""
    data = request.get_json()
    path_or_url = data.get('path', '.').strip() or '.'

    session_id = str(uuid.uuid4())
    SCAN_SESSIONS[session_id] = {
        "status": "starting",
        "message": "Initializing...",
        "scan_complete": False,
        "files_sent": False,
        "scanned_files_data": {},
        "indexing_status": {},
        "error_message": None,
        "display_name": ""
    }

    if len(SCAN_SESSIONS) > 10:
        oldest_session = next(iter(SCAN_SESSIONS))
        del SCAN_SESSIONS[oldest_session]

    thread = threading.Thread(target=background_scan_and_index, args=(session_id, path_or_url))
    thread.start()

    return jsonify({"sessionId": session_id}), 202

@app.route('/structure', methods=['POST'])
def get_structure():
    """Parses a file and returns its structure (functions, classes). Caches results."""
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
        structure, _ = parse_code_structure(file_path, content)
        SCANNED_FILES_STRUCTURE[cache_key] = structure
        return jsonify(structure)

    return jsonify({"error": "File not found in scanned data for this session."}), 404

@app.route('/extract_snippet', methods=['POST'])
def extract_snippet():
    """Extracts a single code snippet (function/class) from a file."""
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
    
    cache_key = f"{session_id}::{file_path}"
    structure = SCANNED_FILES_STRUCTURE.get(cache_key)
    if not structure:
        structure, _ = parse_code_structure(file_path, content)
        SCANNED_FILES_STRUCTURE[cache_key] = structure

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
    
    session = SCAN_SESSIONS.get(session_id)
    if not session or 'scanned_files_data' not in session:
        return jsonify({"error": "Session data not found."}), 404
    
    scanned_files_data = session['scanned_files_data']
    if file_path not in scanned_files_data:
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
            return jsonify({"answer": "Summary not found in the index. It may still be processing. Please try again shortly."})

    else:
        cache_key = f"{session_id}::{file_path}"
        if cache_key in WHOLE_FILE_SUMMARY_CACHE:
            return jsonify({"answer": WHOLE_FILE_SUMMARY_CACHE[cache_key]})

        content = scanned_files_data[file_path]["content"]
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
    
    data = request.get_json()
    session_id = data.get('sessionId')
    if not session_id or session_id not in SCAN_SESSIONS:
         return jsonify({"error": "Invalid or expired session. Please scan a repository again."}), 400

    scanned_files_data = SCAN_SESSIONS[session_id]['scanned_files_data']
    if not scanned_files_data:
        return jsonify({"error": "Please scan a directory or repository first."}), 400

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
        if file_path not in scanned_files_data:
            continue

        content = scanned_files_data[file_path]["content"]
        if "__all__" in items:
            code_context_parts.append(f"--- FILE: {file_path} ---\n\n{content}\n\n")
        else:
            cache_key = f"{session_id}::{file_path}"
            structure = SCANNED_FILES_STRUCTURE.get(cache_key)
            if not structure:
                structure, _ = parse_code_structure(file_path, content)
                SCANNED_FILES_STRUCTURE[cache_key] = structure
            
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
                                        "filter": { "session_id": { "$eq": session_id } }
                                    }
                                }
                            ],
                            "fullTextPipeline": [
                                {
                                    "$search": {
                                        "index": "default_text_index",
                                        "compound": {
                                            "must": [{"text": {"query": query, "path": ["item_name", "description", "summary", "code"]}}],
                                            "filter": [{"term": {"path": "session_id", "query": session_id}}]
                                        }
                                    }
                                },
                                { "$limit": 10 }
                            ]
                        }
                    },
                    "combination": { "weights": { "vectorPipeline": 0.7, "fullTextPipeline": 0.3 }},
                    "scoreDetails": True
                }
            },
            {
                "$project": {
                    "_id": 0, "file_path": 1, "item_name": 1,
                    "description": 1, "summary": 1, "code": 1,
                    "tags": 1, # Return the tags
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