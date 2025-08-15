#!/usr/bin/env python3  
import os  
import sys  
import logging  
import json  
import re  
import subprocess  
import tempfile  
import shutil  
from dotenv import load_dotenv  
from openai import AzureOpenAI  
from flask import Flask, Response, request, jsonify, render_template_string  
  
# --- Initialization & Configuration ---  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
load_dotenv()  
  
# --- Flask App Initialization ---  
app = Flask(__name__)  
  
# --- Configuration ---  
IGNORE_DIRS = {'.git', '.github', '__pycache__', 'node_modules', 'venv', '.vscode', 'build', 'dist', 'docs', '.idea'}  
IGNORE_EXTS = {'.pyc', '.log', '.env', '.DS_Store', '.tmp', '.swo', '.swp'}  
BINARY_EXTS = {'.png', '.jpg', '.jpeg', '.gif', 'ico', '.zip', '.gz', '.pdf', '.exe', '.dll', '.so', '.webp', '.svg'}  
# MAX_CHAT_HISTORY is now managed on the client-side for user control.  
  
# --- Global variables (simple state management for this single-user tool) ---  
SCANNED_FILES_CONTENT = {}  
SCANNED_FILES_STRUCTURE = {}  
SCAN_BASE_PATH = None  
  
# --- Azure OpenAI Client Configuration ---  
try:  
    client = AzureOpenAI(  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    )  
    DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")  
    if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), DEPLOYMENT]):  
        raise ValueError("One or more required Azure environment variables are missing.")  
    logging.info(f"Azure OpenAI client configured successfully for deployment: {DEPLOYMENT}")  
except Exception as e:  
    logging.error(f"Error initializing Azure OpenAI client: {e}")  
    logging.error("Please ensure your .env file is correctly configured.")  
    client = None  # Ensure client is None if initialization fails  
  
# --- Prompt Templates ---  
# CODE_ANALYSIS_SYSTEM_PROMPT
CODE_ANALYSIS_SYSTEM_PROMPT = """You are an expert AI code analyst. Your task is to answer questions about a codebase using the provided context.

**Core Directives:**
- **Strictly Grounded:** Base your entire response *only* on the code snippets and conversation history provided.
- **Cite Sources:** When referencing code, always mention the file path and the specific function or class name.
- **Acknowledge Limits:** If the answer isn't in the context, state that clearly. Do not infer, guess, or use external knowledge.
- **Show Your Work:** Provide concise, step-by-step reasoning for your conclusions.
- **Be a Chatbot:** Use the conversation history to understand the context of follow-up questions."""

# FILE_SELECTION_SYSTEM_PROMPT
FILE_SELECTION_SYSTEM_PROMPT = """You are an AI software architect that identifies critical project files for a new developer.

**Instructions:**
1.  Analyze the provided list of file paths.
2.  Prioritize entry points (`main.py`, `app.js`), configurations (`package.json`, `requirements.txt`), and core logic (`controllers/`, `services/`, `utils/`).
3.  Select the top 5-7 most important files for understanding the project's purpose.
4.  Provide brief, step-by-step reasoning for each file selection.

**Output Format:**
You MUST output a single, valid JSON object. Do not add any other text or markdown fences.
{
  "reasoning": "Your step-by-step analysis and justification for your choices. Use markdown for formatting.",
  "files": [
    "path/to/important_file_1.js",
    "path/to/core_logic.py",
    "path/to/another/key_file.go"
  ]
}"""
  
# --- Core Backend Functions ---  
  
def get_reasoned_llm_response(client, prompt_text, model_deployment, effort="medium"):  
    """  
    Calls a specific reasoning-focused endpoint, expecting a structured response with summaries and a final answer.  
    """  
    if not client:  
        return {"answer": "[Error: OpenAI client not configured]", "summaries": []}  
    try:  
        response = client.responses.create(  
            input=prompt_text,  
            model=model_deployment,  
            reasoning={"effort": effort, "summary": "detailed"}
        )
        """
        Note:
        Even when enabled, reasoning summaries are not generated 
        for every step/request. This is expected behavior.
        """
        response_data = response.model_dump()

        result = {"answer": "Could not extract a final answer.", "summaries": []}
        output_blocks = response_data.get("output", [])
        if output_blocks:  
            summary_section = output_blocks[0].get("summary", [])  
            if summary_section:  
                result["summaries"] = [s.get("text") for s in summary_section if s.get("text")]  
  
            content_section_index = 1 if summary_section else 0  
  
            if len(output_blocks) > content_section_index and output_blocks[content_section_index].get("content"):  
                result["answer"] = output_blocks[content_section_index]["content"][0].get("text", result["answer"])  
  
            if result["answer"] == "Could not extract a final answer.":  
                for block in output_blocks:  
                    if block.get("content"):  
                        for content_item in block["content"]:  
                            if content_item.get("text"):  
                                result["answer"] = content_item["text"]  
                                break  
                    if result["answer"] != "Could not extract a final answer.":  
                        break  
  
        result["answer"] = result["answer"].strip()  
        return result  
    except Exception as e:  
        logging.error(f"Error in get_reasoned_llm_response: {e}")  
        return {"answer": f"[Error calling LLM: {e}]", "summaries": []}  
  
  
def is_github_url(url_string):  
    """Checks if a string is a valid public GitHub repository URL."""  
    github_pattern = re.compile(r'^(https?://)?(www\.)?github\.com/([\w\-.]+)/([\w\-.]+)(\.git)?/?$')  
    return bool(github_pattern.match(url_string))  
  
  
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
    """Scans a directory and returns a dictionary of relative_path: content."""  
    global SCANNED_FILES_CONTENT, SCANNED_FILES_STRUCTURE  
    SCANNED_FILES_CONTENT, SCANNED_FILES_STRUCTURE = {}, {}  
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
                        SCANNED_FILES_CONTENT[relative_path] = content  
                except Exception as e:  
                    logging.warning(f"Could not read file {file_path}: {e}")  
  
    logging.info(f"Scan complete. Found {len(SCANNED_FILES_CONTENT)} relevant files.")  
    return SCANNED_FILES_CONTENT  
  
  
def get_intelligent_selection(file_list):  
    """Uses AI to suggest the most important files from a list of paths."""  
    if not DEPLOYMENT or not client:  
        logging.warning("AI selection skipped: Azure client not configured.")  
        return {}  
  
    logging.info(f"Requesting AI to select key files from {len(file_list)} candidates.")  
    formatted_file_list = "\n".join(sorted(file_list))  
    user_prompt = f"Here is the file structure of the project:\n\n{formatted_file_list}"  
    messages = [  
        {"role": "system", "content": FILE_SELECTION_SYSTEM_PROMPT},  
        {"role": "user", "content": user_prompt}  
    ]  
    try:  
        response = client.chat.completions.create(  
            model=DEPLOYMENT, messages=messages, response_format={"type": "json_object"}  
        )  
        raw_response = response.choices[0].message.content  
        return json.loads(raw_response)  
    except Exception as e:  
        logging.error(f"Error getting intelligent file selection from AI: {e}")  
        return {}  
  
  
def parse_code_structure(file_path, content):  
    """Parses code content to find high-level structures like classes and functions."""  
    ext = os.path.splitext(file_path)[1].lower()  
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
            ('function', re.compile(r'^\s*(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\('))  
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
  
  
def extract_code_block(content, start_line, file_ext):  
    """Extracts a full code block (function/class) based on indentation or braces."""  
    lines = content.splitlines()  
    block_lines = []  
  
    if file_ext == '.py':  
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
    else:  # Brace-based languages  
        block_lines.append(lines[start_line])  
        brace_count = 0  
        in_block = False  
        for i in range(start_line, len(lines)):  
            line = lines[i]  
            if '{' in line:  
                in_block = True  
            if not in_block:  
                continue  
            if i > start_line:  
                block_lines.append(line)  
            brace_count += line.count('{')  
            brace_count -= line.count('}')  
            if brace_count == 0 and in_block:  
                break  
  
    return '\n'.join(block_lines)  
  
  
# --- Flask Routes ---  
  
@app.route('/')  
def index():  
    return render_template_string(HTML_TEMPLATE)  
  
  
@app.route('/scan', methods=['POST'])  
def scan():  
    """Handles scanning, gets AI recommendations, and returns files found."""  
    global SCAN_BASE_PATH  
    data = request.get_json()  
    path_or_url = data.get('path', '.').strip() or '.'  
    scan_path, display_name = None, ""  
  
    try:  
        if is_github_url(path_or_url):  
            scan_path = clone_repo_to_tempdir(path_or_url)  
            display_name = path_or_url.split('/')[-1].replace('.git', '')  
        elif os.path.isdir(path_or_url):  
            scan_path = path_or_url  
            display_name = os.path.basename(os.path.abspath(path_or_url))  
        else:  
            return jsonify({"error": f"Input '{path_or_url}' is not a valid directory or a public GitHub repo URL."}), 400  
  
        files_found = scan_directory(scan_path)  
        SCAN_BASE_PATH = scan_path  
  
        if not files_found:  
            return jsonify({"error": "Scan complete, but no relevant code files were found."}), 404  
  
        recommendations = get_intelligent_selection(list(files_found.keys()))  
        return jsonify({  
            "displayName": display_name,  
            "files": sorted(list(files_found.keys())),  
            "recommendations": recommendations,  
        })  
    except Exception as e:  
        logging.error(f"An unexpected error occurred during scanning: {e}")  
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500  
  
  
@app.route('/get_content')  
def get_content():  
    """Returns the content of all scanned files."""  
    if not SCANNED_FILES_CONTENT:  
        return jsonify({"error": "No files have been scanned yet."}), 404  
    return jsonify(SCANNED_FILES_CONTENT)  
  
  
@app.route('/structure', methods=['POST'])  
def get_structure():  
    """Parses a file and returns its structure (functions, classes). Caches results."""  
    data = request.get_json()  
    file_path = data.get('path')  
    if not file_path:  
        return jsonify({"error": "File path is required."}), 400  
  
    if file_path in SCANNED_FILES_STRUCTURE:  
        return jsonify(SCANNED_FILES_STRUCTURE[file_path])  
    if file_path in SCANNED_FILES_CONTENT:  
        content = SCANNED_FILES_CONTENT[file_path]  
        structure = parse_code_structure(file_path, content)  
        SCANNED_FILES_STRUCTURE[file_path] = structure  
        return jsonify(structure)  
  
    return jsonify({"error": "File not found in scanned data."}), 404  
  
  
@app.route('/chat', methods=['POST'])  
def chat():  
    """Handles chat, builds context with user-controlled conversation history, and gets a reasoned response from the LLM."""  
    if not client:  
        return jsonify({"error": "Azure OpenAI client is not configured. Check server .env file."}), 500  
    if not SCANNED_FILES_CONTENT:  
        return jsonify({"error": "Please scan a directory or repository first."}), 400  
  
    data = request.get_json()  
    user_question = data.get('question')  
    selected_items = data.get('selected_items', {})  
    chat_history = data.get('history', [])  # The client now sends the curated history  
  
    if not user_question:  
        return jsonify({"error": "No user question found in the request."}), 400  
    if not selected_items:  
        return jsonify({"error": "No files or code snippets were selected for context."}), 400  
  
    context_parts = []  
    for file_path, items in selected_items.items():  
        if file_path not in SCANNED_FILES_CONTENT:  
            continue  
  
        content = SCANNED_FILES_CONTENT[file_path]  
        if "__all__" in items:  
            context_parts.append(f"--- FILE: {file_path} ---\n\n{content}\n\n")  
        else:  
            structure = SCANNED_FILES_STRUCTURE.get(file_path) or parse_code_structure(file_path, content)  
            SCANNED_FILES_STRUCTURE[file_path] = structure  
            item_map = {s['name']: s for s in structure}  
            file_ext = os.path.splitext(file_path)[1].lower()  
            for item_name in items:  
                if item_name in item_map:  
                    start_line = item_map[item_name]['start_line']  
                    block_content = extract_code_block(content, start_line, file_ext)  
                    context_parts.append(  
                        f"--- SNIPPET FROM: {file_path} ({item_map[item_name]['type']}: {item_name}) ---\n\n{block_content}\n\n"  
                    )  
  
    if not context_parts:  
        return jsonify({"error": "The selected items could not be found or extracted."}), 400  
  
    dynamic_codebase_context = "".join(context_parts)  
  
    prompt_text = CODE_ANALYSIS_SYSTEM_PROMPT + "\n\n"  
    for message in chat_history:  # Use the curated history from the client  
        prompt_text += f"**{message['role'].capitalize()}**: {message['content']}\n\n"  
  
    prompt_text += f"## CODEBASE CONTEXT\n\n{dynamic_codebase_context}\n\n## USER QUESTION\n\n{user_question}"  
  
    ai_response = get_reasoned_llm_response(client, prompt_text, DEPLOYMENT)  
    return jsonify(ai_response)  

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Explorer AI 3.2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #121921; color: #F9FAFB; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #212934; }
        ::-webkit-scrollbar-thumb { background: #4A5568; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #718096; }
        .mongodb-green { color: #00ED64; }
        .mongodb-green-bg { background-color: #00684A; }
        .mongodb-green-bg-hover:hover { background-color: #00ED64; color: #121921; }
        .mongodb-dark-bg { background-color: #212934; }
        .mongodb-border { border-color: #4A5568; }
        .chat-bubble-user { background-color: #00684A; }
        .chat-bubble-ai { background-color: #212934; }
        .markdown-content pre { background-color: #0e131a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; overflow-x: auto; position: relative; }
        .markdown-content code { background-color: #121921; color: #F9FAFB; padding: 0.2rem 0.4rem; border-radius: 4px; }
        details > summary { cursor: pointer; list-style: none; }
        details > summary::-webkit-details-marker { display: none; }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fadeInUp { animation: fadeInUp 0.3s ease-out forwards; }

        /* Custom colors for code structure types */
        .text-type-class { color: #C778DD; } /* Purple */
        .text-type-function { color: #61AFEF; } /* Blue */
        .text-type-type { color: #56B6C2; } /* Teal */
        .text-type-module { color: #E5C07B; } /* Yellow */
        .context-meter-gradient { background: linear-gradient(to right, #22c55e, #facc15, #ef4444); }
        .chat-context-meter-gradient { background: linear-gradient(to right, #38bdf8, #a78bfa, #f472b6); }

        /* Chat context toggle styles */
        .chat-bubble-wrapper { position: relative; }
        .context-toggle { position: absolute; top: -8px; left: -8px; width: 24px; height: 24px; background-color: #4A5568; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s ease; z-index: 10; border: 2px solid #121921; }
        .context-toggle:hover { transform: scale(1.1); }
        .chat-bubble-wrapper.context-off > .chat-bubble-ai,
        .chat-bubble-wrapper.context-off > .chat-bubble-user { opacity: 0.4; }
        .chat-bubble-wrapper.context-off .context-toggle { background-color: #7f1d1d; }
    </style>
</head>
<body class="flex flex-col h-screen">

    <header class="flex items-center justify-between p-4 border-b mongodb-border shadow-lg">
        <div class="flex items-center space-x-3">
            <svg class="w-8 h-8 mongodb-green" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M16.889 12.042C16.889 15.783 13.824 18.848 10.083 18.848C6.342 18.848 3.277 15.783 3.277 12.042C3.277 8.301 6.342 5.236 10.083 5.236C10.113 5.236 10.142 5.237 10.172 5.237C10.232 5.237 10.29 5.236 10.35 5.236C10.35 6.428 10.813 7.525 11.583 8.301C12.354 9.076 13.451 9.539 14.643 9.539C14.643 9.599 14.644 9.657 14.644 9.717C14.644 9.747 14.645 9.776 14.645 9.806C15.93 9.926 16.889 10.885 16.889 12.162L16.889 12.042ZM14.643 7.46C13.818 7.46 13.088 7.047 12.639 6.388C12.981 6.331 13.334 6.302 13.702 6.302C15.991 6.302 17.86 8.171 17.86 10.46C17.86 10.828 17.831 11.181 17.774 11.523C17.115 11.074 16.602 10.344 16.602 9.539C16.602 8.371 15.749 7.46 14.643 7.46Z" /></svg>
            <h1 class="text-2xl font-bold text-white">Code Explorer <span class="mongodb-green">AI</span></h1>
        </div>
        <div id="status-indicator" class="flex items-center space-x-2">
            <span id="status-text" class="text-sm text-gray-400 transition-all duration-300">Awaiting directory or repo scan...</span>
            <div id="status-dot" class="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
        </div>
    </header>

    <main class="flex-1 flex flex-col md:flex-row overflow-hidden">
        <aside class="w-full md:w-2/5 lg:w-1/3 p-4 border-r mongodb-border overflow-y-auto flex flex-col space-y-4">
            <div>
                <h2 class="text-lg font-semibold mb-2">1. Analyze Codebase</h2>
                <div class="flex space-x-2">
                    <input type="text" id="path-input" placeholder="/path/to/project or GitHub URL" class="flex-1 p-2 rounded-md bg-gray-800 border mongodb-border focus:outline-none focus:ring-2 focus:ring-green-500">
                    <button id="scan-btn" class="px-4 py-2 font-semibold rounded-md mongodb-green-bg mongodb-green-bg-hover transition-all duration-200 active:scale-95">
                        <span id="scan-btn-text">Scan</span>
                        <svg id="scan-spinner" class="animate-spin h-5 w-5 text-white hidden" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    </button>
                </div>
            </div>

            <div id="context-builder-section" class="hidden flex-1 flex flex-col min-h-0">
                <div class="flex justify-between items-center mb-2">
                    <h2 class="text-lg font-semibold">2. Build Context</h2>
                    <span id="context-item-count" class="text-xs font-mono text-gray-400 bg-gray-900 px-2 py-1 rounded">0 items</span>
                </div>
                
                <div id="ai-reasoning-box" class="hidden mongodb-dark-bg p-3 rounded-lg border mongodb-border mb-4">
                    <h3 class="font-semibold text-sm mb-2 text-green-400 flex items-center"><i class="fa-solid fa-lightbulb mr-2"></i>AI Recommended Starting Point</h3>
                    <div id="ai-reasoning-content" class="text-xs text-gray-300 markdown-content prose-sm"></div>
                </div>

                <p class="text-sm text-gray-400 mb-3">Expand files and select functions or classes to analyze.</p>
                <div id="file-tree" class="flex-1 overflow-y-auto pr-2 space-y-1 mongodb-dark-bg p-2 rounded-lg border mongodb-border"></div>
                
                <div class="mt-3 pt-3 border-t border-gray-700 space-y-2">
                    <div class="flex justify-between items-center text-xs">
                        <button id="preview-context-btn" class="text-gray-400 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed" disabled><i class="fa-solid fa-eye mr-1"></i>Preview Context</button>
                        <div id="code-context-meter" class="flex items-center space-x-2 font-mono">
                            <span class="text-right w-24" id="context-token-count">~0 tokens</span>
                            <div class="w-20 h-2 rounded-full bg-gray-700 overflow-hidden"><div id="context-meter-fill" class="h-2 rounded-full context-meter-gradient transition-all duration-300" style="width: 100%; transform: translateX(-100%)"></div></div>
                        </div>
                    </div>
                    <div class="flex justify-between items-center text-xs">
                         <span class="text-gray-400"><i class="fa-solid fa-comments mr-1"></i>Chat History</span>
                        <div id="chat-context-meter" class="flex items-center space-x-2 font-mono">
                            <span class="text-right w-24" id="chat-context-token-count">~0 tokens</span>
                            <div class="w-20 h-2 rounded-full bg-gray-700 overflow-hidden"><div id="chat-context-meter-fill" class="h-2 rounded-full chat-context-meter-gradient transition-all duration-300" style="width: 100%; transform: translateX(-100%)"></div></div>
                        </div>
                    </div>
                </div>
            </div>
        </aside>

        <section class="flex-1 flex flex-col p-4">
            <div id="chat-window" class="flex-1 overflow-y-auto mb-4 pr-2">
                 <div class="chat-bubble-ai p-4 rounded-lg max-w-4xl markdown-content">
                    <p>Welcome to Code Explorer! You can now click the icon on any chat message to exclude it from our conversation context. Provide a local directory or GitHub URL to begin.</p>
                </div>
            </div>
            <div id="preset-buttons" class="mb-2 flex-wrap gap-2 hidden">
                <button class="preset-btn text-sm p-2 rounded-md mongodb-dark-bg hover:bg-gray-700 transition-colors">High-level summary</button>
                <button class="preset-btn text-sm p-2 rounded-md mongodb-dark-bg hover:bg-gray-700 transition-colors">Identify dependencies</button>
                <button class="preset-btn text-sm p-2 rounded-md mongodb-dark-bg hover:bg-gray-700 transition-colors">Explain the core logic</button>
            </div>
            <div class="mt-auto">
                <form id="chat-form" class="flex items-center space-x-2">
                    <input type="text" id="message-input" placeholder="Select code snippets to begin..." class="flex-1 p-3 rounded-md bg-gray-800 border mongodb-border focus:outline-none focus:ring-2 focus:ring-green-500" disabled>
                    <button type="submit" id="send-btn" class="p-3 rounded-md mongodb-green-bg mongodb-green-bg-hover transition-all" disabled>
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                    </button>
                </form>
            </div>
        </section>
    </main>

    <div id="context-modal" class="fixed inset-0 bg-black/70 flex items-center justify-center p-4 hidden z-50">
        <div class="bg-gray-900 rounded-lg shadow-xl w-full max-w-4xl h-full max-h-[90vh] flex flex-col">
            <div class="flex justify-between items-center p-4 border-b mongodb-border">
                <h2 class="text-lg font-semibold">Full Request Context</h2>
                <button id="close-modal-btn" class="text-gray-400 hover:text-white text-2xl">&times;</button>
            </div>
            <pre class="flex-1 overflow-y-auto p-4 text-xs whitespace-pre-wrap"><code id="modal-code-content"></code></pre>
        </div>
    </div>

<script>
document.addEventListener('DOMContentLoaded', () => {
    const ui = {
        scanBtn: document.getElementById('scan-btn'), pathInput: document.getElementById('path-input'),
        messageInput: document.getElementById('message-input'), sendBtn: document.getElementById('send-btn'),
        chatForm: document.getElementById('chat-form'), chatWindow: document.getElementById('chat-window'),
        statusText: document.getElementById('status-text'), statusDot: document.getElementById('status-dot'),
        scanBtnText: document.getElementById('scan-btn-text'), scanSpinner: document.getElementById('scan-spinner'),
        contextBuilderSection: document.getElementById('context-builder-section'), fileTree: document.getElementById('file-tree'),
        presetButtons: document.getElementById('preset-buttons'), contextItemCount: document.getElementById('context-item-count'),
        aiReasoningBox: document.getElementById('ai-reasoning-box'), aiReasoningContent: document.getElementById('ai-reasoning-content'),
        previewContextBtn: document.getElementById('preview-context-btn'), contextTokenCount: document.getElementById('context-token-count'),
        contextMeterFill: document.getElementById('context-meter-fill'), chatContextTokenCount: document.getElementById('chat-context-token-count'),
        chatContextMeterFill: document.getElementById('chat-context-meter-fill'), contextModal: document.getElementById('context-modal'),
        closeModalBtn: document.getElementById('close-modal-btn'), modalCodeContent: document.getElementById('modal-code-content'),
    };

    const CONTEXT_TOKEN_LIMIT = 200000;
    let fileContents = {}; 
    let messageStates = new Map();

    function addMessageToChat(role, content, isThinking = false) {
        const messageId = `msg-${Date.now()}-${Math.random()}`;
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `flex mb-4 animate-fadeInUp ${role === 'user' ? 'justify-end' : 'justify-start'}`;
        
        const bubbleWrapper = document.createElement('div');
        bubbleWrapper.className = 'chat-bubble-wrapper';
        bubbleWrapper.id = messageId;
        
        const bubble = document.createElement('div');
        bubble.className = `p-4 rounded-lg max-w-4xl ${role === 'user' ? 'chat-bubble-user text-white' : 'chat-bubble-ai'}`;
        
        if (isThinking) {
            bubble.innerHTML = `<div class="thinking-indicator flex items-center space-x-2"><div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div><div class="w-2 h-2 bg-green-400 rounded-full animate-pulse" style="animation-delay: 0.2s;"></div><div class="w-2 h-2 bg-green-400 rounded-full animate-pulse" style="animation-delay: 0.4s;"></div><span class="text-sm text-gray-400">Analyzing...</span></div>`;
        } else {
            const contextToggle = document.createElement('div');
            contextToggle.className = 'context-toggle';
            contextToggle.innerHTML = `<i class="fa-solid fa-check text-xs text-green-300"></i>`;
            contextToggle.title = 'Click to exclude from context';
            contextToggle.addEventListener('click', () => toggleMessageContext(messageId));
            bubbleWrapper.appendChild(contextToggle);

            if (role === 'assistant' && typeof content === 'object') {
                let html = '';
                if (content.summaries && content.summaries.length > 0) {
                    const summaryItems = content.summaries.map(s => `<li>${marked.parseInline(s)}</li>`).join('');
                    html += `<details class="mongodb-dark-bg border border-gray-600 rounded-md mb-3"><summary class="p-2 cursor-pointer text-sm font-semibold flex items-center"><i class="fa-solid fa-chevron-right w-4 mr-2 transition-transform"></i>Show Reasoning</summary><div class="p-3 border-t border-gray-600 markdown-content text-sm"><ul class="list-disc pl-5 space-y-1">${summaryItems}</ul></div></details>`;
                }
                html += `<div class="markdown-content">${marked.parse(content.answer)}</div>`;
                bubble.innerHTML = html;
                bubble.querySelector('details')?.addEventListener('toggle', (e) => e.target.querySelector('i').classList.toggle('fa-rotate-90'));
                messageStates.set(messageId, { role, content: content.answer, inContext: true });
            } else {
                bubble.innerHTML = `<div class="markdown-content">${marked.parse(content)}</div>`;
                messageStates.set(messageId, { role, content, inContext: true });
            }
        }
        
        bubbleWrapper.appendChild(bubble);
        messageWrapper.appendChild(bubbleWrapper);
        ui.chatWindow.appendChild(messageWrapper);
        ui.chatWindow.scrollTop = ui.chatWindow.scrollHeight;
        
        if (!isThinking) {
            enhanceCodeBlocks(bubble);
            updateChatContextMeter();
        }
        return messageWrapper;
    }
    
    function toggleMessageContext(messageId) {
        const state = messageStates.get(messageId);
        if (!state) return;
        state.inContext = !state.inContext;
        
        const bubbleWrapper = document.getElementById(messageId);
        const toggleIcon = bubbleWrapper.querySelector('.context-toggle i');
        
        if (state.inContext) {
            bubbleWrapper.classList.remove('context-off');
            toggleIcon.className = 'fa-solid fa-check text-xs text-green-300';
            bubbleWrapper.querySelector('.context-toggle').title = 'Click to exclude from context';
        } else {
            bubbleWrapper.classList.add('context-off');
            toggleIcon.className = 'fa-solid fa-ban text-xs text-red-300';
            bubbleWrapper.querySelector('.context-toggle').title = 'Click to include in context';
        }
        updateChatContextMeter();
    }

    function enhanceCodeBlocks(container) {
        container.querySelectorAll('pre').forEach(pre => {
            if (pre.querySelector('.copy-btn')) return;
            const code = pre.querySelector('code');
            const copyBtn = document.createElement('button');
            copyBtn.innerHTML = '<i class="far fa-copy"></i> Copy';
            copyBtn.className = 'copy-btn absolute top-2 right-2 px-2 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 transition-colors';
            copyBtn.onclick = () => {
                const textArea = document.createElement("textarea");
                textArea.value = code.innerText;
                document.body.appendChild(textArea);
                textArea.select();
                try {
                    document.execCommand('copy');
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                } catch (err) {
                    copyBtn.innerHTML = '<i class="fas fa-times"></i> Failed';
                }
                document.body.removeChild(textArea);
                setTimeout(() => { copyBtn.innerHTML = '<i class="far fa-copy"></i> Copy'; }, 2000);
            };
            pre.appendChild(copyBtn);
        });
    }

    function setScanningState(isScanning) {
        ui.scanBtn.disabled = isScanning;
        ui.pathInput.disabled = isScanning;
        ui.scanSpinner.classList.toggle('hidden', !isScanning);
        ui.scanBtnText.classList.toggle('hidden', isScanning);
    }

    function resetUI() {
        fileContents = {};
        messageStates.clear();
        ui.contextBuilderSection.classList.add('hidden');
        ui.aiReasoningBox.classList.add('hidden');
        ui.fileTree.innerHTML = '';
        ui.presetButtons.classList.add('hidden');
        ui.messageInput.disabled = true;
        ui.sendBtn.disabled = true;
        ui.messageInput.placeholder = "Select code snippets to begin...";
        ui.statusDot.classList.remove('bg-green-500', 'bg-red-500');
        ui.statusDot.classList.add('bg-yellow-500', 'animate-pulse');
        updateContextMeter();
        updateChatContextMeter();
    }
    
    function updateContextItemCount() {
        const selectedCount = ui.fileTree.querySelectorAll('input[type="checkbox"]:checked').length;
        ui.contextItemCount.textContent = `${selectedCount} items`;
    }

    function updateContextMeter() {
        let charCount = 0;
        const selectedItems = getSelectedItems();
        for (const path in selectedItems) {
            if (selectedItems[path].includes('__all__')) {
                charCount += (fileContents[path] || '').length;
            } else {
                charCount += selectedItems[path].length * 500;
            }
        }
        const tokenEstimate = Math.floor(charCount / 3.5);
        ui.contextTokenCount.textContent = `~${tokenEstimate.toLocaleString()} tokens`;
        const percentage = Math.min(100, (tokenEstimate / CONTEXT_TOKEN_LIMIT) * 100);
        ui.contextMeterFill.style.transform = `translateX(-${100 - percentage}%)`;
    }
    
    function updateChatContextMeter() {
        let charCount = 0;
        for (const state of messageStates.values()) {
            if (state.inContext) {
                charCount += (typeof state.content === 'string' ? state.content.length : 0);
            }
        }
        const tokenEstimate = Math.floor(charCount / 3.5);
        ui.chatContextTokenCount.textContent = `~${tokenEstimate.toLocaleString()} tokens`;
        const percentage = Math.min(100, (tokenEstimate / CONTEXT_TOKEN_LIMIT) * 100);
        ui.chatContextMeterFill.style.transform = `translateX(-${100 - percentage}%)`;
    }

    function updateChatUI() {
        const selectedCount = ui.fileTree.querySelectorAll('input[type="checkbox"]:checked').length;
        if (selectedCount > 0) {
            ui.messageInput.disabled = false;
            ui.sendBtn.disabled = false;
            ui.previewContextBtn.disabled = false;
            ui.messageInput.placeholder = `Ask about the selected code...`;
            if (ui.presetButtons.classList.contains('hidden')) {
                 ui.presetButtons.classList.remove('hidden');
                 ui.presetButtons.classList.add('flex', 'animate-fadeInUp');
            }
        } else {
            ui.messageInput.disabled = true;
            ui.sendBtn.disabled = true;
            ui.previewContextBtn.disabled = true;
            ui.messageInput.placeholder = "Select code snippets to begin...";
            ui.presetButtons.classList.add('hidden');
            ui.presetButtons.classList.remove('flex', 'animate-fadeInUp');
        }
        updateContextItemCount();
        updateContextMeter();
        updateChatContextMeter();
    }
    
    ui.scanBtn.addEventListener('click', async () => {
        const path = ui.pathInput.value.trim() || '.';
        setScanningState(true);
        resetUI();
        ui.statusText.textContent = 'Scanning and analyzing...';
        try {
            const response = await fetch('/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Unknown error');
            
            const contentResponse = await fetch('/get_content');
            fileContents = await contentResponse.json();
            if (!contentResponse.ok) {
                console.warn("Could not fetch file contents for context preview.");
                fileContents = {};
            }
            
            ui.statusText.textContent = `Scanned: ${result.displayName}`;
            ui.statusDot.classList.remove('bg-yellow-500', 'animate-pulse');
            ui.statusDot.classList.add('bg-green-500');

            const recommendedFiles = result.recommendations?.files || [];
            if(result.recommendations?.reasoning) {
                ui.aiReasoningContent.innerHTML = marked.parse(result.recommendations.reasoning);
                ui.aiReasoningBox.classList.remove('hidden');
                ui.aiReasoningBox.classList.add('animate-fadeInUp');
            }

            result.files.forEach(file => createFileTreeItem(file, recommendedFiles.includes(file)));
            
            ui.contextBuilderSection.classList.remove('hidden');
            ui.contextBuilderSection.classList.add('animate-fadeInUp');
            ui.chatWindow.innerHTML = '';
            addMessageToChat('assistant', {answer: `Analysis complete for \`${result.displayName}\`. I've recommended some starting files. You can adjust the selection and ask me anything.`, summaries: []});
            updateChatUI();
        } catch (error) {
            ui.statusText.textContent = 'Scan failed.';
            ui.statusDot.classList.add('bg-red-500');
            addMessageToChat('assistant', {answer: `**Error:** ${error.message}`, summaries: []});
        } finally {
            setScanningState(false);
        }
    });
    
    function createFileTreeItem(file, isRecommended = false) {
        const details = document.createElement('details');
        details.className = 'bg-gray-800/50 rounded-md';
        details.dataset.path = file;
        details.dataset.loaded = 'false';

        const summary = document.createElement('summary');
        summary.className = 'flex items-center p-2 cursor-pointer hover:bg-gray-700/50 rounded-md';
        
        const fileCheckbox = document.createElement('input');
        fileCheckbox.type = 'checkbox';
        fileCheckbox.className = 'h-4 w-4 rounded text-green-500 bg-gray-700 border-gray-600 focus:ring-green-600';
        fileCheckbox.dataset.type = 'file';
        fileCheckbox.checked = isRecommended;
        fileCheckbox.addEventListener('change', (e) => {
            details.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = e.target.checked);
            updateChatUI();
        });

        const icon = document.createElement('i');
        icon.className = `fa-solid fa-chevron-right w-6 text-center transition-transform`;
        details.addEventListener('toggle', () => icon.classList.toggle('fa-rotate-90'));

        const ICONS = { file: 'fa-regular fa-file-code', class: 'fa-solid fa-cube text-type-class', function: 'fa-solid fa-microchip text-type-function', type: 'fa-solid fa-tag text-type-type', module: 'fa-solid fa-cubes text-type-module' };
        const fileIcon = document.createElement('i');
        fileIcon.className = `${ICONS.file} mx-2 text-gray-400`;

        const label = document.createElement('span');
        label.textContent = file;
        label.className = 'text-sm';
        
        summary.append(fileCheckbox, icon, fileIcon, label);
        details.appendChild(summary);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'pl-8 pr-2 pb-2';
        details.appendChild(contentDiv);
        
        ui.fileTree.appendChild(details);

        details.addEventListener('toggle', async (e) => {
            if (details.open && details.dataset.loaded === 'false') {
                contentDiv.innerHTML = `<div class="text-xs text-gray-500">Loading structure...</div>`;
                try {
                    const response = await fetch('/structure', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({path: file})
                    });
                    const structure = await response.json();
                    if (!response.ok) throw new Error(structure.error);

                    details.dataset.loaded = 'true';
                    contentDiv.innerHTML = '';
                    if (structure.length > 0) {
                        structure.forEach(item => {
                           const itemDiv = document.createElement('div');
                           itemDiv.className = 'flex items-center mt-1';
                           const itemCheckbox = document.createElement('input');
                           itemCheckbox.type = 'checkbox';
                           itemCheckbox.className = 'h-4 w-4 rounded text-green-500 bg-gray-700 border-gray-600 focus:ring-green-600';
                           itemCheckbox.value = item.name;
                           itemCheckbox.dataset.type = 'item';
                           itemCheckbox.checked = fileCheckbox.checked;
                           itemCheckbox.addEventListener('change', () => {
                                const allItems = contentDiv.querySelectorAll('input[data-type="item"]');
                                const checkedItems = contentDiv.querySelectorAll('input[data-type="item"]:checked');
                                fileCheckbox.checked = allItems.length === checkedItems.length;
                                fileCheckbox.indeterminate = checkedItems.length > 0 && checkedItems.length < allItems.length;
                                updateChatUI();
                           });
                           
                           const itemIcon = document.createElement('i');
                           const iconClass = ICONS[item.type] || 'fa-solid fa-code';
                           itemIcon.className = `${iconClass} w-6 text-center`;
                           
                           const itemLabel = document.createElement('span');
                           itemLabel.textContent = item.name;
                           const colorClass = `text-type-${item.type}`;
                           itemLabel.className = `text-sm ml-2 ${colorClass}`;
                           
                           itemDiv.append(itemCheckbox, itemIcon, itemLabel);
                           contentDiv.appendChild(itemDiv);
                        });
                    } else {
                        contentDiv.innerHTML = `<div class="text-xs text-gray-500 italic">No functions or classes found.</div>`;
                    }
                } catch(error) {
                    contentDiv.innerHTML = `<div class="text-xs text-red-400">Error loading structure: ${error.message}</div>`;
                }
            }
        });
    }
    
    function getSelectedItems() {
        const selected = {};
        ui.fileTree.querySelectorAll('details').forEach(detail => {
            const filePath = detail.dataset.path;
            const fileCheckbox = detail.querySelector('input[data-type="file"]');
            const itemCheckboxes = detail.querySelectorAll('input[data-type="item"]:checked');
            if (fileCheckbox.checked) {
                selected[filePath] = ['__all__'];
            } else if (itemCheckboxes.length > 0) {
                selected[filePath] = Array.from(itemCheckboxes).map(cb => cb.value);
            }
        });
        return selected;
    }

    async function handleSendMessage(messageText) {
        if (!messageText.trim()) return;
        const selectedItems = getSelectedItems();
        if (Object.keys(selectedItems).length === 0) {
             addMessageToChat('assistant', {answer: `**Error:** Please select files or code snippets before asking a question.`, summaries: []});
             return;
        }

        addMessageToChat('user', messageText);
        ui.messageInput.value = '';
        const thinkingMessage = addMessageToChat('assistant', '', true);
        ui.sendBtn.disabled = true;
        ui.messageInput.disabled = true;

        const historyForPrompt = [];
        for (const state of messageStates.values()) {
            if (state.inContext) {
                historyForPrompt.push({ role: state.role, content: state.content });
            }
        }

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: messageText,
                    selected_items: selectedItems,
                    history: historyForPrompt
                }),
            });
            
            thinkingMessage.remove();
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'The server returned an error.');
            
            addMessageToChat('assistant', result);

        } catch (error) {
            thinkingMessage.remove();
            addMessageToChat('assistant', { answer: `<p class="text-red-400"><strong>Error:</strong> ${error.message}</p>`, summaries: [] });
        } finally {
            ui.sendBtn.disabled = false;
            ui.messageInput.disabled = false;
            updateChatUI();
        }
    }

    ui.chatForm.addEventListener('submit', (e) => { e.preventDefault(); handleSendMessage(ui.messageInput.value); });
    
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.addEventListener('click', () => {
            const questions = {
                'High-level summary': "Provide a high-level summary of this project based on the selected context.",
                'Identify dependencies': "From the selected code, identify the main dependencies and libraries used.",
                'Explain the core logic': "Based on the selected files, explain the core logic and primary purpose of this application.",
            };
            handleSendMessage(questions[button.textContent]);
        });
    });

    ui.pathInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') ui.scanBtn.click(); });

    ui.previewContextBtn.addEventListener('click', () => {
        // This function would need to be updated to show the curated chat history as well
        const codeContext = buildFullContext(ui.messageInput.value || "[Your question here]");
        ui.modalCodeContent.textContent = codeContext; // Simplified for now
        ui.contextModal.classList.remove('hidden');
    });
    ui.closeModalBtn.addEventListener('click', () => ui.contextModal.classList.add('hidden'));
    ui.contextModal.addEventListener('click', (e) => {
        if (e.target === ui.contextModal) ui.contextModal.classList.add('hidden');
    });
});
</script>
</body>
</html>
"""

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    print("--- Code Explorer Web UI ---")
    print(f"🚀 Starting server at http://{host}:{port}")
    print("🔧 Make sure your .env file is configured correctly and Git is installed.")
    print("👉 Open the URL in your browser to start analyzing code with transparent reasoned responses!")
    print("--------------------------------")
    app.run(host=host, port=port, debug=False)
