"""Language-aware code parsing: structure extraction, action tagging, and a
line-window fallback so that *every* file in a repo gets indexed, not just
Python/JS/TS.
"""
import ast
import os
import re

# --- Scan filters ---
IGNORE_DIRS = {'.git', '.github', '__pycache__', 'node_modules', 'venv', '.venv',
               '.vscode', 'build', 'dist', 'docs', '.idea', '.next', 'target',
               '.mypy_cache', '.pytest_cache', 'vendor'}
IGNORE_EXTS = {'.pyc', '.log', '.env', '.ds_store', '.tmp', '.swo', '.swp', '.lock',
               '.map', '.min.js', '.min.css'}
BINARY_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.zip', '.gz', '.pdf',
               '.exe', '.dll', '.so', '.webp', '.svg', '.woff', '.woff2', '.ttf',
               '.mp4', '.mp3', '.wav', '.bin', '.dylib', '.class', '.jar'}

STRUCTURE_EXTS = {'.py', '.js', '.jsx', '.ts', '.tsx'}


# --- Action Tagging Definitions ---
ACTION_DEFINITIONS = {
    'data-storage': {
        'imports': {
            'sqlalchemy', 'django', 'pymongo', 'psycopg2', 'mysql', 'sqlite3', 'redis', 'cassandra',
            'motor', 'asyncpg', 'peewee', 'dataset', 'records', 'aiosqlite', 'tortoise', 'pony',
            'mongoose', 'sequelize', 'prisma', 'typeorm', 'knex', 'slonik', 'pg', 'mysql2', 'ioredis'
        },
        'regex': re.compile(r"""
            \b(Model|Base|Document|EmbeddedDocument)\b\s*\(         |
            class\s+\w+\(models\.Model\):                            |
            db\.session\.(query|add|commit|execute|flush)            |
            \.objects\.(filter|get|all|create)                       |
            \.collection\s*\[\s*['"]                                 |
            \.collection\s*\(\s*['"]                                 |
            \b(create_engine|declarative_base|sessionmaker)\b\s*\(   |
            \b(Column|String|Integer|Float|Boolean|ForeignKey|relationship|backref)\b\s*\( |
            \b(execute|executemany|fetchone|fetchall)\b\s*\(         |
            redis\.Redis\(|redis\.StrictRedis\(                        |
            new\s+(mongoose|Sequelize)\.Schema\s*\(                  |
            new\s+PrismaClient\s*\(                                  |
            \.prisma\.\w+\.(findUnique|findMany|create)              |
            @(Entity|Table|PrimaryGeneratedColumn|Column)\(          |
            \b(createConnection|getManager|getRepository)\b\s*\(     |
            Sequelize\.define\s*\(                                   |
            knex\.schema\.                                          |
            db\.collection\s*\(                                      |
            \.connect\s*\(\s*['"](postgres|mongodb|mysql):
        """, re.IGNORECASE | re.VERBOSE)
    },
    'network-request': {
        'imports': {
            'requests', 'aiohttp', 'httpx', 'urllib3', 'urllib', 'fastapi', 'flask', 'django', 'sanic', 'tornado',
            'axios', 'got', 'superagent', 'node-fetch', 'express', 'koa', 'fastify', 'hapi'
        },
        'regex': re.compile(r"""
            \b(requests|httpx)\.(get|post|put|delete|request)\s*\(    |
            aiohttp\.ClientSession                                  |
            urllib\.request\.urlopen                                |
            @(app|api|router)\.(get|post|put|delete)\s*\(\s*['"]      |
            app\.route\s*\(\s*['"]                                   |
            \bfetch\s*\(                                             |
            \b(axios|got|superagent)\.(get|post|put|delete)           |
            (app|router)\.(get|post|put|delete|use)\s*\(\s*['"]       |
            express\.Router                                          |
            http\.createServer
        """, re.IGNORECASE | re.VERBOSE)
    },
    'file-io': {
        'imports': {
            'os', 'shutil', 'pathlib', 'io', 'tempfile',
            'fs', 'path'
        },
        'regex': re.compile(r"""
            \bopen\s*\(                                              |
            \b(os\.path|pathlib\.Path)                               |
            \b(shutil|tempfile)\.                                    |
            \.(read|write|read_csv|to_csv)\(                         |
            \bfs\.(readFile|writeFile|readFileSync|writeFileSync|createReadStream|promises) |
            path\.(join|resolve|dirname)
        """, re.IGNORECASE | re.VERBOSE)
    }
}


class PythonCodeParser(ast.NodeVisitor):
    """A robust AST parser for Python."""
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
    try:
        tree = ast.parse(content)
        parser = PythonCodeParser()
        parser.visit(tree)
        return sorted(parser.structure, key=lambda x: x['start_line'])
    except SyntaxError:
        return []


def _parse_js_ts_structure(content):
    structure = []
    pattern = re.compile(
        r"""
        ^
        \s*
        (?:export\s+)?
        (?:default\s+)?
        (?:async\s+)?
        (?:
            (?:function(?:\s+\*)?|class|interface|type)\s+
            (?P<name1>[\w\$]+)
            |
            (?:const|let|var)\s+
            (?P<name2>[\w\$]+)\s*[:=]
        )
        """,
        re.MULTILINE | re.VERBOSE
    )

    for match in pattern.finditer(content):
        name = match.group('name1') or match.group('name2')
        if not name:
            continue
        line_number = content.count('\n', 0, match.start()) + 1
        full_match_text = match.group(0)
        item_type = 'function'
        if 'class' in full_match_text:
            item_type = 'class'
        elif 'interface' in full_match_text:
            item_type = 'interface'
        elif 'type' in full_match_text:
            item_type = 'interface'
        structure.append({'name': name, 'type': item_type, 'start_line': line_number})

    return sorted(structure, key=lambda x: x['start_line'])


def get_imports_from_content(content, file_path):
    """Extract imported top-level modules from file content."""
    imports = set()
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.py':
        py_pattern = re.compile(r'^\s*(?:from\s+([^\s.]+)|import\s+([^\s.]+))', re.MULTILINE)
        for match in py_pattern.finditer(content):
            module = (match.group(1) or match.group(2))
            if module:
                imports.add(module.split('.')[0])
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        js_pattern = re.compile(r'(?:import|from|require)\s*\(?\s*[\'"]([^\'"]+)[\'"]')
        for match in js_pattern.finditer(content):
            module_name = match.group(1)
            if module_name and not module_name.startswith(('.', '/')):
                if module_name.startswith('@'):
                    parts = module_name.split('/')
                    if len(parts) > 0:
                        imports.add(parts[0][1:])
                    if len(parts) > 1:
                        imports.add(parts[1])
                else:
                    imports.add(module_name.split('/')[0])
    return imports


def get_import_header(content, file_path, max_lines=40):
    """Return the raw import/require lines from the top of a file, as a compact
    string. Used to give the LLM dependency context when analyzing a snippet."""
    ext = os.path.splitext(file_path)[1].lower()
    lines = content.splitlines()
    header = []
    if ext == '.py':
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                header.append(stripped)
            if len(header) >= max_lines:
                break
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        for line in lines[:120]:
            stripped = line.strip()
            if stripped.startswith(('import ', 'const ', 'require', 'export ')) and \
               ('require(' in stripped or 'from ' in stripped or stripped.startswith('import ')):
                header.append(stripped)
            if len(header) >= max_lines:
                break
    return "\n".join(header)


def analyze_content_for_tags(content, imports):
    tags = set()
    for tag, definition in ACTION_DEFINITIONS.items():
        if not imports.isdisjoint(definition['imports']):
            tags.add(tag)
        if definition['regex'].search(content):
            tags.add(tag)
    return sorted(list(tags))


def parse_code_structure(file_path, content):
    """Dispatcher: select parser, get imports, assign tags."""
    ext = os.path.splitext(file_path)[1].lower()
    structure = []

    if ext == '.py':
        structure = _parse_python_structure(content)
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        structure = _parse_js_ts_structure(content)

    all_imports = get_imports_from_content(content, file_path)
    file_level_tags = set(analyze_content_for_tags(content, all_imports))

    for item in structure:
        item_code = extract_code_block(content, item['start_line'], item.get('end_line'), ext)
        item_tags = analyze_content_for_tags(item_code, all_imports)
        item['tags'] = item_tags
        file_level_tags.update(item_tags)

    return structure, sorted(list(file_level_tags))


def extract_code_block(content, start_line, end_line=None, file_ext='.js'):
    """Extract a full code block via end_line (AST) or brace/indentation counting."""
    lines = content.splitlines()
    start_line_idx = start_line - 1

    if start_line_idx >= len(lines) or start_line_idx < 0:
        return ""

    if end_line is not None and end_line <= len(lines):
        return '\n'.join(lines[start_line_idx: end_line])

    if file_ext == '.py':
        initial_indent = len(lines[start_line_idx]) - len(lines[start_line_idx].lstrip(' '))
        block_lines = [lines[start_line_idx]]
        for i in range(start_line_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                block_lines.append(line)
                continue
            current_indent = len(line) - len(line.lstrip(' '))
            if current_indent > initial_indent:
                block_lines.append(line)
            else:
                break
        return '\n'.join(block_lines)
    else:
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
            if i == start_line_idx and '=>' in line and not line.strip().endswith('{'):
                break
        return '\n'.join(block_lines)


def line_window_snippets(content, window=60, overlap=15):
    """Fallback 'structure' for files we can't parse (Go, Rust, Markdown, YAML,
    etc.). Splits the file into overlapping line windows so every file still
    produces searchable, embeddable snippets.

    Returns a list of dicts shaped like the structure items from the real
    parsers: {name, type, start_line, end_line, tags}.
    """
    lines = content.splitlines()
    n = len(lines)
    if n == 0:
        return []

    snippets = []
    step = max(1, window - overlap)
    start = 0
    part = 1
    while start < n:
        end = min(n, start + window)
        snippets.append({
            'name': f'lines_{start + 1}_{end}' if n > window else 'file_contents',
            'type': 'block',
            'start_line': start + 1,
            'end_line': end,
            'tags': [],
        })
        part += 1
        if end >= n:
            break
        start += step
    return snippets


def get_structure_or_windows(file_path, content):
    """Return (structure, file_tags). For supported languages, uses the real
    parsers. For everything else, falls back to overlapping line windows so the
    file still gets indexed."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in STRUCTURE_EXTS:
        structure, file_tags = parse_code_structure(file_path, content)
        if structure:
            return structure, file_tags
        # Parsed but empty (e.g. a script with only top-level statements):
        # fall through to windows so we still index it.
        windows = line_window_snippets(content)
        return windows, file_tags

    imports = get_imports_from_content(content, file_path)
    file_tags = analyze_content_for_tags(content, imports)
    return line_window_snippets(content), file_tags
