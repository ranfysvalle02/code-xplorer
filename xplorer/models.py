"""Model registry: a single place that knows every embedding + chat model, which
provider serves it, whether it's currently available, and how to call it.

Providers:
  - ollama  (default, local, free)   -> chat + embeddings + reranking-via-LLM
  - voyage  (optional, VOYAGE_API_KEY) -> code embeddings + rerank-2.5
  - azure   (optional, AZURE_* vars)   -> chat only

The registry makes embedding models trivially pickable at runtime and never
hard-codes a vector dimension: dims are probed once per model and cached.
"""
import logging
import re
import threading
import time

logger = logging.getLogger("xplorer.models")

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

try:
    import voyageai
except ImportError:  # pragma: no cover
    voyageai = None


def slugify(model_id: str) -> str:
    """Turn 'ollama/qwen3-embedding:4b' into 'ollama_qwen3_embedding_4b' for use
    as a Mongo field name + vector index name."""
    return re.sub(r'[^a-z0-9]+', '_', model_id.lower()).strip('_')


class EmbeddingModel:
    def __init__(self, model_id, provider, model_name, dims=None, max_tokens=8192,
                 query_prefix="", doc_prefix="", input_type_aware=False,
                 batch_size=32, label="", notes=""):
        self.id = model_id
        self.provider = provider
        self.model_name = model_name
        self.dims = dims
        self.max_tokens = max_tokens
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.input_type_aware = input_type_aware
        self.batch_size = batch_size
        self.label = label or model_id
        self.notes = notes

    @property
    def slug(self):
        return slugify(self.id)

    @property
    def field_path(self):
        return f"embeddings.{self.slug}"

    @property
    def index_name(self):
        return f"vec_{self.slug}"

    def to_dict(self, available=False):
        return {
            "id": self.id,
            "provider": self.provider,
            "model_name": self.model_name,
            "dims": self.dims,
            "label": self.label,
            "notes": self.notes,
            "available": available,
            "slug": self.slug,
        }


# Built-in catalog. dims are best-known defaults; they're re-probed on first use.
_OLLAMA_EMBED = [
    EmbeddingModel("ollama/qwen3-embedding:0.6b", "ollama", "qwen3-embedding:0.6b",
                   dims=1024, label="Qwen3 Embedding 0.6B", notes="Fast, light code/text embeddings"),
    EmbeddingModel("ollama/qwen3-embedding:4b", "ollama", "qwen3-embedding:4b",
                   dims=2560, label="Qwen3 Embedding 4B", notes="Strong code embeddings (default)"),
    EmbeddingModel("ollama/qwen3-embedding:8b", "ollama", "qwen3-embedding:8b",
                   dims=4096, label="Qwen3 Embedding 8B", notes="Highest quality, heavier"),
    EmbeddingModel("ollama/nomic-embed-text", "ollama", "nomic-embed-text",
                   dims=768, query_prefix="search_query: ", doc_prefix="search_document: ",
                   label="Nomic Embed Text", notes="Classic general-purpose embeddings"),
    EmbeddingModel("ollama/embeddinggemma", "ollama", "embeddinggemma",
                   dims=768, label="EmbeddingGemma", notes="Google Gemma embeddings"),
    EmbeddingModel("ollama/mxbai-embed-large", "ollama", "mxbai-embed-large",
                   dims=1024, label="mxbai Embed Large", notes="High-quality general embeddings"),
]

_VOYAGE_EMBED = [
    EmbeddingModel("voyage/voyage-code-3", "voyage", "voyage-code-3",
                   dims=1024, input_type_aware=True, batch_size=96,
                   label="Voyage code-3", notes="Specialized code embeddings (cloud)"),
    EmbeddingModel("voyage/voyage-3.5", "voyage", "voyage-3.5",
                   dims=1024, input_type_aware=True, batch_size=96,
                   label="Voyage 3.5", notes="General-purpose embeddings (cloud)"),
    EmbeddingModel("voyage/voyage-3.5-lite", "voyage", "voyage-3.5-lite",
                   dims=1024, input_type_aware=True, batch_size=128,
                   label="Voyage 3.5 Lite", notes="Fast, cheap embeddings (cloud)"),
]

# Suggested local chat models (any pulled Ollama model also works).
_OLLAMA_CHAT_SUGGESTED = [
    {"id": "qwen3.6:27b", "label": "Qwen3.6 27B", "notes": "Best local reasoning"},
    {"id": "qwen3:14b", "label": "Qwen3 14B", "notes": "Fast local reasoning"},
    {"id": "llama3.1:latest", "label": "Llama 3.1 8B", "notes": "General chat"},
    {"id": "llama3:latest", "label": "Llama 3 8B", "notes": "General chat"},
    {"id": "phi3:latest", "label": "Phi-3 3.8B", "notes": "Tiny + quick"},
]


class ModelRegistry:
    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self._dims_cache = {}
        self._pulled_cache = None
        self._pulled_cache_ts = 0.0
        self._voyage_ok = None

        self.ollama = None
        if ollama is not None:
            try:
                self.ollama = ollama.Client(host=config.ollama_host)
            except Exception as e:  # pragma: no cover
                logger.error(f"Could not create Ollama client: {e}")

        self.voyage = None
        if config.voyage_enabled and voyageai is not None:
            try:
                self.voyage = voyageai.Client(api_key=config.voyage_api_key)
            except Exception as e:
                logger.error(f"Could not create Voyage client: {e}")

        self.azure = None
        if config.azure_enabled:
            try:
                from openai import AzureOpenAI
                self.azure = AzureOpenAI(
                    api_version=config.azure_api_version,
                    azure_endpoint=config.azure_endpoint,
                    api_key=config.azure_api_key,
                    timeout=60.0,
                )
            except Exception as e:
                logger.error(f"Could not create Azure client: {e}")

        self._embed_by_id = {m.id: m for m in (_OLLAMA_EMBED + _VOYAGE_EMBED)}

        # Active selections (mutable at runtime via /models/select).
        self.active_embedding_id = config.embedding_model
        self.active_llm = config.llm_model
        self.active_fast_llm = config.llm_fast_model

    # ---------------------------------------------------------------- Ollama
    def ollama_up(self):
        if self.ollama is None:
            return False
        try:
            self.ollama.list()
            return True
        except Exception:
            return False

    def pulled_models(self, force=False):
        """Set of model names currently pulled in Ollama (cached ~5s)."""
        now = time.time()
        if not force and self._pulled_cache is not None and now - self._pulled_cache_ts < 5:
            return self._pulled_cache
        names = set()
        if self.ollama is not None:
            try:
                resp = self.ollama.list()
                for m in getattr(resp, "models", []) or []:
                    name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
                    if name:
                        names.add(name)
            except Exception as e:
                logger.warning(f"Ollama list failed: {e}")
        self._pulled_cache = names
        self._pulled_cache_ts = now
        return names

    def _ollama_has(self, model_name):
        pulled = self.pulled_models()
        if model_name in pulled:
            return True
        # Tolerate the ':latest' tag in either direction.
        if ":" not in model_name and f"{model_name}:latest" in pulled:
            return True
        if model_name.endswith(":latest") and model_name[:-len(":latest")] in pulled:
            return True
        return False

    def _ollama_capabilities(self, model_name):
        if self.ollama is None:
            return []
        try:
            info = self.ollama.show(model_name)
            caps = getattr(info, "capabilities", None)
            if caps is None and isinstance(info, dict):
                caps = info.get("capabilities")
            return list(caps or [])
        except Exception:
            return []

    # ------------------------------------------------------------- Embeddings
    def embedding_catalog(self):
        """Full embedding catalog with live availability. Merges in any pulled
        Ollama model that advertises the 'embedding' capability."""
        pulled = self.pulled_models()
        catalog = list(_OLLAMA_EMBED)
        known_names = {m.model_name for m in _OLLAMA_EMBED}

        # Discover extra embedding models the user has pulled.
        for name in sorted(pulled):
            if name in known_names or name.endswith(":latest") and name[:-7] in known_names:
                continue
            if "embed" in name.lower():
                caps_ok = True
            else:
                caps_ok = "embedding" in self._ollama_capabilities(name)
            if caps_ok and not any(m.model_name == name for m in catalog):
                catalog.append(EmbeddingModel(
                    f"ollama/{name}", "ollama", name,
                    label=name, notes="Discovered local embedding model"))
                known_names.add(name)

        catalog.extend(_VOYAGE_EMBED)
        return catalog

    def get_embedding(self, model_id):
        """Return an EmbeddingModel for an id, constructing one on the fly for
        unknown ollama models."""
        if model_id in self._embed_by_id:
            return self._embed_by_id[model_id]
        # Dynamically discovered / arbitrary ollama model.
        if model_id.startswith("ollama/"):
            name = model_id[len("ollama/"):]
            em = EmbeddingModel(model_id, "ollama", name, label=name,
                                notes="Local embedding model")
            self._embed_by_id[model_id] = em
            return em
        if model_id.startswith("voyage/"):
            name = model_id[len("voyage/"):]
            em = EmbeddingModel(model_id, "voyage", name, input_type_aware=True,
                                batch_size=96, label=name)
            self._embed_by_id[model_id] = em
            return em
        return None

    def active_embedding(self):
        em = self.get_embedding(self.active_embedding_id)
        if em is None:
            # Fall back to the configured default spec.
            em = _OLLAMA_EMBED[1]
        return em

    def embedding_available(self, em):
        if em.provider == "voyage":
            return self.voyage_ok()
        return self._ollama_has(em.model_name)

    def resolve_startup_embedding(self):
        """Pick a sensible active embedding model at boot: honor the configured
        one if it's available, otherwise auto-select the best already-pulled
        local model so the app works out of the box. Returns (id, auto_fallback)."""
        configured = self.get_embedding(self.active_embedding_id)
        if configured and self.embedding_available(configured):
            return self.active_embedding_id, False

        # Prefer a code-oriented local model, then any available embedder.
        preference = [
            "ollama/qwen3-embedding:4b", "ollama/qwen3-embedding:0.6b",
            "ollama/qwen3-embedding:8b", "ollama/nomic-embed-text",
            "ollama/mxbai-embed-large", "ollama/embeddinggemma",
        ]
        catalog = {m.id: m for m in self.embedding_catalog()}
        for pid in preference:
            m = catalog.get(pid)
            if m and self.embedding_available(m):
                self.active_embedding_id = pid
                return pid, (pid != self.active_embedding_id)
        for m in self.embedding_catalog():
            if self.embedding_available(m):
                self.active_embedding_id = m.id
                return m.id, True
        # Nothing available; keep configured (health banner will prompt a pull).
        return self.active_embedding_id, False

    def probe_dims(self, em):
        """Determine the vector dimension of an embedding model (cached)."""
        if em.id in self._dims_cache:
            return self._dims_cache[em.id]
        with self._lock:
            if em.id in self._dims_cache:
                return self._dims_cache[em.id]
            vecs = self.embed(["dimension probe"], em, kind="doc")
            dims = len(vecs[0])
            self._dims_cache[em.id] = dims
            em.dims = dims
            return dims

    def embed(self, texts, em, kind="doc"):
        """Embed a list of texts with retries + batching. kind is 'doc' or 'query'."""
        if not texts:
            return []
        if em.provider == "voyage":
            return self._embed_voyage(texts, em, kind)
        return self._embed_ollama(texts, em, kind)

    def _with_prefix(self, texts, em, kind):
        prefix = em.query_prefix if kind == "query" else em.doc_prefix
        if prefix:
            return [f"{prefix}{t}" for t in texts]
        return texts

    def _embed_ollama(self, texts, em, kind):
        if self.ollama is None:
            raise RuntimeError("Ollama client is not available.")
        inputs = self._with_prefix(texts, em, kind)
        out = []
        bs = max(1, em.batch_size)
        for i in range(0, len(inputs), bs):
            batch = inputs[i:i + bs]
            vecs = self._retry(lambda: self._ollama_embed_call(em.model_name, batch),
                               what=f"ollama embed {em.model_name}")
            out.extend(vecs)
        return out

    def _ollama_embed_call(self, model_name, batch):
        resp = self.ollama.embed(model=model_name, input=batch)
        vecs = getattr(resp, "embeddings", None)
        if vecs is None and isinstance(resp, dict):
            vecs = resp.get("embeddings")
        if not vecs:
            raise RuntimeError("Ollama returned no embeddings.")
        return [list(v) for v in vecs]

    def _embed_voyage(self, texts, em, kind):
        if self.voyage is None:
            raise RuntimeError("Voyage client is not configured (set VOYAGE_API_KEY).")
        input_type = None
        if em.input_type_aware:
            input_type = "query" if kind == "query" else "document"
        out = []
        bs = max(1, em.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            res = self._retry(
                lambda: self.voyage.embed(batch, model=em.model_name, input_type=input_type),
                what=f"voyage embed {em.model_name}")
            out.extend([list(v) for v in res.embeddings])
        return out

    # ------------------------------------------------------------------- Chat
    def chat_model_catalog(self):
        """Suggested chat models + any other pulled Ollama model, with availability."""
        pulled = self.pulled_models()
        catalog = []
        seen = set()
        for item in _OLLAMA_CHAT_SUGGESTED:
            catalog.append({**item, "provider": "ollama",
                            "available": self._ollama_has(item["id"])})
            seen.add(item["id"])
        for name in sorted(pulled):
            if name in seen or name.endswith(":latest") and name[:-7] in seen:
                continue
            if "embed" in name.lower():
                continue
            catalog.append({"id": name, "label": name, "provider": "ollama",
                            "available": True, "notes": "Pulled local model"})
            seen.add(name)
        if self.config.azure_enabled:
            aid = f"azure/{self.config.azure_deployment}"
            catalog.append({"id": aid, "label": f"Azure {self.config.azure_deployment}",
                            "provider": "azure", "available": True,
                            "notes": "Azure OpenAI deployment"})
        return catalog

    def chat(self, messages, model=None, schema=None, think=False,
             temperature=0.0, num_ctx=None):
        """Chat completion. `schema` may be a JSON-schema dict (or pydantic model
        class) to request structured output. Returns the response text."""
        model = model or self.active_llm
        if isinstance(schema, type):  # a pydantic model class
            try:
                schema = schema.model_json_schema()
            except Exception:
                schema = None

        if model.startswith("azure/"):
            return self._chat_azure(messages, schema, temperature)
        return self._chat_ollama(messages, model, schema, think, temperature, num_ctx)

    def _chat_ollama(self, messages, model, schema, think, temperature, num_ctx):
        if self.ollama is None:
            raise RuntimeError("Ollama client is not available.")
        options = {"temperature": temperature}
        options["num_ctx"] = min(num_ctx or self.config.num_ctx_cap, self.config.num_ctx_cap)
        fmt = schema if schema else None

        def _call():
            kwargs = dict(model=model, messages=messages, options=options, keep_alive="30m")
            if fmt is not None:
                kwargs["format"] = fmt
            # think is only meaningful for reasoning models; pass through.
            try:
                return self.ollama.chat(think=think, **kwargs)
            except TypeError:
                return self.ollama.chat(**kwargs)

        resp = self._retry(_call, what=f"ollama chat {model}")
        msg = getattr(resp, "message", None)
        if msg is None and isinstance(resp, dict):
            msg = resp.get("message")
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return (content or "").strip()

    def _chat_azure(self, messages, schema, temperature):
        if self.azure is None:
            raise RuntimeError("Azure client not configured.")
        response_format = {"type": "json_object"} if schema else None
        resp = self._retry(lambda: self.azure.chat.completions.create(
            model=self.config.azure_deployment,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        ), what="azure chat")
        return resp.choices[0].message.content.strip()

    # ---------------------------------------------------------------- Rerank
    def rerank(self, query, documents, top_k=None):
        """Return a list of (original_index, score) sorted best-first, or None if
        no reranker is available (caller should fall back to fusion order)."""
        if not documents:
            return []
        if self.voyage is not None:
            try:
                res = self.voyage.rerank(query, documents, model="rerank-2.5",
                                         top_k=top_k or len(documents))
                return [(r.index, r.relevance_score) for r in res.results]
            except Exception as e:
                logger.warning(f"Voyage rerank failed, falling back: {e}")
        return None

    # ---------------------------------------------------------------- Voyage
    def voyage_ok(self):
        if not self.config.voyage_enabled or self.voyage is None:
            return False
        if self._voyage_ok is not None:
            return self._voyage_ok
        try:
            self.voyage.embed(["ping"], model="voyage-3.5-lite", input_type="document")
            self._voyage_ok = True
        except Exception as e:
            logger.warning(f"Voyage key validation failed: {e}")
            self._voyage_ok = False
        return self._voyage_ok

    # ------------------------------------------------------------------- Pull
    def pull_stream(self, model_name):
        """Yield progress dicts while pulling an Ollama model."""
        if self.ollama is None:
            yield {"status": "error", "error": "Ollama is not available."}
            return
        try:
            for progress in self.ollama.pull(model_name, stream=True):
                if hasattr(progress, "model_dump"):
                    yield progress.model_dump()
                elif isinstance(progress, dict):
                    yield progress
                else:
                    yield {"status": str(progress)}
            self.pulled_models(force=True)
        except Exception as e:
            yield {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------ Utils
    def _retry(self, fn, attempts=3, base_delay=0.8, what="operation"):
        last = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                last = e
                wait = base_delay * (2 ** i)
                logger.warning(f"{what} failed (attempt {i + 1}/{attempts}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        raise last

    def health(self):
        """Snapshot of provider + model availability for the UI banner."""
        active_em = self.active_embedding()
        ollama_up = self.ollama_up()
        return {
            "ollama": {
                "host": self.config.ollama_host,
                "up": ollama_up,
            },
            "voyage": {
                "configured": self.config.voyage_enabled,
                "ok": self.voyage_ok() if self.config.voyage_enabled else False,
            },
            "azure": {"configured": self.config.azure_enabled},
            "active": {
                "embedding": active_em.id,
                "embedding_available": self.embedding_available(active_em),
                "embedding_dims": active_em.dims,
                "llm": self.active_llm,
                "llm_available": (self._ollama_has(self.active_llm)
                                  if not self.active_llm.startswith("azure/") else True),
                "fast_llm": self.active_fast_llm,
                "fast_llm_available": (self._ollama_has(self.active_fast_llm)
                                       if not self.active_fast_llm.startswith("azure/") else True),
            },
        }
