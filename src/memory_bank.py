"""
Memory Bank: Stores and retrieves memory items
"""
import numpy as np
import math
import re
from typing import List, Dict, Tuple, Optional, Union
import copy

# Retriever embedding dimensions
RETRIEVER_EMBEDDING_DIMS = {
    'contriever': 768,
    'dpr': 768,
    'dragon': 1024,
    # Add more retrievers as needed
}


def get_retriever_embedding_dim(retriever_name: str) -> int:
    """Get embedding dimension for a retriever, with fallback"""
    return RETRIEVER_EMBEDDING_DIMS.get(retriever_name, 768)


class MemoryItem:
    """Single memory item with dual embeddings (retriever + state encoder)."""
    def __init__(self, content: str, embedding: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None,
                 state_encoder_embedding: Optional[np.ndarray] = None):
        self.content = content
        self.embedding = embedding  # Retriever embedding (for QA evaluation)
        self.state_encoder_embedding = state_encoder_embedding
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = 0  # timestep
        self.created_at = 0
        # History tracking
        self.content_history: List[str] = []  # Previous contents before updates (chronological order)
        self.operation_history: List[str] = []  # Sequence of operations applied (e.g., ['insert', 'update', 'update'])

    def to_dict(self):
        return {
            'content': self.content,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'state_encoder_embedding': (
                self.state_encoder_embedding.tolist()
                if self.state_encoder_embedding is not None else None
            ),
            'metadata': self.metadata,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'created_at': self.created_at,
            'content_history': self.content_history,
            'operation_history': self.operation_history
        }

    @classmethod
    def from_dict(cls, data):
        state_emb = data.get('state_encoder_embedding')
        if state_emb is None:
            for key, value in data.items():
                if key.endswith('_embedding') and key not in ('embedding', 'state_encoder_embedding'):
                    state_emb = value
                    break
        item = cls(
            content=data['content'],
            embedding=np.array(data['embedding']) if data.get('embedding') is not None else None,
            metadata=data.get('metadata', {}),
            state_encoder_embedding=(
                np.array(state_emb) if state_emb is not None else None
            )
        )
        item.access_count = data.get('access_count', 0)
        item.last_accessed = data.get('last_accessed', 0)
        item.created_at = data.get('created_at', 0)
        item.content_history = data.get('content_history', [])
        item.operation_history = data.get('operation_history', [])
        return item


class MemoryBank:
    """
    Memory Bank stores memory items and supports retrieval via semantic search.
    Supports dual embeddings: retriever (for QA eval) and state encoder (for training).
    """
    def __init__(self, retriever_name: str = 'contriever', top_k: int = 5,
                 state_encoder=None):
        self.memories: List[MemoryItem] = []
        self.retriever_name = retriever_name
        self.top_k = top_k
        self.timestep = 0
        self._faiss_index = None
        self._state_encoder = state_encoder  # StateEncoder for computing state embeddings

    @property
    def state_encoder(self):
        return self._state_encoder

    @state_encoder.setter
    def state_encoder(self, encoder):
        self._state_encoder = encoder

    def set_state_encoder(self, encoder):
        """Set the state encoder for computing embeddings."""
        self.state_encoder = encoder

    def _compute_state_encoder_embedding(self, content: str) -> Optional[np.ndarray]:
        """Compute state encoder embedding for a single content string."""
        if self.state_encoder is None:
            return None
        emb = self.state_encoder._encode_texts(content)
        if emb.ndim == 2:
            return emb[0]
        return emb

    def initialize_from_sessions(self, sessions: List[str], embeddings: np.ndarray,
                                  state_encoder_embeddings: Optional[np.ndarray] = None):
        """Initialize memory bank from conversation sessions/turns."""
        assert len(sessions) == embeddings.shape[0], "Sessions and embeddings length mismatch"

        self.memories = []
        for i, (session, emb) in enumerate(zip(sessions, embeddings)):
            state_emb = state_encoder_embeddings[i] if state_encoder_embeddings is not None else None
            memory_item = MemoryItem(
                content=session,
                embedding=emb,
                state_encoder_embedding=state_emb,
                metadata={'source': 'initial', 'session_idx': i}
            )
            memory_item.created_at = self.timestep
            self.memories.append(memory_item)

        self._rebuild_index()

    def add_memory(self, content: str, embedding: np.ndarray,
                   metadata: Optional[Dict] = None,
                   operation_name: Optional[Union[str, List[str]]] = None,
                   state_encoder_embedding: Optional[np.ndarray] = None):
        """Add a new memory item with optional state encoder embedding.

        Args:
            content: Memory content text
            embedding: Retriever embedding
            metadata: Optional metadata dict
            operation_name: Operation name or list of names that created this memory
        """
        # Compute state encoder embedding if encoder is available and not provided
        if state_encoder_embedding is None and self.state_encoder is not None:
            state_encoder_embedding = self._compute_state_encoder_embedding(content)

        memory_item = MemoryItem(
            content=content,
            embedding=embedding,
            state_encoder_embedding=state_encoder_embedding,
            metadata=metadata
        )
        # print(memory_item.content)
        memory_item.created_at = self.timestep
        # Record the operation that created this memory
        if operation_name:
            if isinstance(operation_name, (list, tuple, set)):
                for name in operation_name:
                    if name:
                        memory_item.operation_history.append(str(name))
            else:
                memory_item.operation_history.append(str(operation_name))
        self.memories.append(memory_item)
        self._rebuild_index()

    def update_memory(self, index: int, new_content: str, new_embedding: np.ndarray,
                      operation_name: Optional[Union[str, List[str]]] = None,
                      new_state_encoder_embedding: Optional[np.ndarray] = None):
        """Update an existing memory item.

        Args:
            index: Index of the memory to update
            new_content: New content text
            new_embedding: New retriever embedding
            operation_name: Operation name or list of names that performed this update
        """
        if 0 <= index < len(self.memories):
            mem = self.memories[index]
            # Save old content to history before updating
            mem.content_history.append(mem.content)
            # Update content and embeddings
            mem.content = new_content
            mem.embedding = new_embedding
            # Compute state encoder embedding if encoder is available and not provided
            if new_state_encoder_embedding is None and self.state_encoder is not None:
                new_state_encoder_embedding = self._compute_state_encoder_embedding(new_content)
            mem.state_encoder_embedding = new_state_encoder_embedding
            mem.last_accessed = self.timestep
            # Record the operation
            if operation_name:
                if isinstance(operation_name, (list, tuple, set)):
                    for name in operation_name:
                        if name:
                            mem.operation_history.append(str(name))
                else:
                    mem.operation_history.append(str(operation_name))
            self._rebuild_index()
        else:
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")

    def delete_memory(self, index: int):
        """Delete a memory item"""
        if 0 <= index < len(self.memories):
            del self.memories[index]
            self._rebuild_index()
        else:
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")

    def retrieve(self, query_embedding: np.ndarray, top_k: Optional[int] = None,
                 use_state_encoder: bool = False, return_embeddings: bool = False) -> Union[
                     Tuple[List[str], List[int]],
                     Tuple[List[str], List[int], Optional[np.ndarray]]
                 ]:
        """
        Retrieve top-k most relevant memories

        Args:
            query_embedding: Query embedding vector
            top_k: Number of memories to retrieve (default: self.top_k)
            use_state_encoder: If True, use state encoder embeddings for retrieval (training).
                              If False, use retriever embeddings (QA evaluation).
            return_embeddings: If True, also return embeddings aligned with retrieved memories.

        Returns:
            (memory_contents, memory_indices) or
            (memory_contents, memory_indices, memory_embeddings) if return_embeddings=True
        """
        if top_k is None:
            top_k = self.top_k

        if len(self.memories) == 0:
            if return_embeddings:
                return [], [], None
            return [], []

        # Get embeddings matrix based on embedding type
        embeddings = self._get_embeddings_matrix(use_state_encoder=use_state_encoder)
        embeddings_for_sim = embeddings

        if use_state_encoder or self.retriever_name == 'dragon':
            emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_for_sim = embeddings / (emb_norms + 1e-8)

        # Compute similarities (normalize embeddings only for similarity)
        if isinstance(query_embedding, np.ndarray):
            q = query_embedding.reshape(1, -1).astype('float32')
        else:
            q = np.asarray(query_embedding, dtype='float32').reshape(1, -1)

        # Normalize query
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

        # Compute similarities
        similarities = np.dot(q_norm, embeddings_for_sim.T)[0]

        # Get top-k indices
        actual_k = min(top_k, len(self.memories))
        top_indices = np.argsort(similarities)[::-1][:actual_k]

        # Update access statistics
        for idx in top_indices:
            self.memories[idx].access_count += 1
            self.memories[idx].last_accessed = self.timestep

        # Return contents and indices
        contents = [self.memories[idx].content for idx in top_indices]
        if return_embeddings:
            if len(top_indices) > 0:
                retrieved_embeddings = embeddings[top_indices].copy()
            else:
                retrieved_embeddings = np.zeros((0, embeddings.shape[1]), dtype=np.float32)
            return contents, top_indices.tolist(), retrieved_embeddings
        return contents, top_indices.tolist()

    def retrieve_hybrid_text(self, query_text: str, query_embedding: np.ndarray,
                             top_k: Optional[int] = None,
                             use_state_encoder: bool = False,
                             return_embeddings: bool = False,
                             semantic_weight: float = 0.55,
                             lexical_weight: float = 0.45) -> Union[
                                 Tuple[List[str], List[int]],
                                 Tuple[List[str], List[int], Optional[np.ndarray]]
                             ]:
        """Retrieve with a blend of embedding similarity and lexical overlap.

        LoCoMo questions often hinge on exact entities, activities, and dates. A pure
        embedding ranker can miss those sparse cues, so this keeps semantic retrieval
        but adds a lightweight BM25-style text score over memory contents.
        """
        if top_k is None:
            top_k = self.top_k

        if len(self.memories) == 0:
            if return_embeddings:
                return [], [], None
            return [], []

        embeddings = self._get_embeddings_matrix(use_state_encoder=use_state_encoder)
        embeddings_for_sim = embeddings
        if use_state_encoder or self.retriever_name == 'dragon':
            emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_for_sim = embeddings / (emb_norms + 1e-8)

        q = np.asarray(query_embedding, dtype='float32').reshape(1, -1)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        semantic = np.dot(q_norm, embeddings_for_sim.T)[0]
        semantic = self._minmax(semantic)

        lexical = self._lexical_scores(query_text)
        lexical = self._minmax(lexical)

        semantic_weight = max(0.0, float(semantic_weight))
        lexical_weight = max(0.0, float(lexical_weight))
        total_weight = semantic_weight + lexical_weight
        if total_weight <= 0:
            total_weight = 1.0
            semantic_weight = 1.0
        combined = (
            (semantic_weight / total_weight) * semantic
            + (lexical_weight / total_weight) * lexical
        )

        actual_k = min(top_k, len(self.memories))
        top_indices = np.argsort(combined)[::-1][:actual_k]

        for idx in top_indices:
            self.memories[idx].access_count += 1
            self.memories[idx].last_accessed = self.timestep

        contents = [self.memories[idx].content for idx in top_indices]
        if return_embeddings:
            retrieved_embeddings = embeddings[top_indices].copy() if len(top_indices) > 0 else None
            return contents, top_indices.tolist(), retrieved_embeddings
        return contents, top_indices.tolist()

    def _lexical_scores(self, query_text: str) -> np.ndarray:
        query_terms = self._tokenize_for_retrieval(query_text)
        if not query_terms:
            return np.zeros(len(self.memories), dtype=np.float32)

        docs = [self._tokenize_for_retrieval(mem.content) for mem in self.memories]
        n_docs = max(len(docs), 1)
        doc_lens = np.array([max(len(doc), 1) for doc in docs], dtype=np.float32)
        avgdl = float(doc_lens.mean()) if len(doc_lens) else 1.0

        df = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        q_terms = list(dict.fromkeys(query_terms))
        scores = np.zeros(len(docs), dtype=np.float32)
        k1 = 1.5
        b = 0.75
        for idx, doc in enumerate(docs):
            if not doc:
                continue
            counts = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            dl = float(doc_lens[idx])
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                term_df = df.get(term, 0)
                idf = math.log(1.0 + (n_docs - term_df + 0.5) / (term_df + 0.5))
                denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6))
                score += idf * (tf * (k1 + 1.0) / max(denom, 1e-6))
            scores[idx] = score
        return scores

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        lo = float(arr.min())
        hi = float(arr.max())
        if hi - lo < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _tokenize_for_retrieval(text: str) -> List[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
            "does", "for", "from", "had", "has", "have", "he", "her", "his",
            "how", "i", "in", "is", "it", "its", "me", "of", "on", "or",
            "she", "the", "their", "them", "they", "to", "was", "were",
            "what", "when", "where", "which", "who", "whom", "why", "with",
            "you", "your"
        }
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        return [tok for tok in tokens if len(tok) > 1 and tok not in stopwords]

    def _get_embeddings_matrix(self, use_state_encoder: bool = False) -> np.ndarray:
        """
        Get embeddings as a matrix [N, D]

        Args:
            use_state_encoder: If True, use state encoder embeddings. Otherwise use retriever embeddings.
        """
        embeddings = []
        embedding_dim = None

        # First pass: collect embeddings and determine dimension
        for mem in self.memories:
            if use_state_encoder:
                emb = mem.state_encoder_embedding
            else:
                emb = mem.embedding

            if emb is not None:
                embeddings.append(emb)
                if embedding_dim is None:
                    embedding_dim = emb.shape[0]
            else:
                embeddings.append(None)  # Placeholder for missing embeddings

        # Fallback dimension based on retriever if no embeddings found
        if embedding_dim is None:
            if use_state_encoder:
                if self.state_encoder is not None:
                    embedding_dim = getattr(self.state_encoder, 'embedding_dim', None)
                if embedding_dim is None:
                    embedding_dim = 768  # Default state encoder hidden size
            else:
                embedding_dim = get_retriever_embedding_dim(self.retriever_name)

        # Second pass: replace None with zero vectors of correct dimension
        for i, emb in enumerate(embeddings):
            if emb is None:
                embeddings[i] = np.zeros(embedding_dim, dtype=np.float32)

        return np.vstack(embeddings).astype('float32')

    def _rebuild_index(self):
        """Rebuild FAISS index if needed"""
        # Optional: build FAISS index for faster retrieval
        # For now, we use simple numpy operations which are sufficient for moderate sizes
        self._faiss_index = None

    def get_all_contents(self) -> List[str]:
        """Get all memory contents"""
        return [mem.content for mem in self.memories]

    def get_memory_at(self, index: int) -> MemoryItem:
        """Get memory item at specific index"""
        if 0 <= index < len(self.memories):
            return self.memories[index]
        else:
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")

    def __len__(self):
        return len(self.memories)

    def step(self):
        """Increment timestep"""
        self.timestep += 1

    def to_dict(self):
        """Serialize to dict"""
        return {
            'memories': [mem.to_dict() for mem in self.memories],
            'retriever_name': self.retriever_name,
            'top_k': self.top_k,
            'timestep': self.timestep
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict"""
        bank = cls(
            retriever_name=data.get('retriever_name', 'contriever'),
            top_k=data.get('top_k', 5)
        )
        bank.memories = [MemoryItem.from_dict(m) for m in data.get('memories', [])]
        bank.timestep = data.get('timestep', 0)
        bank._rebuild_index()
        return bank

    def copy(self):
        """Create a deep copy of the memory bank"""
        return copy.deepcopy(self)
