from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "../../data/knowledge")

# Local model path — baked into the Docker image at build time (see Dockerfile).
# Falls back to downloading from HuggingFace Hub if the local path doesn't exist.
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/all-MiniLM-L6-v2")


def _load_documents_from_folder(folder: str) -> List[Document]:
    """
    Load all .txt files from `folder`.

    Each file may start with optional key: value metadata lines followed by
    a blank line, then the document body. Example:

        source: fraud_policy
        type: merchant_abuse
        version: 2.1

        Merchant Abuse Policy: ...

    Files without a blank-line separator are treated as body-only with
    metadata derived from the filename.
    """
    docs: List[Document] = []
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return docs

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(folder, fname)
        with open(fpath, encoding="utf-8") as f:
            raw = f.read()

        metadata: Dict[str, Any] = {"filename": fname}
        body = raw.strip()

        # Parse leading key: value lines separated by a blank line
        if "\n\n" in raw:
            header, rest = raw.split("\n\n", 1)
            parsed: Dict[str, str] = {}
            for line in header.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    parsed[k.strip()] = v.strip()
            if parsed:
                metadata.update(parsed)
                body = rest.strip()

        docs.append(Document(page_content=body, metadata=metadata))

    return docs


class FraudKnowledgeBase:
    def __init__(
        self,
        persist_path: str = "./data/vector_store",
        knowledge_dir: str = KNOWLEDGE_DIR,
    ):
        self.persist_path = persist_path
        self.knowledge_dir = knowledge_dir
        local_path = os.path.abspath(_LOCAL_MODEL_PATH)
        model_name = local_path if os.path.isdir(local_path) else _MODEL_NAME
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )
        self.vector_store: Optional[FAISS] = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        index_file = os.path.join(self.persist_path, "index.faiss")
        if os.path.exists(index_file):
            self.vector_store = FAISS.load_local(
                self.persist_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            docs = _load_documents_from_folder(self.knowledge_dir)
            if not docs:
                raise FileNotFoundError(
                    f"No .txt files found in {self.knowledge_dir}. "
                    "Add documents there before running."
                )
            os.makedirs(self.persist_path, exist_ok=True)
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(self.persist_path)

    def rebuild(self) -> int:
        """Re-read all .txt files and rebuild the vector store from scratch."""
        docs = _load_documents_from_folder(self.knowledge_dir)
        if not docs:
            raise FileNotFoundError(f"No .txt files found in {self.knowledge_dir}.")
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.persist_path)
        return len(docs)

    def add_documents(self, documents: List[Document]) -> None:
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        self.vector_store.save_local(self.persist_path)

    def ingest_text(self, text: str, metadata: Dict[str, Any]) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents([text], metadatas=[metadata])
        self.add_documents(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": round(float(score), 4),
            }
            for doc, score in results
        ]
