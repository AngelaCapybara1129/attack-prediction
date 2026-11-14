"""事件向量資料庫。

- 設定 MILVUS_URI（例如 http://localhost:19530 或 ./milvus_lite.db）且安裝 pymilvus 時使用 Milvus；
- 否則使用本機 numpy 向量庫（存於 backend/data/rag/），功能相同，適合單機開發與展示。
"""
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.paths import RAG_DIR

SearchHit = Tuple[str, float, Dict[str, Any]]


class VectorStore:
    backend = "base"

    def upsert(self, key: str, vector: np.ndarray, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def search(self, vector: np.ndarray, k: int = 5, kind: Optional[str] = None,
               exclude: Optional[str] = None) -> List[SearchHit]:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class LocalVectorStore(VectorStore):
    backend = "local"

    def __init__(self, dim: int, path: Optional[str] = None):
        self.dim = dim
        self.path = path or str(RAG_DIR / "vectors.json")
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._payloads: List[Dict[str, Any]] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if int(data.get("dim", self.dim)) != self.dim:
                print("[RAG] 本機向量庫維度不符，重新建立")
                return
            self._keys = list(data["keys"])
            self._payloads = list(data["payloads"])
            self._matrix = np.asarray(data["vectors"], dtype=np.float32).reshape(-1, self.dim)
        except Exception as e:
            print(f"[RAG] ⚠️ 讀取本機向量庫失敗: {e}")

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "keys": self._keys, "payloads": self._payloads,
                       "vectors": self._matrix.round(5).tolist()}, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def upsert(self, key: str, vector: np.ndarray, payload: Dict[str, Any]) -> None:
        v = _normalize(vector)
        with self._lock:
            if key in self._keys:
                i = self._keys.index(key)
                self._matrix[i] = v
                self._payloads[i] = payload
            else:
                self._keys.append(key)
                self._payloads.append(payload)
                self._matrix = np.vstack([self._matrix, v[None]])
            self._save()

    def search(self, vector: np.ndarray, k: int = 5, kind: Optional[str] = None,
               exclude: Optional[str] = None) -> List[SearchHit]:
        with self._lock:
            if self._matrix.shape[0] == 0:
                return []
            sims = self._matrix @ _normalize(vector)
            order = np.argsort(-sims)
            hits: List[SearchHit] = []
            for i in order:
                key = self._keys[i]
                if key == exclude:
                    continue
                if kind and self._payloads[i].get("kind") != kind:
                    continue
                hits.append((key, float(sims[i]), self._payloads[i]))
                if len(hits) >= k:
                    break
            return hits

    def delete(self, key: str) -> None:
        with self._lock:
            if key in self._keys:
                i = self._keys.index(key)
                self._keys.pop(i)
                self._payloads.pop(i)
                self._matrix = np.delete(self._matrix, i, axis=0)
                self._save()

    def count(self) -> int:
        return len(self._keys)


class MilvusVectorStore(VectorStore):
    backend = "milvus"

    def __init__(self, dim: int, uri: str, token: str = "", collection: str = "attack_events"):
        from pymilvus import MilvusClient  # 延遲載入：未安裝時改用本機向量庫

        self.dim = dim
        self.collection = collection
        self.client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
        if not self.client.has_collection(collection):
            self.client.create_collection(
                collection_name=collection,
                dimension=dim,
                primary_field_name="id",
                id_type="string",
                max_length=128,
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )

    def upsert(self, key: str, vector: np.ndarray, payload: Dict[str, Any]) -> None:
        self.client.upsert(
            collection_name=self.collection,
            data=[{
                "id": key,
                "vector": _normalize(vector).tolist(),
                "kind": str(payload.get("kind", "")),
                "payload": json.dumps(payload, ensure_ascii=False),
            }],
        )

    def search(self, vector: np.ndarray, k: int = 5, kind: Optional[str] = None,
               exclude: Optional[str] = None) -> List[SearchHit]:
        filters = []
        if kind:
            filters.append(f'kind == "{kind}"')
        if exclude:
            filters.append(f'id != "{exclude}"')
        res = self.client.search(
            collection_name=self.collection,
            data=[_normalize(vector).tolist()],
            limit=k,
            filter=" and ".join(filters),
            output_fields=["kind", "payload"],
        )
        hits: List[SearchHit] = []
        for item in (res[0] if res else []):
            entity = item.get("entity", {})
            try:
                payload = json.loads(entity.get("payload") or "{}")
            except Exception:
                payload = {}
            hits.append((str(item.get("id")), float(item.get("distance", 0.0)), payload))
        return hits

    def delete(self, key: str) -> None:
        self.client.delete(collection_name=self.collection, ids=[key])

    def count(self) -> int:
        try:
            stats = self.client.get_collection_stats(self.collection)
            return int(stats.get("row_count", 0))
        except Exception:
            return -1


def create_vector_store(dim: int) -> VectorStore:
    uri = os.getenv("MILVUS_URI", "").strip()
    if uri:
        try:
            store = MilvusVectorStore(dim, uri, os.getenv("MILVUS_TOKEN", "").strip())
            print(f"[RAG] ✅ 使用 Milvus 向量庫：{uri}")
            return store
        except Exception as e:
            print(f"[RAG] ⚠️ Milvus 連線失敗（{e}），改用本機向量庫")
    return LocalVectorStore(dim)
