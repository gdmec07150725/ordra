"""本地中文 Embedding：CPU 推理、有限缓存和失败冷却，不发送聊天内容到外部服务。"""

from collections import OrderedDict
from hashlib import sha256
import logging
from pathlib import Path
from threading import RLock
from time import monotonic


class LocalEmbeddings:
    model_name = "BAAI/bge-small-zh-v1.5"

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.status = "not_loaded"
        self._model = None
        self._retry_after = 0.0
        self._lock = RLock()
        # 只缓存文本哈希到向量，没有会话、位置或原文；候选始终由当前数据库提供。
        self._vectors = OrderedDict()

    def _load_model(self):
        from fastembed import TextEmbedding
        return TextEmbedding(self.model_name, cache_dir=str(self.cache_dir), threads=2,
                             providers=["CPUExecutionProvider"], local_files_only=True)

    def warm_up(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if monotonic() < self._retry_after:
                return False
            try:
                self._model = self._load_model()
                self.status = "ready"
                return True
            except Exception as error:
                self._failed(error)
                return False

    def _failed(self, error):
        self._model = None
        self._vectors.clear()
        self.status = "unavailable"
        self._retry_after = monotonic() + 60
        logging.getLogger(__name__).warning("本地 Embedding 暂不可用：%s", type(error).__name__)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        """计算余弦相似度；锁避免多个会话同时加载模型或修改 LRU。"""
        if not texts:
            return []
        with self._lock:
            if not self.warm_up():
                raise RuntimeError("本地 Embedding 暂不可用。")
            try:
                import numpy as np

                def normalized(vector):
                    array = np.asarray(vector, dtype=np.float32)
                    norm = np.linalg.norm(array)
                    if array.ndim != 1 or not np.isfinite(array).all() or not np.isfinite(norm) or norm <= 0:
                        raise ValueError("Embedding 向量无效。")
                    return array / norm

                # v1.5 支持无前缀；本项目短对话样例中直接编码的召回效果更好。
                query_vector = normalized(next(self._model.query_embed(query)))
                keys = [sha256(text.encode("utf-8")).digest() for text in texts]
                by_key = {}
                # 先消费本次所有缓存命中，避免顺序扫描把尚未使用的缓存逐出。
                for key in dict.fromkeys(keys):
                    if key in self._vectors:
                        by_key[key] = float(np.dot(query_vector, self._vectors[key]))
                        self._vectors.move_to_end(key)
                missing = list({key: text for key, text in zip(keys, texts) if key not in by_key}.items())
                for start in range(0, len(missing), 32):
                    batch = missing[start:start + 32]
                    vectors = self._model.passage_embed([text for _, text in batch], batch_size=32)
                    for (key, _), vector in zip(batch, vectors, strict=True):
                        self._vectors[key] = normalized(vector)
                        by_key[key] = float(np.dot(query_vector, self._vectors[key]))
                    while len(self._vectors) > 2048:
                        self._vectors.popitem(last=False)
                return [by_key[key] for key in keys]
            except Exception as error:
                self._failed(error)
                raise


if __name__ == "__main__":
    # 显式安装步骤才联网；聊天服务启动和推理只读取本机文件。
    from huggingface_hub import snapshot_download
    model_cache = Path(__file__).resolve().parent / ".local" / "embeddings"
    snapshot_download("Qdrant/bge-small-zh-v1.5", cache_dir=model_cache,
                      allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "model_optimized.onnx"])
    if not LocalEmbeddings(model_cache).warm_up():
        raise SystemExit("模型未能加载，请检查本机依赖与下载文件。")
    print("中文 Embedding 模型已就绪，聊天时可离线生成向量。")
