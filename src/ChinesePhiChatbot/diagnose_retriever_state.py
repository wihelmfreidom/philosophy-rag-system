import os
import gc
import asyncio
import re
from typing import List, Dict, Any, Tuple

# ========== 环境配置 ==========
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["OMP_NUM_THREADS"] = "4"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

print("🔍 启动哲学史问答系统（直接加载现有索引）...")
print("=" * 60)


# ========== 查询嵌入模型（与原代码一致） ==========
class QueryEmbeddings:
    def __init__(self, model_path):
        print(f"🔄 加载查询模型: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


query_embeddings = QueryEmbeddings("D:/models/bge-small-zh-v1.5")


# ========== 改进后的父子检索器（支持从子块恢复父块缓存） ==========
class ParentChildRetriever:
    """
    简化版父子检索器，支持从已存储的子块元数据中重建父块缓存
    """

    def __init__(self, vectorstore, top_k: int = 3):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.parent_cache = {}  # 父块缓存，键为 parent_id

    def invoke(self, query: str) -> List[Document]:
        """
        检索流程：子块匹配 → 获取父块 → 去重 → 返回完整父块
        如果缓存中缺失父块，则从子块元数据中提取并填充缓存
        """
        # 1. 检索子块
        child_docs = self.vectorstore.similarity_search(query, k=self.top_k * 2)

        # 2. 按父块ID聚合，去重
        seen_parents = set()
        parent_docs = []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if not parent_id or parent_id in seen_parents:
                continue

            # 如果缓存中没有该父块，尝试从当前子块元数据构建
            if parent_id not in self.parent_cache:
                # 从子块元数据中提取父块信息（子块存储了父块完整内容）
                parent_full_content = child.metadata.get("parent_full_content")
                if parent_full_content is None:
                    continue  # 缺少必要信息，跳过

                # 构建父块缓存条目
                self.parent_cache[parent_id] = {
                    "title": child.metadata.get("chapter_title", ""),
                    "content": parent_full_content,
                    "metadata": {
                        k: v
                        for k, v in child.metadata.items()
                        if k
                        not in [
                            "child_id",
                            "chunk_index",
                            "is_child",
                            "parent_full_content",
                        ]
                    },
                }

            # 从缓存获取父块信息
            parent_info = self.parent_cache[parent_id]
            seen_parents.add(parent_id)

            # 构建父块文档（用于生成回答）
            parent_doc = Document(
                page_content=parent_info["content"],
                metadata={
                    **parent_info["metadata"],
                    "retrieved_child": child.page_content,  # 记录触发命中的子块片段
                    "child_similarity_source": True,
                },
            )
            parent_docs.append(parent_doc)

            if len(parent_docs) >= self.top_k:
                break

        print(
            f"🔍 检索: '{query[:30]}...' → 命中 {len(child_docs)} 个子块，返回 {len(parent_docs)} 个父块"
        )
        for i, p in enumerate(parent_docs, 1):
            print(f"   [{i}] {p.metadata.get('chapter_title', '未知')}")

        return parent_docs

    def ainvoke(self, query: str):
        """异步版本（与原代码一致）"""
        return self.invoke(query)


# ========== 初始化向量库和检索器（仅加载现有数据） ==========
PERSIST_DIR = "D:/chroma_db/philosophy_history"

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=query_embeddings,
    collection_name="philosophy_parent_child",
)

retriever = ParentChildRetriever(vectorstore, top_k=3)


def diagnose_retriever_state(retriever, vectorstore):
    """诊断 parent_cache 与向量库的同步状态"""
    print("=" * 60)
    print("🔍 诊断报告")
    print("=" * 60)

    # 1. 检查 parent_cache 状态
    cache_ids = set(retriever.parent_cache.keys())
    print(f"\n📦 Parent Cache 状态:")
    print(f"   缓存父块数量: {len(cache_ids)}")
    if len(cache_ids) > 0:
        print(f"   示例ID: {list(cache_ids)[:3]}")

    # 2. 检查向量库中的子块
    # 获取所有文档（Chroma 的 get() 方法）
    all_docs = vectorstore.get()
    if not all_docs or "metadatas" not in all_docs:
        print("\n⚠️ 向量库为空或无法读取")
        return

    metadatas = all_docs["metadatas"]
    child_parent_ids = [m.get("parent_id") for m in metadatas if m.get("is_child")]
    unique_child_parents = set(child_parent_ids)

    print(f"\n📚 向量库状态:")
    print(f"   子块总数: {len(child_parent_ids)}")
    print(f"   涉及父块ID数: {len(unique_child_parents)}")
    if len(unique_child_parents) > 0:
        print(f"   示例Parent ID: {list(unique_child_parents)[:3]}")

    # 3. 对比分析
    print(f"\n🔍 一致性检查:")

    # 在cache但无子块引用
    orphan_cache = cache_ids - unique_child_parents
    if orphan_cache:
        print(f"   ⚠️ 孤缓存（有父块无子块）: {len(orphan_cache)} 个")
        print(f"      示例: {list(orphan_cache)[:2]}")

    # 有子块但无cache
    missing_cache = unique_child_parents - cache_ids
    if missing_cache:
        print(f"   ❌ 缺失缓存（有子块无父块）: {len(missing_cache)} 个")
        print(f"      示例: {list(missing_cache)[:2]}")
        print(f"      这就是你'命中6个子块返回0个父块'的原因！")

    # 4. 抽样检查内容匹配度
    print(f"\n📝 内容抽样检查:")
    sample_docs = vectorstore.similarity_search("独化", k=3)
    for i, doc in enumerate(sample_docs, 1):
        pid = doc.metadata.get("parent_id", "N/A")
        has_content = (
            "parent_content" in doc.metadata or "parent_full_content" in doc.metadata
        )
        print(f"   [{i}] Parent ID: {pid}")
        print(f"       自带父块内容: {has_content}")
        print(f"       文本片段: {doc.page_content[:40]}...")


# 使用方法：
diagnose_retriever_state(retriever, vectorstore)
