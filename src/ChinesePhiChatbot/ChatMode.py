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
        检索流程：子块匹配 → 从子块metadata直接提取父块（不依赖缓存）
        """
        # 1. 检索子块
        child_docs = self.vectorstore.similarity_search(query, k=self.top_k * 3)

        if not child_docs:
            return []

        seen_parents = set()
        parent_docs = []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if not parent_id or parent_id in seen_parents:
                continue

            seen_parents.add(parent_id)

            # 🎯 关键修复：直接从子块 metadata 读取父块内容，不查缓存！
            parent_content = child.metadata.get(
                "parent_full_content"
            ) or child.metadata.get("parent_content")
            parent_title = child.metadata.get("parent_title") or child.metadata.get(
                "chapter_title", "未知章节"
            )

            if parent_content and len(parent_content) > 50:
                # 成功获取父块内容
                parent_doc = Document(
                    page_content=parent_content,
                    metadata={
                        "chapter_title": parent_title,
                        "parent_id": parent_id,
                        "retrieved_child": child.page_content[:150],  # 显示触发片段
                        "source": child.metadata.get("source", ""),
                    },
                )
                parent_docs.append(parent_doc)
            else:
                # 万一 metadata 里没有，fallback 返回子块本身
                print(f"⚠️ Parent ID {parent_id} 无内容，使用子块")
                parent_docs.append(child)

            if len(parent_docs) >= self.top_k:
                break

        print(
            f"🔍 检索: '{query[:20]}...' → 命中 {len(child_docs)} 子块，返回 {len(parent_docs)} 父块"
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

print(f"✅ 已加载向量库，包含 {vectorstore._collection.count()} 个子块")


# ========== 配置LLM（与原代码一致，请替换为有效密钥） ==========
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

llm = ChatOpenAI(
    api_key="c8ff5d6d099742b0855805b4e59245c4.1ORR5pA9RqwR5wcq",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    model="glm-4-flash",
    temperature=0.3,
    max_tokens=2048,
    streaming=True,
)

print("🤖 LLM就绪（流式模式）")


# ========== 构建RAG链（与原代码一致） ==========
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是中国哲学史领域的资深研究者，擅长解读经典文本及其现代阐释。

基于以下从哲学史著作中检索到的完整章节内容，回答用户的问题。

回答要求：
1. 严格基于提供的文本内容，准确引用原文和作者观点坚决不可以自己捏造和胡编任何内容，
2. 注意区分：经典原文（《老子》等）、历代注疏、现代学者阐释
3. 保持哲学论证的严谨性，理清概念层次
4. 如果涉及多个章节，请分别说明不同章节的侧重点
5. 在回答末尾注明引用来源（章节标题）

检索到的相关章节：
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs: List[Document]) -> str:
    """格式化父块文档"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("chapter_title", "未知章节")
        content = doc.page_content
        if len(content) > 1500:
            content = content[:1500] + "...（后续内容省略）"

        formatted.append(f"【章节{i}：{title}】\n{content}\n")

    return "\n---\n".join(formatted)


rag_chain = (
    RunnableParallel(
        {
            "context": lambda x: format_docs(retriever.invoke(x)),
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG链构建完成，进入交互模式")
print("=" * 60)


# ========== 流式查询函数（与原代码一致） ==========
async def ask_question_stream(question: str):
    """流式执行问答"""
    print(f"\n❓ 问题: {question}")
    print("-" * 50)
    print("💡 回答: ", end="", flush=True)

    try:
        full_answer = ""
        async for chunk in rag_chain.astream(question):
            print(chunk, end="", flush=True)
            full_answer += chunk

        print("\n")

        # 显示来源详情（可选）
        source_docs = retriever.invoke(question)
        print(f"\n📚 引用来源 ({len(source_docs)} 个章节):")
        for i, doc in enumerate(source_docs, 1):
            title = doc.metadata.get("chapter_title", "?")
            trigger = doc.metadata.get("retrieved_child", "")[:40]
            print(f"   [{i}] 《{title}》")
            print(f"       触发片段: {trigger}...")

        return full_answer, source_docs

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return "", []


def ask_question_sync(question: str):
    """同步包装器"""
    return asyncio.run(ask_question_stream(question))


# ========== 交互式问答循环 ==========
if __name__ == "__main__":
    print("\n💬 输入你的问题（'退出'结束）:")
    while True:
        user_q = input("> ").strip()
        if user_q.lower() in ["退出", "quit", "q"]:
            break
        if user_q:
            ask_question_sync(user_q)
            print()

    print("\n🧹 清理资源...")
    gc.collect()
    print("✅ 完成")
