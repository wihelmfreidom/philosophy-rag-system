
import os
import gc
from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# 环境配置（复用你的设置）
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"


class QueryEmbeddings:
    """复用你的代码"""

    def __init__(self, model_path: str = "D:/models/bge-small-zh-v1.5"):
        print(f"🔄 加载查询模型: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class ParentChildRetriever:
    """复用你的代码，完全不变"""

    def __init__(self, vectorstore, top_k: int = 3):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.parent_cache = {}

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


@dataclass
class RAGSystem:
    """RAG系统容器"""

    vectorstore: Any = None
    retriever: Any = None
    llm: Any = None
    rag_chain: Any = None


def initialize_rag() -> RAGSystem:
    """初始化RAG系统（复用你的配置）"""
    print("🔍 初始化哲学史问答系统...")

    query_embeddings = QueryEmbeddings("D:/models/bge-small-zh-v1.5")

    PERSIST_DIR = "D:/chroma_db/philosophy_history"
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=query_embeddings,
        collection_name="philosophy_parent_child",
    )
# 初始化 Chroma 向量数据库对象，并将其传递给 ParentChildRetriever 实例，确保检索器能够访问向量数据库进行查询。
    retriever = ParentChildRetriever(vectorstore, top_k=3)

    llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        model="glm-4-flash",
        temperature=0.3,
        max_tokens=2048,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是中国哲学史领域的资深研究者，擅长解读经典文本及其现代阐释。

基于以下从哲学史著作中检索到的完整章节内容，回答用户的问题。

回答要求：
1. 严格基于提供的文本内容，准确引用原文和作者观点
2. 注意区分：经典原文（《老子》等）、历代注疏、现代学者阐释
3. 保持哲学论证的严谨性，理清概念层次
4. 如果涉及多个章节，请分别说明不同章节的侧重点
5. 在回答末尾注明引用来源（章节标题）

输出约束（严格执行，千万不允许违反，后果很严重！！！）：
1. 禁止使用 JSON 格式、Markdown 代码块或结构化数据语法
2. 使用自然语言段落阐述观点，适当分段换行
3. 引用格式统一为：如学者所言（《书名》），**禁止**在文末添加"引用来源"、"参考章节"等总结性文字
4. 正文内提及章节时，使用括号注明即可，如（见《心外无理》章节），不要单独列出来源清单


检索到的相关章节：
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs: List[Document]) -> str:
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
# 这里使用了 RunnableParallel 来同时处理问题和检索上下文，并将格式化后的上下文传递给提示模板，确保 LLM 能够获得完整的章节内容进行回答。
    print(f"✅ 系统就绪！向量库包含 {vectorstore._collection.count()} 个子块")

    return RAGSystem(
        vectorstore=vectorstore, retriever=retriever, llm=llm, rag_chain=rag_chain
    )
