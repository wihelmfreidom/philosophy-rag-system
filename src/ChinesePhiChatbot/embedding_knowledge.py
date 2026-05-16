import os
import gc
import asyncio
import re
from typing import List, Dict, Any

# ========== 环境配置 ==========
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["OMP_NUM_THREADS"] = "4"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

print("🔍 启动哲学史问答系统（父子索引 + 流式输出）...")
print("=" * 60)


# ========== 第一步：查询嵌入模型 ==========
class QueryEmbeddings:
    def __init__(self, model_path):
        print(f"🔄 加载查询模型: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


query_embeddings = QueryEmbeddings("D:/models/bge-small-zh-v1.5")


# ========== 第二步：自定义父子索引系统 ==========


class PhilosophicalTextSplitter:
    """
    针对哲学史材料的专用切分器
    父块：大标题（如"三 无之以为用"）到下一个大标题之间的完整内容
    子块：父块内部的自然段落组
    """

    def __init__(self):
        # 匹配大标题：中文数字 + 空格 + 标题文字
        self.title_pattern = re.compile(
            r"^[一二三四五六七八九十百千]+[\s　]+.+$", re.MULTILINE
        )

    def split_by_parent(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        按大标题切分父块
        返回: [{"title": "三 无之以为用", "content": "完整内容", "start_idx": 0}, ...]
        """
        # 找到所有大标题位置
        matches = list(self.title_pattern.finditer(text))

        parents = []
        for i, match in enumerate(matches):
            title = match.group().strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            parents.append(
                {
                    "title": title,
                    "content": content,
                    "source": source,
                    "parent_id": f"{source}_{i}",
                    "metadata": {
                        "chapter_title": title,
                        "chapter_number": title.split()[0] if title.split() else "",
                        "parent_id": f"{source}_{i}",
                    },
                }
            )

        return parents

    def split_children(
        self, parent: Dict[str, Any], chunk_size: int = 300, overlap: int = 50
    ) -> List[Document]:
        """
        将父块切分为子块（用于向量检索）
        策略：按自然段落分割，保持引用-论述的完整性
        """
        content = parent["content"]

        # 移除标题，保留正文
        lines = content.split("\n")
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else content

        # 使用文本切分器，优先按段落分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
            length_function=len,
        )

        child_texts = splitter.split_text(body)

        documents = []
        for idx, text in enumerate(child_texts):
            doc = Document(
                page_content=text,
                metadata={
                    **parent["metadata"],
                    "child_id": f"{parent['parent_id']}_child_{idx}",
                    "chunk_index": idx,
                    "is_child": True,
                    # 关键：存储父块完整内容引用
                    "parent_content": parent["content"],
                    "parent_title": parent["title"],
                },
            )
            documents.append(doc)

        return documents


class ParentChildRetriever:
    """
    简化版父子检索器（兼容 LangChain 1.0）
    无需 InMemoryByteStore，直接通过 metadata 关联
    """

    def __init__(self, vectorstore, top_k: int = 3):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.parent_cache = {}  # 缓存父块内容

    def add_documents(self, documents: List[Document]):
        """添加子块文档到向量库，同时缓存父块"""
        for doc in documents:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in self.parent_cache:
                # 缓存父块完整内容
                self.parent_cache[parent_id] = {
                    "title": doc.metadata.get("parent_title", ""),
                    "content": doc.metadata.get("parent_content", ""),
                    "metadata": {
                        k: v
                        for k, v in doc.metadata.items()
                        if k
                        not in ["child_id", "chunk_index", "is_child", "parent_content"]
                    },
                }

        # 只嵌入子块（移除parent_content避免冗余存储）
        docs_to_embed = []
        for doc in documents:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={
                    k: v for k, v in doc.metadata.items() if k != "parent_content"
                },  # 不存储大内容到向量库
            )
            docs_to_embed.append(doc_copy)

        self.vectorstore.add_documents(docs_to_embed)
        print(
            f"✅ 索引完成：{len(docs_to_embed)} 个子块，{len(self.parent_cache)} 个父块"
        )

    def invoke(self, query: str) -> List[Document]:
        """
        检索流程：子块匹配 → 获取父块 → 去重 → 返回完整父块
        """
        # 1. 检索子块
        child_docs = self.vectorstore.similarity_search(query, k=self.top_k * 2)

        # 2. 按父块ID聚合，去重
        seen_parents = set()
        parent_docs = []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if (
                parent_id
                and parent_id not in seen_parents
                and parent_id in self.parent_cache
            ):
                seen_parents.add(parent_id)
                parent_info = self.parent_cache[parent_id]

                # 构建父块文档（用于生成）
                parent_doc = Document(
                    page_content=parent_info["content"],
                    metadata={
                        **parent_info["metadata"],
                        "retrieved_child": child.page_content,  # 记录哪个子块触发了命中
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
        """异步版本（LangChain 1.0 标准）"""
        return self.invoke(query)  # 简化实现，实际可改为异步


# ========== 第三步：初始化向量库和检索器 ==========
PERSIST_DIR = "D:/chroma_db/philosophy_history"

# 清空或创建新库（首次运行）
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=query_embeddings,
    collection_name="philosophy_parent_child",
)

retriever = ParentChildRetriever(vectorstore, top_k=3)

print("✅ 父子索引系统初始化完成")


# ========== 第四步：文档处理函数（处理你的哲学史文本） ==========


def process_philosophy_text(file_path: str, source_name: str = ""):
    """
    处理哲学史文本文件，构建父子索引
    输入：原始文本文件路径
    """
    print(f"\n📖 处理文本: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 使用专用切分器
    splitter = PhilosophicalTextSplitter()

    # 1. 切分父块（按大标题）
    parents = splitter.split_by_parent(text, source=source_name or file_path)
    print(f"   识别到 {len(parents)} 个大章节")

    # 2. 切分子块并构建文档
    all_children = []
    for parent in parents:
        children = splitter.split_children(
            parent, chunk_size=250, overlap=30  # 子块较小，精准匹配
        )
        all_children.extend(children)
        print(f"   [{parent['title']}] → {len(children)} 个子块")

    # 3. 添加到检索器
    retriever.add_documents(all_children)

    return len(parents), len(all_children)


# ========== 第五步：配置LLM ==========
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

llm = ChatOpenAI(
    api_key="",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    model="glm-4-flash",
    temperature=0.3,  # 哲学文本需要更严谨的生成
    max_tokens=2048,
    streaming=True,
)

print("🤖 LLM就绪（流式模式）")


# ========== 第六步：构建RAG链 ==========
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
        # 截断过长的内容，保留核心论述
        if len(content) > 1500:
            content = content[:1500] + "...（后续内容省略）"

        formatted.append(f"【章节{i}：{title}】\n{content}\n")

    return "\n---\n".join(formatted)


# 构建RAG链（兼容流式）
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

print("✅ RAG链构建完成")
print("=" * 60)


# ========== 第七步：流式查询函数 ==========
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

        # 显示来源详情
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


# ========== 第八步：使用示例 ==========

if __name__ == "__main__":

    # 示例1：处理文本（首次运行）
    # 假设你的文本文件路径
    text_file = r"D:\FastApiProjects\quickstart\src\ChinesePhiChatbot\chinesephi.pdf"
    if os.path.exists(text_file):
        process_philosophy_text(text_file, source_name="中国哲学十五讲")
        vectorstore.persist()
        print("✅ 文本处理完成并持久化")

    # 示例2：测试查询（假设已有索引）
    print("\n🧪 测试查询...")

    test_questions = [
        "什么是无之以为用？",
        "老子如何看待柔弱与刚强的关系？",
        "道生之，德畜之这句话怎么理解？",
        "企者不立是什么意思？",
    ]

    for q in test_questions:
        ask_question_sync(q)
        print("\n" + "=" * 60)

    # 交互模式
    print("\n💬 输入你的问题（'退出'结束）:")
    while True:
        user_q = input("> ").strip()
        if user_q.lower() in ["退出", "quit", "q"]:
            break
        if user_q:
            ask_question_sync(user_q)
            print()


# ========== 清理 ==========
print("\n🧹 清理资源...")
gc.collect()
print("✅ 完成")
