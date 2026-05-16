import os
import gc
import asyncio
import re
from typing import List, Dict, Any, Tuple

# ========== 环境配置 ==========
os.environ["HF_HOME"] = "D:/huggingface_cache"  # 修改操作系统的环境变量
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
from langchain_community.document_loaders import PyMuPDFLoader


print("🔍 启动哲学史问答系统（父子索引 + 流式输出）...")
print("=" * 60)


# ========== 第一步：查询嵌入模型 ==========
class QueryEmbeddings:
    def __init__(self, model_path):
        print(f"🔄 加载查询模型: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    # normalize_embeddings=True：对输出的向量进行 L2 归一化，使其模长为 1。归一化后的向量在计算余弦相似度时等价于点积，可提升检索效率。
    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


query_embeddings = QueryEmbeddings("D:/models/bge-small-zh-v1.5")


# ========== 第二步：自定义父子索引系统 ==========


class PhilosophicalTextSplitter:  # 实际中没有被使用，这个是处理txt文本的切分器，PDF文本使用PDFPhilosophicalSplitter
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
        # 正则表达式用于匹配大标题，支持中文数字开头，后跟空格（包括全角空格）和标题文字。MULTILINE 模式允许 ^ 匹配每行开头。

    def split_by_parent(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        按大标题切分父块
        返回: [{"title": "三 无之以为用", "content": "完整内容", "start_idx": 0}, ...]
        """
        # 找到所有大标题位置
        matches = list(self.title_pattern.finditer(text))
        # finditer 方法返回一个迭代器，包含所有匹配项的 Match 对象。每个 Match 对象包含匹配文本和位置等信息。使用 finditer
        # 找到文本中所有匹配大标题的位置，返回一个迭代器，转换为列表。每个 match 对象包含匹配的起始索引（start()）、
        # 结束索引（end()）和匹配的文本（group()）
        parents = []
        for i, match in enumerate(
            matches
        ):  # i 是当前大标题的索引，match 是当前大标题的 Match 对象
            title = match.group().strip()  # 获取匹配的标题文本，并去除首尾空白
            start = match.start()  # 获取当前大标题在文本中的起始索引位置
            end = (
                matches[i + 1].start() if i + 1 < len(matches) else len(text)
            )  # 获取当前大标题的结束索引位置，如果是最后一个大标题，则结束索引为文本长度
            content = text[
                start:end
            ].strip()  # 提取当前大标题对应的内容，即从当前大标题的起始位置到下一个大标题的起始位置之间的文本，并去除首尾空白

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
        self,
        parent: Dict[str, Any],
        chunk_size: int = 300,
        overlap: int = 50,  # chunk_size 是子块的最大字符数，overlap 是子块之间的重叠字符数
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
            separators=[
                "\n\n",
                "\n",
                "。",
                "；",
                "，",
                " ",
            ],  # 优先按段落分割，再按句子分割，最后按空格分割
            length_function=len,
        )

        child_texts = splitter.split_text(body)

        documents = (
            []
        )  # Document 是 LangChain 中表示文档的标准类，包含 page_content（文本内容）和 metadata（元数据字典）。
        for idx, text in enumerate(child_texts):
            doc = Document(
                page_content=text,
                metadata={
                    **parent[
                        "metadata"
                    ],  # metadata 继承父块的元数据（**parent["metadata"]），并添加子块特有的字段
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


# 用户查询嵌入后，从向量库中召回最相关的子块。由于子块元数据中存有 parent_content，可以将父块完整内容作为上下文传递给大模型，从而生成更准确的回答。
# 这种方式既利用了子块的高精度检索，又能在后续生成时提供完整的上下文，避免了传统父子检索中需要额外存储父块内容的复杂性。
# 如果只切分父块而不切分子块的话，检索时只能返回父块，无法利用子块的精细匹配来提升检索质量。而切分子块后，每个子块都带有父块的完整内容引用，可以在检索到相关子块后直接使用父块内容进行生成，极大提升了回答的准确性和上下文丰富度。
class ParentChildRetriever:
    """
    简化版父子检索器（兼容 LangChain 1.0）
    无需 InMemoryByteStore，直接通过 metadata 关联
    """

    #   父子检索器的核心思想是：在切分文档时，子块（用于向量检索）包含对父块完整内容的引用（通过 metadata 存储）。这样，在检索到相关子块后，可以直接获取父块的完整内容作为上下文进行生成，无需额外存储父块内容到向量库中。
    def __init__(self, vectorstore, top_k: int = 3):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.parent_cache = {}  # 缓存父块内容

    def add_documents(self, documents: List[Document]):
        """添加子块文档到向量库，同时缓存父块"""
        for doc in documents:
            # documents 是由 PhilosophicalTextSplitter.split_children 生成的子块文档列表，
            # 每个文档的 metadata 中包含 parent_id、parent_title、parent_content 等字段。
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in self.parent_cache:
                # 缓存父块完整内容
                self.parent_cache[parent_id] = (
                    {  # 缓存父块信息到 parent_cache 字典，键为 parent_id，值为包含父块标题、内容和其他元数据的字典，以便后续检索时快速获取父块信息。
                        "title": doc.metadata.get("parent_title", ""),
                        "content": doc.metadata.get("parent_content", ""),
                        "metadata": {
                            k: v
                            for k, v in doc.metadata.items()
                            if k
                            not in [
                                "child_id",
                                "chunk_index",
                                "is_child",
                                "parent_content",
                            ]
                        },  # 继承父块的元数据（移除子块特有的字段如 child_id、chunk_index、
                        # is_child，并排除 parent_content，因为已单独存储）。这样缓存中的元数据干净，适合最终返回给生成环节。
                    }
                )

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
                        **parent_info["metadata"],  # 父块的元数据（章节标题、编号等）
                        "retrieved_child": child.page_content,  # 记录哪个子块触发了命中
                        "child_similarity_source": True,  # 标记这是通过子块相似度检索到的父块
                    },
                )
                parent_docs.append(parent_doc)

                if len(parent_docs) >= self.top_k:  # 只返回 top_k 个父块，避免过多冗余
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
PERSIST_DIR = "D:/chroma_db/philosophy_history_v4"

# 清空或创建新库（首次运行）
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=query_embeddings,
    collection_name="philosophy_parent_child_v4",
)

retriever = ParentChildRetriever(vectorstore, top_k=3)

print("✅ 父子索引系统初始化完成")


# ========== 第四步：文档处理函数（处理你的哲学史文本） ==========


class PDFPhilosophicalSplitter:
    """
    针对哲学史 PDF 的专用切分器（修复 parent_id 冲突版）
    父块：大标题（"三 无之以为用"）到下一个大标题之间的完整内容（可能跨多页）
    子块：父块内部的自然段落组
    """

    def __init__(self):
        # 匹配大标题：支持"第五讲"前缀 + 中文数字 + 空格/全角空格 + 标题文字
        self.title_pattern = re.compile(
            r"^\s*(?:第[一二三四五六七八九十百千]+讲\s+)?([一二三四五六七八九十百千]+)[\s　、]+(.+?)\s*$",
            re.MULTILINE,
        )
        # 清理 PDF 提取的噪声（脚注标记等）
        self.noise_pattern = re.compile(r"\[\d+\]")

    def clean_text(self, text: str) -> str:
        """清理 PDF 提取的格式噪声"""
        # 合并断行（PDF 常把一行断成多行）
        text = re.sub(r"(?<=[^。！？\n])\n(?=[^一二三四五六七八九十百千])", "", text)
        return text.strip()

    def identify_titles(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        在所有页面中识别大标题位置
        返回: [{"title": "三 无之以为用", "page": 45, "char_idx": 120, ...}, ...]
        """
        titles = []

        for doc_idx, doc in enumerate(documents):
            page_num = doc.metadata.get("page", doc_idx + 1)
            text = self.clean_text(doc.page_content)

            for match in self.title_pattern.finditer(text):
                chapter_num = match.group(1)
                chapter_name = match.group(2)
                full_title = f"{chapter_num} {chapter_name}"

                titles.append(
                    {
                        "title": full_title,
                        "page": page_num,
                        "char_idx": match.start(),
                        "chapter_number": chapter_num,
                        "chapter_name": chapter_name,
                        "doc_idx": doc_idx,
                        "matched_text": match.group(0),
                    }
                )

        # 按文档顺序和字符位置排序
        titles.sort(key=lambda x: (x["doc_idx"], x["char_idx"]))

        print(f"🔍 识别到 {len(titles)} 个大章节标题:")
        for t in titles[:5]:
            print(f"   第{t['page']}页: {t['title']}")
        if len(titles) > 5:
            print(f"   ... 等共 {len(titles)} 个")

        return titles

    def build_parent_blocks(
        self, documents: List[Document], titles: List[Dict], source: str = ""
    ) -> List[Dict[str, Any]]:
        """
        根据标题位置，跨页构建父块
        关键修复：使用全局唯一 parent_id 避免章节序号冲突
        """
        if not titles:
            print("⚠️ 未识别到任何标题，将按页切分")
            return self._fallback_by_pages(documents)

        parents = []

        for i, title_info in enumerate(titles):
            start_doc_idx = title_info["doc_idx"]
            start_char_idx = title_info["char_idx"]

            # 确定结束位置（下一个标题或文档末尾）
            if i + 1 < len(titles):
                end_doc_idx = titles[i + 1]["doc_idx"]
                end_char_idx = titles[i + 1]["char_idx"]
            else:
                end_doc_idx = len(documents) - 1
                end_char_idx = len(self.clean_text(documents[end_doc_idx].page_content))

            # 提取内容（可能跨多页）
            content_parts = []
            page_range = []

            for doc_idx in range(start_doc_idx, end_doc_idx + 1):
                doc = documents[doc_idx]
                page_num = doc.metadata.get("page", doc_idx + 1)
                text = self.clean_text(doc.page_content)

                if doc_idx == start_doc_idx and doc_idx == end_doc_idx:
                    part = text[start_char_idx:end_char_idx]
                elif doc_idx == start_doc_idx:
                    part = text[start_char_idx:]
                elif doc_idx == end_doc_idx:
                    part = text[:end_char_idx]
                else:
                    part = text

                if part.strip():
                    content_parts.append(part)
                    page_range.append(str(page_num))

            full_content = "\n\n".join(content_parts)
            page_range_str = (
                "-".join([page_range[0], page_range[-1]])
                if len(page_range) > 1
                else page_range[0]
            )

            # 🎯 关键修复：使用全局唯一 ID，格式：来源_序号_标题_索引
            # 这样即使不同大章节都有"三 性善"，ID 也不同
            unique_parent_id = f"{source}_{title_info['chapter_number']}_{title_info['chapter_name']}_{i}"

            parents.append(
                {
                    "title": title_info["title"],
                    "chapter_number": title_info["chapter_number"],
                    "chapter_name": title_info["chapter_name"],
                    "content": full_content,
                    "start_page": int(page_range[0]),
                    "end_page": int(page_range[-1]),
                    "page_range": page_range_str,
                    "parent_id": unique_parent_id,
                    "metadata": {
                        "chapter_title": title_info["title"],
                        "chapter_number": title_info["chapter_number"],
                        "start_page": int(page_range[0]),
                        "end_page": int(page_range[-1]),
                        "page_range": page_range_str,
                        "parent_id": unique_parent_id,  # 确保 metadata 中也唯一
                        "source": source,
                    },
                }
            )

        print(f"✂️ 构建完成：{len(parents)} 个父块（章节）")
        total_chars = sum(len(p["content"]) for p in parents)
        avg_chars = total_chars // len(parents) if parents else 0
        print(f"   总字符数: {total_chars}, 平均: {avg_chars} 字/章节")

        return parents

    def split_children(
        self, parent: Dict[str, Any], chunk_size: int = 300, overlap: int = 50
    ) -> List[Document]:
        """
        将父块（完整章节）切分为子块
        策略：保留段落边界，区分原文引用和作者论述
        """
        content = parent["content"]

        # 使用文本切分器，优先按段落分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
            length_function=len,
        )

        child_texts = splitter.split_text(content)
        documents = []

        for idx, text in enumerate(child_texts):
            # 启发式识别内容类型
            content_type = self._classify_content_type(text)

            doc = Document(
                page_content=text,
                metadata={
                    **parent["metadata"],
                    "child_id": f"{parent['parent_id']}_child_{idx}",
                    "chunk_index": idx,
                    "is_child": True,
                    "content_type": content_type,  # 原文/注释/论述
                    "parent_content": parent["content"],  # 完整父块内容（用于生成）
                    "parent_title": parent["title"],
                },
            )
            documents.append(doc)

        return documents

    def _classify_content_type(self, text: str) -> str:
        """启发式识别内容类型（原文/注释/论述）"""
        text = text.strip()

        # 判断是否为经典原文引用（《老子》《庄子》等）
        if any(
            marker in text[:30]
            for marker in [
                "《老子》",
                "《庄子》",
                "《论语》",
                "《孟子》",
                "曰：",
                "故曰",
            ]
        ):
            return "经典原文"
        # 判断是否为学者注释
        elif any(marker in text[:30] for marker in ["注》", "认为", "解释", "疏》"]):
            return "学者注释"
        # 判断是否为作者论述
        elif len(text) > 100 and "。" in text:
            return "作者论述"
        else:
            return "一般内容"

    def _fallback_by_pages(self, documents: List[Document]) -> List[Dict]:
        """降级方案：按页作为父块（当无法识别标题时使用）"""
        parents = []
        for doc_idx, doc in enumerate(documents):
            page_num = doc.metadata.get("page", doc_idx + 1)
            parents.append(
                {
                    "title": f"第{page_num}页",
                    "chapter_number": str(page_num),
                    "chapter_name": "",
                    "content": self.clean_text(doc.page_content),
                    "start_page": page_num,
                    "end_page": page_num,
                    "page_range": str(page_num),
                    "parent_id": f"page_{page_num}_{doc_idx}",  # 确保唯一
                    "metadata": {
                        "chapter_title": f"第{page_num}页",
                        "chapter_number": str(page_num),
                        "start_page": page_num,
                        "end_page": page_num,
                        "page_range": str(page_num),
                        "parent_id": f"page_{page_num}_{doc_idx}",
                        "source": "unknown",
                    },
                }
            )
        return parents


# ========== 处理流程函数（使用上述类） ==========


def process_philosophy_pdf(pdf_path: str, source_name: str = "") -> Tuple[int, int]:
    """
    处理哲学史 PDF，构建父子索引
    返回: (父块数量, 子块数量)
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    print(f"\n📖 处理 PDF: {pdf_path}")
    print("-" * 50)

    # 1. 加载 PDF
    print("📄 正在加载 PDF...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ 加载完成：{len(documents)} 页")

    # 2. 使用专用切分器
    splitter = PDFPhilosophicalSplitter()

    # 3. 识别标题并构建父块
    titles = splitter.identify_titles(documents)
    parents = splitter.build_parent_blocks(documents, titles, source=source_name)

    # 4. 切分子块并添加到检索器
    all_children = []
    for parent in parents:
        children = splitter.split_children(parent, chunk_size=280, overlap=40)
        all_children.extend(children)
        print(
            f"   [{parent['title']}] 第{parent['page_range']}页 → {len(children)} 个子块"
        )

    # 5. 添加到检索器（假设 retriever 是全局或传入的）
    if "retriever" in globals():
        retriever.add_documents(all_children)
        print(f"\n✅ 索引完成：{len(parents)} 个章节，{len(all_children)} 个检索单元")
    else:
        print(f"\n⚠️ 返回文档列表，请手动添加到检索器")
        return all_children, parents

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
        # 流式输出的实现：使用 astream 方法逐 chunk 获取生成的回答，并立即打印出来，同时将完整回答保存在 full_answer 变量中，以便后续显示来源详情。
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
    PDF_PATH = "src/ChinesePhiChatbot/chinesephi.pdf"

    # 处理 PDF（首次运行）
    if os.path.exists(PDF_PATH):
        num_parents, num_children = process_philosophy_pdf(
            PDF_PATH, source_name="中国哲学史"
        )

        # 持久化向量库
        if "vectorstore" in globals():
            vectorstore.persist()
            print(f"💾 向量库已持久化到: {PERSIST_DIR}")

        print(f"\n📊 统计: {num_parents} 个章节, {num_children} 个子块")
    else:
        print(f"⚠️ 未找到 PDF: {PDF_PATH}")

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


# 调用关系：main 函数中调用了 process_philosophy_pdf 来处理 PDF 文档，构建父子索引，并将子块添加到检索器中。
# 之后，ask_question_sync 函数被用来测试查询和交互式问答，内部调用了 ask_question_stream 来执行流式问答逻辑。
# ask_question_stream 中调用了 rag_chain.astream 来执行 RAG 链，并使用 retriever.invoke 来获取相关章节作为上下文。
