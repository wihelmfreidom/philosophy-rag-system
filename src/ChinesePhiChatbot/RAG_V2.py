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

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


query_embeddings = QueryEmbeddings("D:/models/bge-small-zh-v1.5")


# ========== 第二步：自定义父子索引系统（修复版） ==========


class PDFPhilosophicalSplitter:
    """
    针对哲学史 PDF 的专用切分器（修复版：支持讲次层级识别）
    父块：讲次（第X讲）+ 大标题（"三 无之以为用"）到下一个大标题之间的完整内容
    子块：父块内部的自然段落组
    """

    def __init__(self):
        # 匹配讲次标题：第X讲 + 标题文字（如"第一讲 性与天道"）
        self.lecture_pattern = re.compile(
            r"^\s*第([一二三四五六七八九十百千]+)讲[\s　]+(.+?)\s*$", re.MULTILINE
        )

        # 匹配章节小标题：中文数字 + 空格/全角空格 + 标题文字（如"三 无之以为用"）
        # 排除讲次标题（通过负向前瞻避免匹配"第X讲"）
        self.title_pattern = re.compile(
            r"^\s*(?:(?!第[一二三四五六七八九十百千]+讲))([一二三四五六七八九十百千]+)[\s　、]+(.+?)\s*$",
            re.MULTILINE,
        )

        # 清理 PDF 提取的噪声
        self.noise_pattern = re.compile(r"\[\d+\]")

    def clean_text(self, text: str) -> str:
        """清理 PDF 提取的格式噪声"""
        # 合并断行（PDF 常把一行断成多行）
        text = re.sub(r"(?<=[^。！？\n])\n(?=[^一二三四五六七八九十百千])", "", text)
        return text.strip()

    def identify_titles(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        在所有页面中识别讲次和章节标题
        返回: [{"title": "三 无之以为用", "lecture": "第一讲", "lecture_title": "性与天道", ...}, ...]
        """
        titles = []
        current_lecture = "未知讲次"
        current_lecture_num = "0"
        current_lecture_title = ""

        for doc_idx, doc in enumerate(documents):
            page_num = doc.metadata.get("page", doc_idx + 1)
            text = self.clean_text(doc.page_content)

            # 首先检查是否遇到新的讲次（如"第一讲 性与天道"）
            lecture_match = self.lecture_pattern.search(text)
            if lecture_match:
                current_lecture_num = lecture_match.group(1)
                current_lecture_title = lecture_match.group(2)
                current_lecture = f"第{current_lecture_num}讲"
                print(f"📚 识别到讲次: {current_lecture} {current_lecture_title}")

            # 识别该讲下的章节小标题
            for match in self.title_pattern.finditer(text):
                chapter_num = match.group(1)
                chapter_name = match.group(2)
                full_title = f"{chapter_num} {chapter_name}"

                titles.append(
                    {
                        "title": full_title,
                        "lecture": current_lecture,
                        "lecture_number": current_lecture_num,
                        "lecture_title": current_lecture_title,
                        "chapter_number": chapter_num,
                        "chapter_name": chapter_name,
                        "page": page_num,
                        "doc_idx": doc_idx,
                        "char_idx": match.start(),
                        "matched_text": match.group(0),
                    }
                )

        # 按文档顺序和字符位置排序
        titles.sort(key=lambda x: (x["doc_idx"], x["char_idx"]))

        print(f"🔍 识别到 {len(titles)} 个章节标题，分布在多个讲次:")
        lecture_stats = {}
        for t in titles:
            lec = t["lecture"]
            lecture_stats[lec] = lecture_stats.get(lec, 0) + 1

        for lec, count in sorted(lecture_stats.items()):
            print(f"   {lec}: {count} 个章节")

        return titles

    def build_parent_blocks(
        self, documents: List[Document], titles: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        根据标题位置，跨页构建父块（修复版：使用讲次+章节作为唯一ID）
        """
        if not titles:
            print("⚠️ 未识别到任何标题，将按页切分")
            return self._fallback_by_pages(documents)

        parents = []

        for i, title_info in enumerate(titles):
            start_doc_idx = title_info["doc_idx"]
            start_char_idx = title_info["char_idx"]

            # 确定结束位置
            if i + 1 < len(titles):
                end_doc_idx = titles[i + 1]["doc_idx"]
                end_char_idx = titles[i + 1]["char_idx"]
            else:
                # 最后一个标题：到文档末尾
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
                    # 起止在同一页
                    part = text[start_char_idx:end_char_idx]
                elif doc_idx == start_doc_idx:
                    # 起始页：从标题开始到页末
                    part = text[start_char_idx:]
                elif doc_idx == end_doc_idx:
                    # 结束页：从页首到结束位置
                    part = text[:end_char_idx]
                else:
                    # 中间页：整页
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

            # 关键修复：使用"讲次_章节"作为唯一ID，避免不同讲次的"章节三"冲突
            unique_parent_id = (
                f"{title_info['lecture']}_章节{title_info['chapter_number']}"
            )

            # 构建完整标题（包含讲次信息）
            full_chapter_title = f"{title_info['lecture']} {title_info['title']}"

            parents.append(
                {
                    "title": title_info["title"],
                    "lecture": title_info["lecture"],
                    "lecture_title": title_info["lecture_title"],
                    "chapter_number": title_info["chapter_number"],
                    "chapter_name": title_info["chapter_name"],
                    "content": full_content,
                    "start_page": int(page_range[0]),
                    "end_page": int(page_range[-1]),
                    "page_range": page_range_str,
                    "parent_id": unique_parent_id,  # 如："第一讲_章节三"
                    "metadata": {
                        "lecture": title_info["lecture"],
                        "lecture_title": title_info["lecture_title"],
                        "chapter_title": full_chapter_title,  # 显示用："第一讲 三 无之以为用"
                        "short_title": title_info["title"],  # 简短标题："三 无之以为用"
                        "chapter_number": title_info["chapter_number"],
                        "start_page": int(page_range[0]),
                        "end_page": int(page_range[-1]),
                        "page_range": page_range_str,
                        "parent_id": unique_parent_id,
                    },
                }
            )

        print(f"✂️ 构建完成：{len(parents)} 个父块（章节）")
        total_chars = sum(len(p["content"]) for p in parents)
        print(
            f"   总字符数: {total_chars}, 平均: {total_chars//len(parents) if parents else 0} 字/章节"
        )

        return parents

    def _fallback_by_pages(self, documents: List[Document]) -> List[Dict]:
        """未识别标题时的降级方案：按页作为父块"""
        parents = []
        for doc_idx, doc in enumerate(documents):
            page_num = doc.metadata.get("page", doc_idx + 1)
            parents.append(
                {
                    "title": f"第{page_num}页",
                    "lecture": "未知讲次",
                    "chapter_number": str(page_num),
                    "chapter_name": "",
                    "content": self.clean_text(doc.page_content),
                    "start_page": page_num,
                    "end_page": page_num,
                    "page_range": str(page_num),
                    "parent_id": f"未知讲次_页面{page_num}",
                    "metadata": {
                        "lecture": "未知讲次",
                        "lecture_title": "",
                        "chapter_title": f"第{page_num}页",
                        "chapter_number": str(page_num),
                        "start_page": page_num,
                        "end_page": page_num,
                        "page_range": str(page_num),
                        "parent_id": f"未知讲次_页面{page_num}",
                    },
                }
            )
        return parents

    def split_children(
        self, parent: Dict[str, Any], chunk_size: int = 300, overlap: int = 50
    ) -> List[Document]:
        """
        将父块（完整章节）切分为子块
        """
        content = parent["content"]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
            length_function=len,
        )

        child_texts = splitter.split_text(content)

        documents = []
        for idx, text in enumerate(child_texts):
            content_type = self._classify_content_type(text)

            doc = Document(
                page_content=text,
                metadata={
                    **parent["metadata"],
                    "child_id": f"{parent['parent_id']}_child_{idx}",  # 如："第一讲_章节三_child_0"
                    "chunk_index": idx,
                    "is_child": True,
                    "content_type": content_type,
                    "parent_content_preview": (
                        parent["content"][:500] + "..."
                        if len(parent["content"]) > 500
                        else parent["content"]
                    ),
                    "parent_full_content": parent["content"],  # 完整父块用于生成
                },
            )
            documents.append(doc)

        return documents

    def _classify_content_type(self, text: str) -> str:
        """启发式识别内容类型"""
        text = text.strip()
        if "《老子》" in text or text.startswith("道") or "故曰" in text[:10]:
            return "经典原文"
        elif "注》" in text or "认为" in text[:20] or "曰：" in text:
            return "学者注释"
        elif len(text) > 100 and "。" in text:
            return "作者论述"
        else:
            return "一般内容"


class ParentChildRetriever:
    """
    父子检索器（修复版：使用复合ID避免冲突）
    """

    def __init__(self, vectorstore, top_k: int = 3):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.parent_cache = {}

    def add_documents(self, documents: List[Document]):
        """添加子块文档到向量库，同时缓存父块"""
        for doc in documents:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in self.parent_cache:
                # 缓存父块完整内容（现在parent_id是唯一的）
                self.parent_cache[parent_id] = {
                    "title": doc.metadata.get("chapter_title", ""),
                    "content": doc.metadata.get("parent_full_content", ""),
                    "metadata": {
                        k: v
                        for k, v in doc.metadata.items()
                        if k
                        not in [
                            "child_id",
                            "chunk_index",
                            "is_child",
                            "parent_content_preview",
                            "parent_full_content",
                        ]
                    },
                }

        # 只嵌入子块（移除parent_full_content避免冗余存储）
        docs_to_embed = []
        for doc in documents:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={
                    k: v
                    for k, v in doc.metadata.items()
                    if k not in ["parent_full_content", "parent_content_preview"]
                },
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

        # 2. 按父块ID聚合，去重（现在parent_id是唯一的，不会跨讲次混淆）
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
                        "retrieved_child": child.page_content,
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
            # 显示完整的讲次信息
            lec = p.metadata.get("lecture", "")
            short = p.metadata.get("short_title", "未知")
            print(f"   [{i}] {lec} {short}")

        return parent_docs

    def ainvoke(self, query: str):
        """异步版本（LangChain 1.0 标准）"""
        return self.invoke(query)


# ========== 第三步：初始化向量库和检索器 ==========
PERSIST_DIR = "D:/chroma_db/philosophy_history_v2"  # 修改路径，避免与旧缓存冲突

# 清空或创建新库（首次运行）
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=query_embeddings,
    collection_name="philosophy_parent_child_v2",  # 修改collection名
)

retriever = ParentChildRetriever(vectorstore, top_k=3)

print("✅ 父子索引系统初始化完成（修复版：支持讲次层级）")


# ========== 第四步：文档处理函数 ==========


def process_philosophy_pdf(pdf_path: str, source_name: str = "") -> Tuple[int, int]:
    """
    处理哲学史 PDF，构建父子索引（修复版）
    """
    print(f"\n📖 处理 PDF: {pdf_path}")
    print("-" * 50)

    # 1. 加载 PDF
    print("📄 正在加载 PDF...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ 加载完成：{len(documents)} 页")

    # 2. 使用修复后的专用切分器
    splitter = PDFPhilosophicalSplitter()

    # 3. 识别讲次和标题，构建父块
    titles = splitter.identify_titles(documents)
    parents = splitter.build_parent_blocks(documents, titles)

    # 4. 切分子块并添加到检索器
    all_children = []
    for parent in parents:
        children = splitter.split_children(parent, chunk_size=280, overlap=40)
        all_children.extend(children)
        # 简化的进度显示
        lec = parent["lecture"]
        title = parent["title"]
        print(f"   [{lec} {title}] 第{parent['page_range']}页 → {len(children)} 个子块")

    # 5. 添加到检索器
    if "retriever" in globals():
        retriever.add_documents(all_children)
        print(f"\n✅ 索引完成：{len(parents)} 个章节，{len(all_children)} 个检索单元")
        print(f"   讲次分布: {len(set(p['lecture'] for p in parents))} 个讲次")
    else:
        print(f"\n⚠️ 返回文档列表，请手动添加到检索器")
        return all_children, parents

    return len(parents), len(all_children)


# ========== 第五步：配置LLM ==========
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
5. 在回答末尾注明引用来源（讲次和章节标题）

检索到的相关章节：
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs: List[Document]) -> str:
    """格式化父块文档（显示讲次信息）"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("chapter_title", "未知章节")
        content = doc.page_content
        # 截断过长的内容，保留核心论述
        if len(content) > 1500:
            content = content[:1500] + "...（后续内容省略）"

        formatted.append(f"【章节{i}：{title}】\n{content}\n")

    return "\n---\n".join(formatted)


# 构建RAG链
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
    """流式执行问答（修复版：显示正确来源）"""
    print(f"\n❓ 问题: {question}")
    print("-" * 50)
    print("💡 回答: ", end="", flush=True)

    try:
        full_answer = ""
        async for chunk in rag_chain.astream(question):
            print(chunk, end="", flush=True)
            full_answer += chunk
        print("\n")

        # 显示来源详情（现在会显示正确的讲次）
        source_docs = retriever.invoke(question)
        print(f"\n📚 引用来源 ({len(source_docs)} 个章节):")
        for i, doc in enumerate(source_docs, 1):
            lec = doc.metadata.get("lecture", "")
            short = doc.metadata.get("short_title", "?")
            trigger = doc.metadata.get("retrieved_child", "")[:40]
            print(f"   [{i}] {lec} {short}")
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

    print("开始处理 PDF 并构建索引（如果尚未完成）...")

    # 示例1：处理文本（首次运行）
    PDF_PATH = "src/ChinesePhiChatbot/chinesephi.pdf"

    # 处理 PDF（首次运行）
    if os.path.exists(PDF_PATH):
        # 建议：如果是重新索引，先清空旧数据
        # retriever.parent_cache.clear()
        # vectorstore.delete_collection()

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

    # 示例2：测试查询
    print("\n🧪 测试查询...")

    test_questions = [
        "什么是无之以为用？",
        "老子如何看待柔弱与刚强的关系？",
        "孟子性善论的主要内容是什么？",
        "王弼的'大衍义'是指什么？",
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
