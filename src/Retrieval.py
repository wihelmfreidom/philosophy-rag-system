import os
import gc
import asyncio

# ========== 查询服务阶段（LangChain 1.0 标准 + 流式输出） ==========
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["OMP_NUM_THREADS"] = "4"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

print("🔍 启动《雕刻时光》问答系统（流式输出）...")
print("=" * 50)


# ========== 第一步：查询嵌入模型 ==========
class QueryEmbeddings:
    def __init__(self, model_path):
        print(f"🔄 加载查询模型: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: list):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


query_embeddings = QueryEmbeddings("D:/models/bge-small-zh-v1.5")


# ========== 第二步：加载向量数据库 ==========
PERSIST_DIR = "D:/chroma_db/sculpting_in_time"

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=query_embeddings,
    collection_name="sculpting_in_time",
)

retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
)

print("✅ 向量数据库加载成功")


# ========== 第三步：配置LLM（必须开启流式） ==========
os.environ["OPENAI_API_KEY"] = "your-deepseek-api-key"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

# 关键：streaming=True 启用流式
# llm = ChatOpenAI(model="deepseek-chat", temperature=0.3, streaming=True)  # 必须开启！
llm = ChatOpenAI(
    api_key="c8ff5d6d099742b0855805b4e59245c4.1ORR5pA9RqwR5wcq",
    base_url="https://open.bigmodel.cn/api/paas/v4",  # 智谱 OpenAI 兼容端点
    model="glm-4-flash",
    temperature=0.7,
    max_tokens=2048,
    streaming=True,  # 开启流式模式（关键！）
)
print("🤖 LLM就绪（流式模式）")


# ========== 第四步：构建RAG链 ==========
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是安德烈·塔可夫斯基电影理论的资深研究者。
基于以下从《雕刻时光》中检索到的片段，回答用户的问题。

回答要求：
1. 忠实于原文，引用具体观点
2. 保持塔可夫斯基的哲学深度和诗意表达
3. 如果文本不足以回答问题，明确说明
4. 在回答末尾注明引用来源（页码）

检索到的相关文本：
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "未知")
        content = doc.page_content
        formatted.append(f"[片段{i} - 第{page}页]\n{content}")
    return "\n\n".join(formatted)


# 基础RAG链（用于流式生成回答）
rag_chain = (
    RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG链构建完成")
print("=" * 50)


# ========== 第五步：流式查询函数 ==========
async def ask_question_stream(question: str):
    """流式执行问答，逐字输出"""
    print(f"\n❓ 问题: {question}")
    print("-" * 50)
    print("💡 回答: ", end="", flush=True)  # 不换行，准备流式输出

    try:
        # 先检索来源（非流式，需要等待）
        context_docs = retriever.invoke(question)

        # 流式生成回答
        full_answer = ""
        async for chunk in rag_chain.astream(question):
            print(chunk, end="", flush=True)  # 逐字打印
            full_answer += chunk

        print()  # 回答结束后换行

        # 显示来源
        print(f"\n📚 引用来源 ({len(context_docs)} 个片段):")
        for i, doc in enumerate(context_docs, 1):
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"   [{i}] 第{page}页: {preview}...")

        return full_answer, context_docs

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return "", []


def ask_question_sync(question: str):
    """同步包装器（用于非async环境）"""
    return asyncio.run(ask_question_stream(question))


# ========== 第六步：测试运行 ==========
if __name__ == "__main__":
    # 测试流式输出
    ask_question_sync("什么是雕刻时光？")
    print("\n" + "=" * 50)

    ask_question_sync("塔可夫斯基如何看待电影与现实的区别？")
    print("\n" + "=" * 50)

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
del rag_chain, vectorstore, query_embeddings
gc.collect()
print("✅ 完成")
