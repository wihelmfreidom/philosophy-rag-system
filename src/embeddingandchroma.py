import os
import gc

# ========== 第一步：环境配置（必须在最开始） ==========
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["OMP_NUM_THREADS"] = "4"  # 限制线程数，减少内存碎片

# 创建缓存目录
os.makedirs("D:/huggingface_cache", exist_ok=True)

# 向量数据库存储路径（D盘）
PERSIST_DIR = "D:/chroma_db/sculpting_in_time"

# 现在导入其他库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer

print("🚀 开始构建《雕刻时光》向量索引...")
print("=" * 50)

# ========== 第二步：加载PDF ==========
pdf_path = "src/SculptingInTime.pdf"  # 根据实际路径修改
print(f"📄 正在加载PDF: {pdf_path}")

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ 加载完成：{len(documents)} 页")

# ========== 第三步：分块 ==========
print("✂️ 正在分块...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # 中文约400字
    chunk_overlap=80,  # 重叠保持连贯
    separators=["\n\n\n", "\n\n", "\n", "。", "，", " ", ""],
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"✅ 分块完成：{len(chunks)} 块")

# 添加唯一ID
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{i:04d}"

# ========== 第四步：初始化本地嵌入模型 ==========
# 1. 定义你的本地模型文件夹路径（确保路径下包含 config.json 等文件）
model_local_path = "D:/models/bge-small-zh-v1.5"

print(f"🔄 正在从本地加载模型：{model_local_path} ...")

# 2. 修改初始化代码：直接传入路径，不再需要 cache_folder
model = SentenceTransformer(model_local_path)


# --- 以下代码保持不变 ---
class CustomEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=8
        ).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


embeddings = CustomEmbeddings(model)
print("✅ 本地模型加载完成！")

# ========== 第五步：分批嵌入并存储 ==========
print(f"💾 开始嵌入并存储到: {PERSIST_DIR}")
os.makedirs(PERSIST_DIR, exist_ok=True)

batch_size = 50  # 每批处理50个chunk
vectorstore = None

total_batches = (len(chunks) + batch_size - 1) // batch_size

for i in range(0, len(chunks), batch_size):
    batch_num = i // batch_size + 1
    batch = chunks[i : i + batch_size]
    print(f"   处理批次 {batch_num}/{total_batches} ({len(batch)} chunks)...")

    if vectorstore is None:
        # 第一批：创建数据库
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="sculpting_in_time",
        )
    else:
        # 后续批次：追加
        vectorstore.add_documents(batch)

    # 每批处理后强制垃圾回收
    gc.collect()

# 持久化到磁盘
vectorstore.persist()
print(f"✅ 向量数据库已保存到: {PERSIST_DIR}")

# ========== 第六步：彻底释放模型内存 ==========
print("🧹 正在释放模型内存...")

# 删除模型对象
del embeddings
del vectorstore

# 强制垃圾回收
gc.collect()


print("✅ 模型内存已完全释放！")
print("=" * 50)
print("🎉 索引构建完成！")
print(f"   向量数据库位置: {PERSIST_DIR}")
print("   现在可以关闭此程序，后续查询无需重新嵌入。")
