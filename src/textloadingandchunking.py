from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 第一步：加载PDF文件
pdf_path = "src\SculptingInTime.pdf"  # 替换为您的文件路径

# 使用PyMuPDFLoader（推荐用于中文PDF，保留页面结构较好）
loader = PyMuPDFLoader(pdf_path)

# 加载文档
documents = loader.load()

# 查看加载结果
print(f"总共加载了 {len(documents)} 页")
print(f"第17页内容预览（前200字符）：\n{documents[16].page_content[:200]}")
print(f"第17页元数据：{documents[16].metadata}")


# 第一步：合并所有文本，但保留页面映射
full_text = "\n\n".join([doc.page_content for doc in documents])

# 第二步：使用递归字符切分器，优先在文章边界处切分
# 对于随笔集，设置较大的 chunk_size，优先保护段落完整性
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # 每块约800字（考虑到中文密度，约等于英文1500字符的信息量）
    chunk_overlap=150,  # 重叠150字，保持上下文连贯
    separators=[
        "\n\n\n",
        "\n\n",
        "\n",
        "。",
        "，",
        " ",
        "",
    ],  # 优先在段落、句子边界切分
    length_function=len,  # 使用字符数而非token数（中文场景更直观）
    is_separator_regex=False,
)

# 执行分块
chunks = text_splitter.split_documents(documents)

print(f"原文档页数：{len(documents)}")
print(f"分块后数量：{len(chunks)}")
print(f"\n第12块内容预览：\n{chunks[11].page_content[:300]}...")
print(f"\n第12块元数据：{chunks[11].metadata}")
print(f"\n第13块内容预览：\n{chunks[12].page_content[:300]}...")
print(f"\n第13块元数据：{chunks[12].metadata}")
