# philosophy-rag-system
A native-implementation RAG Q&amp;A system for philosophy texts, featuring hybrid retrieval (keyword + vector similarity) and two-level chunking. Built with FastAPI + Chroma.
# Philosophy RAG Q&A System

&gt; 一个面向哲学长文本的原生实现RAG问答系统，支持向量检索与两级分块策略。

## 为什么做这个项目？

传统RAG框架（如LangChain）对长文本哲学文献的处理效果不佳：
- 哲学文本逻辑链长，简单分块会破坏论证连续性

## 核心设计

### 两级分块策略（Two-Level Chunking）
- **粗粒度**：按章节/段落分割，保留完整论证结构
- **细粒度**：在粗块内按语义窗口二次切分，平衡召回与精度


### 3. 技术栈
- FastAPI（高性能API服务）
- Chroma（向量数据库）
- Sentence-Transformers（文本向量化）
