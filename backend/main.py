from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator
import json
import uvicorn
from contextlib import asynccontextmanager

# 从 rag_core 导入（复用你的代码）
from rag_core import initialize_rag, RAGSystem

# 全局状态
rag_system: RAGSystem = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    rag_system = initialize_rag()
    yield
    print("🧹 清理资源...")
    import gc

    gc.collect()


app = FastAPI(title="哲学史问答系统", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    title: str
    trigger_snippet: str


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vectorstore_count": (
            rag_system.vectorstore._collection.count() if rag_system else 0
        ),
    }


@app.post("/chat/stream")
async def chat_stream(request: QuestionRequest):
    try:
        # 获取来源信息
        source_docs = rag_system.retriever.invoke(request.question)
        sources_data = [
            {
                "title": doc.metadata.get("chapter_title", "未知"),
                "trigger_snippet": doc.metadata.get("retrieved_child", "")[:100],
            }
            for doc in source_docs
        ]

        async def generate():
            async for chunk in rag_system.rag_chain.astream(request.question):
                yield f"data: {chunk}\n\n"

            yield f"event: sources\ndata: {json.dumps(sources_data, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/index-corruption")
async def check_index_corruption():
    """检查索引是否所有子块都指向相同父块"""
    # 获取所有子块
    all_data = rag_system.vectorstore._collection.get()

    parent_ids = {}
    titles = {}

    for metadata in all_data["metadatas"]:
        pid = metadata.get("parent_id")
        title = metadata.get("chapter_title")
        parent_ids[pid] = parent_ids.get(pid, 0) + 1
        titles[title] = titles.get(title, 0) + 1

    return {
        "total_chunks": len(all_data["ids"]),
        "unique_parent_ids": len(parent_ids),
        "parent_id_distribution": parent_ids,  # 看是否只有一个 ID 被大量使用
        "unique_titles": len(titles),
        "title_distribution": titles,  # 看是否只有"尽心知性"等几个标题
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
