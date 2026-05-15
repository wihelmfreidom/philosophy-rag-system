from fastapi import FastAPI  # FastAPI 是一个为你的 API 提供了所有功能的 Python 类。
import uvicorn

app = (
    FastAPI()
)  # 这个实例将是创建你所有 API 的主要交互对象。这个 app 同样在如下命令中被 uvicorn 所引用


@app.get("/")
async def root():
    return {"message": "Hello yuan"}


@app.get("/king")
async def root():
    return {"message": "Hello king"}


@app.post(
    "/items",
    tags=["AAA"],
    summary="this is summary",
    description="this is description",
    response_description="this is response_description",
    deprecated=False,
)
async def root():
    return {"items": "items data"}


if __name__ == "__main__":

    uvicorn.run("quickstart:app", host="127.0.0.1", port=8080, reload=True)
