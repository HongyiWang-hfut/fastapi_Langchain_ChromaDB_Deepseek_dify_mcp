"""兼容入口：将 FastAPI app 暴露给 `uvicorn main:app`。"""

from app.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
