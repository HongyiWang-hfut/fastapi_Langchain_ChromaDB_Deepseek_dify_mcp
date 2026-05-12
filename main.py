"""校园智能问答助手：FastAPI HTTP 入口，启动时加载项目根 ``.env``。"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

_env_path = Path(__file__).resolve().parent / ".env"  # 与 main.py 同目录，不随终端 cd 变化
load_dotenv(_env_path)  # 将 KEY 写入 os.environ，供后续路由与其它模块使用

app = FastAPI(  # ASGI 应用实例，供 uvicorn 挂载
    title="Campus Smart Q&A Assistant",
    description="Step 1: Hello World API",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """根路径探活。"""
    return {"message": "Hello World", "service": "campus-qa-assistant"}


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查（网关/编排常用）。"""
    return {"status": "ok"}
