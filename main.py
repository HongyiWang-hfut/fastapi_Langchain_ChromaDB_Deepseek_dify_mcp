"""Campus Q&A assistant — step 1: minimal FastAPI app."""

from fastapi import FastAPI

app = FastAPI(
    title="Campus Smart Q&A Assistant",
    description="Step 1: Hello World API",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Hello World", "service": "campus-qa-assistant"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
