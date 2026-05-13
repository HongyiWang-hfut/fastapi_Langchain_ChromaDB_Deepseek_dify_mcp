# Campus Smart Q&A Assistant

Minimal FastAPI RAG service that retrieves context from Chroma and asks DeepSeek via OpenAI-compatible API.

## Environment

Create a `.env` file next to `main.py` with:

- `DEEPSEEK_API_KEY=...`
- `DEEPSEEK_BASE_URL=https://api.deepseek.com/v1` (optional)
- `DEEPSEEK_MODEL=deepseek-chat` (optional)

## Run

```bash
uvicorn main:app --reload
```

## Test (curl)

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"图书馆借书能借多少册？\"}"
```

## Tiny client

```bash
python scripts/ask_client.py "图书馆借书能借多少册？"
```

