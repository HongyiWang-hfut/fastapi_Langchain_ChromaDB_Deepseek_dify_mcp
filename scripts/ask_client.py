import json
import sys
import urllib.request


def main() -> None:
    question = "图书馆借书能借多少册？"
    if len(sys.argv) > 1:
        question = sys.argv[1]

    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        print(body)


if __name__ == "__main__":
    main()

