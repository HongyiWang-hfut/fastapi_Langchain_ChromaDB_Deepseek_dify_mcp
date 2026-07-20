"""知识库上传/检索/删除 端到端验证（绕过 Git Bash 中文编码坑，用标准 UTF-8）。"""
import io
import json
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:8000"
KEY = "campus-qa-dev-key"
HDR = {"X-API-Key": KEY}

FNAME = "e2e_test_紫蓬山.txt"
# 紫蓬山校区 = 宣城校区别称
CONTENT = "合肥工业大学宣城校区别称紫蓬山校区，位于安徽省宣城市。紫蓬山校区以工为主，多学科协调发展。"


def post_multipart(path, field_name, filename, data_bytes):
    boundary = "----e2eboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + data_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={**HDR, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def get_json(path, params=None):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def delete_json(path, params=None):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HDR, method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


print("=== 1) upload ===")
st, up = post_multipart("/knowledge/upload", "file", FNAME, CONTENT.encode("utf-8"))
print("status", st, "resp", up)

print("=== 2) search 紫蓬山校区 ===")
st, se = get_json("/knowledge/search", {"q": "紫蓬山校区", "k": 5})
print("count", se["count"])
for h in se["hits"]:
    print("  src=", h.get("metadata", {}).get("source"),
          "sf=", h.get("metadata", {}).get("source_file"),
          "|", h["content"][:28])

print("=== 3) list files, find test entry ===")
st, lf = get_json("/knowledge/files")
test_entry = next((f for f in lf["files"] if f.get("filename") == FNAME), None)
print("found entry:", test_entry)

print("=== 4) delete by key ===")
if test_entry:
    st, de = delete_json("/knowledge/files", {"key": test_entry["key"]})
    print("status", st, "resp", de)

    print("=== 5) search again (should be gone) ===")
    st, se2 = get_json("/knowledge/search", {"q": "紫蓬山校区", "k": 5})
    still_there = any(h.get("metadata", {}).get("source_file") == FNAME for h in se2["hits"])
    print("count", se2["count"], "still_there=", still_there)
    print("RESULT:", "PASS" if (not still_there and (test_entry is not None)) else "FAIL")
else:
    print("RESULT: FAIL (entry not found)")
