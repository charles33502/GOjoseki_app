from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from werkzeug.exceptions import HTTPException  # 404/405/400... 這種 HTTP 層例外
import json
import requests
import os
import traceback
import time
import uuid
import logging

# ----------------------------
# 讀取 .env：本機開發時把 GEMINI_API_KEY 等環境變數載入
# ----------------------------
load_dotenv()  # 讀取 .env 到環境變數（本機開發方便）

# ----------------------------
# 建立 Flask app + 設定 CORS：允許前端網域呼叫後端 API
# ----------------------------
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://go-joseki-interactive.vercel.app"])


# ----------------------------
# 1) Prometheus metrics
# ----------------------------
# Prometheus 指標宣告：只宣告「型別/名稱/labels/buckets」，真正數值在 runtime inc/observe
# ----------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
#  # HTTP 請求數（method/endpoint/status）
#  # Counter：只會累加，用 rate()/increase() 看速度或區間增量

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 3, 5, 8, 13, 21),
)
#  # HTTP 延遲分佈（method/endpoint），會產生 _bucket/_count/_sum
#  # buckets 決定 Grafana/PromQL 算 p95/p99 的解析度與範圍


HTTP_EXCEPTIONS_TOTAL = Counter(
    "http_exceptions_total",
    "Total exceptions",
    ["endpoint", "exception_type"],
)
#  # 後端未預期例外計數（endpoint/exception_type）
#  # 這裡通常對應「500 / bug」，不是 404 那種

GEMINI_REQUESTS_TOTAL = Counter(
    "gemini_requests_total",
    "Total Gemini API requests",
    ["status"],  # success / http_error / timeout / other_error
)
#  # Gemini 呼叫次數（依 status 分桶）
#  # 之後可以算成功率、429/timeout 比例等

GEMINI_REQUEST_DURATION_SECONDS = Histogram(
    "gemini_request_duration_seconds",
    "Gemini API request duration in seconds",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 3, 5, 8, 13, 21),
)
#  # Gemini 呼叫延遲分佈（observe 在 finally）


# ----------------------------
# 2) Structured logging (JSON)
# ----------------------------
# Structured logging：用 JSON 一行一筆，方便之後在 Render/Cloud Logging 搜索/彙整
# ----------------------------

logger = logging.getLogger("gojoseki") # 自訂 logger 名稱，方便在平台上篩選
logger.setLevel(logging.INFO) # 設定 log 等級
handler = logging.StreamHandler() # 印到 stdout（平台通常會收集 stdout）
handler.setFormatter(logging.Formatter("%(message)s"))  # 輸出 JSON string
logger.handlers = [handler] # 只用這個 handler（避免重複輸出）

def log_event(event: str, **fields):   # 統一的 JSON log helper：每次呼叫就打一行 JSON
    payload = {
        "event": event,
        "ts": time.time(),
        "request_id": getattr(g, "request_id", None),
        "method": request.method if request else None,
        "path": request.path if request else None,
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))   # 確保中文不被轉成 \uXXXX


# ----------------------------
# 3) Load questions.json safely (避免部署路徑踩雷) 和 GEMINI_API_KEY
# ----------------------------
# 載入題庫 JSON：用「檔案相對 app.py 的絕對路徑」避免部署路徑不同導致找不到檔案
# 載入GEMINI_API_KEY：從環境變數讀取，部署前記得設定（Render 的 Secret Environment Variables）
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(BASE_DIR, "questions_gemini.json")

with open(QUESTIONS_PATH, encoding="utf-8") as file:
    questions_gemini = {q["id"]: q for q in json.load(file)}
print("✅ 成功載入 Gemini 題庫")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ----------------------------
# 4) Request middleware: request_id + latency + counters 
# ----------------------------
# Request middleware：
# - before_request：建立 request_id、記起始時間
# - after_request ：計算 latency、更新 HTTP metrics、回傳 header、寫 log
# ----------------------------

@app.before_request
def _before_request():
    g.start_time = time.perf_counter()
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4()) # 建 request_id：優先用上游 header，否則生成 UUID

@app.after_request
def _after_request(response):
    duration = time.perf_counter() - getattr(g, "start_time", time.perf_counter())
    endpoint = request.path  # 也可以用 request.endpoint，但 path 更直觀


    # metrics
    HTTP_REQUESTS_TOTAL.labels(request.method, endpoint, str(response.status_code)).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(request.method, endpoint).observe(duration)

    # response header
    response.headers["X-Request-ID"] = g.request_id  # 把 request_id 回傳給前端/上游，方便串 log

    # log
    log_event(
        "http_request",
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response

# ----------------------------
# 全域例外處理：
# - HTTPException(404/405/400...)：保留原狀態碼，只做簡短 log
# - 其他 Exception：算作後端錯誤(500)，更新 exceptions counter + 回 JSON
# ----------------------------
@app.errorhandler(Exception)
def _handle_exception(e):
    # 讓 404/405/400... 這種 HTTPException 保持原本狀態碼
    if isinstance(e, HTTPException):
        log_event(
            "http_exception",
            exception_type=type(e).__name__,
            code=e.code,
            message=str(e),
        )
        return e  # 交回給 Flask/werkzeug 自己回應（404 就會是 404）

    endpoint = request.path if request else "unknown"
    HTTP_EXCEPTIONS_TOTAL.labels(endpoint, type(e).__name__).inc()
    log_event(
        "exception",
        exception_type=type(e).__name__,
        message=str(e),
        traceback=traceback.format_exc(limit=8),
    )
    return jsonify({"error": "Internal server error", "request_id": getattr(g, "request_id", None)}), 500


# ----------------------------
# 5) Prometheus endpoint
# ----------------------------
# /metrics：Prometheus scrape 入口（瀏覽器打開也能直接看） 
# ----------------------------

@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ----------------------------
# 6) Gemini call with timeout + key check
# ----------------------------
# Gemini 呼叫封裝：
# - 檢查 API key
# - 設 timeout
# - 依結果更新 gemini_requests_total(status=...)
# - 不論成功失敗都 observe gemini_request_duration_seconds（finally）
# ----------------------------

def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        GEMINI_REQUESTS_TOTAL.labels("other_error").inc()
        return "伺服器未設定 GEMINI_API_KEY"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    t0 = time.perf_counter() # 觀察 Gemini 呼叫的實際延遲，包含網路和 Gemini 處理時間
    try:
        res = requests.post(url, headers=headers, json=data, timeout=(3, 20)) # 連線超時 3 秒、整體超時 20 秒（根據實測調整）
        res.raise_for_status()
        body = res.json()

        GEMINI_REQUESTS_TOTAL.labels("success").inc()
        return body["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.Timeout:
        GEMINI_REQUESTS_TOTAL.labels("timeout").inc()
        raise

    except requests.exceptions.HTTPError:
        GEMINI_REQUESTS_TOTAL.labels("http_error").inc()
        raise

    except Exception:
        GEMINI_REQUESTS_TOTAL.labels("other_error").inc()
        raise

    finally:
        GEMINI_REQUEST_DURATION_SECONDS.observe(time.perf_counter() - t0) # 無論成功失敗都觀察 Gemini 呼叫的延遲，方便後續分析 Gemini 成本和性能

# ----------------------------
# 7) Existing routes
# ----------------------------
# API routes：
# - GET /questions：回傳題目列表
# - POST /answer：用 Gemini 生成對使用者回答的回饋
# ----------------------------

@app.route("/questions", methods=["GET"])
def get_questions():
    return jsonify([
        {"qid": qid, "prompt": q["prompt"], "sgf": q["sgf"]}
        for qid, q in questions_gemini.items()
    ])

@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json() or {}
    qid = int(data.get("qid", 1))
    user_answer = (data.get("answer", "") or "").strip()

    q = questions_gemini.get(qid)
    if not q:
        return jsonify({"error": "題號不存在"}), 404

    a = q["analysis"]
    prompt = f"""題目：{q['prompt']}

使用者的回答是：「{user_answer}」

定石分析資料如下：
- 白棋棋位：{a['white point']}
- 黑棋棋位：{a['black point']}
- 白棋：{a['white']}
- 黑棋：{a['black']}
- 總結：{a['summary']}

請扮演一位親切的圍棋老師，幫忙分析這位使用者的回答。請針對他的回答內容進行說明，
包括是否合理、常見誤區、潛在想法或可改進之處。語氣請自然、鼓勵學習、避免使用「正確答案是…」這種說法，
字數控制在300字以內。
"""

    ai_reply = call_gemini(prompt)
    log_event("gemini_called", qid=qid, answer_len=len(user_answer))  # 加一個 log：每次 /answer 的 prompt 長度，方便後續成本分析
    return jsonify({"response": ai_reply, "request_id": g.request_id})


# ----------------------------
# 8) 本機啟動入口：python app.py
# ----------------------------
# - host=0.0.0.0 讓容器/Render 能對外綁定
# - port=10000 對應你本機測試的 http://127.0.0.1:10000
# - debug 由環境變數 FLASK_DEBUG 控制（true/false）
# ----------------------------
if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=10000, debug=debug_mode)

