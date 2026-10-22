"""Protocol layer: A2A 1.0 HTTP+JSON surface for the invoice action agent.

Routes (relative to BASE_PATH, default /a2a; also served at the origin root):
  GET  /.well-known/agent-card.json   (origin level, public)
  POST {base}/message:send
  GET  {base}/tasks/{id}
  GET  {base}/tasks
  POST {base}/tasks/{id}:cancel
"""
import asyncio
import base64
import copy
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from decide import decide_batch, to_proposal, LLM_MODEL
from store import Store, dumps

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("agent")

A2A_MT = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"
MAX_BYTES = 512 * 1024

BASE_PATH = "/" + os.environ.get("BASE_PATH", "/a2a").strip("/")
BASE_PATH = "" if BASE_PATH == "/" else BASE_PATH
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").strip()
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()

TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
STATES = {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED",
          "TASK_STATE_AUTH_REQUIRED"} | TERMINAL

store = Store()
_locks: dict[str, asyncio.Lock] = {}


def lock_for(key: str) -> asyncio.Lock:
    lk = _locks.get(key)
    if lk is None:
        lk = _locks[key] = asyncio.Lock()
    return lk


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------------ responses
class ApiError(Exception):
    def __init__(self, http: int, reason: str, message: str, grpc: str, headers=None):
        self.http, self.reason, self.message, self.grpc, self.headers = http, reason, message, grpc, headers


def a2a_response(obj, status: int = 200, headers=None, media=A2A_MT) -> Response:
    return Response(dumps(obj).encode("utf-8"), status_code=status, media_type=media, headers=headers)


def error_response(e: ApiError) -> Response:
    body = {"error": {
        "code": e.http, "status": e.grpc, "message": e.message,
        "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                     "reason": e.reason, "domain": "a2a-protocol.org"}],
    }}
    return a2a_response(body, e.http, e.headers)


def not_found():
    return ApiError(404, "TASK_NOT_FOUND", "Task not found", "NOT_FOUND")


def bad(message: str, reason: str = "INVALID_PARAMS"):
    return ApiError(400, reason, message, "INVALID_ARGUMENT")


# ------------------------------------------------------------------ request checks
def principal_of(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    m = re.fullmatch(r"[Bb]earer ([^\s]+)", auth.strip())
    if not m:
        raise ApiError(401, "UNAUTHENTICATED", "Missing or invalid bearer token", "UNAUTHENTICATED",
                       {"WWW-Authenticate": "Bearer"})
    return hashlib.sha256(m.group(1).encode()).hexdigest()


def check_version(request: Request):
    v = request.headers.get("a2a-version")
    if v is None:
        v = request.query_params.get("A2A-Version") or request.query_params.get("a2a-version")
    v = (v or "").strip()
    if not re.fullmatch(r"1\.0(\.\d+)?", v):
        raise ApiError(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version; this agent supports 1.0",
                       "FAILED_PRECONDITION")


def check_accept(request: Request):
    acc = request.headers.get("accept")
    if not acc:
        return
    types = [a.split(";")[0].strip().lower() for a in acc.split(",")]
    ok = {"*/*", "application/*", A2A_MT, "application/json"}
    if not any(t in ok for t in types):
        raise ApiError(406, "CONTENT_TYPE_NOT_SUPPORTED", "Response media type not acceptable",
                       "INVALID_ARGUMENT")


def check_content_type(request: Request, body: bytes, optional: bool = False):
    ct = request.headers.get("content-type")
    if optional and not body.strip() and not ct:
        return
    media = (ct or "").split(";")[0].strip().lower()
    if media != A2A_MT:
        raise ApiError(415, "CONTENT_TYPE_NOT_SUPPORTED", "Content-Type must be application/a2a+json",
                       "INVALID_ARGUMENT")


async def read_json(request: Request, body: bytes):
    try:
        data = json.loads(body.decode("utf-8")) if body.strip() else {}
    except (ValueError, UnicodeDecodeError):
        raise bad("Request body is not valid JSON", "PARSE_ERROR")
    if not isinstance(data, dict):
        raise bad("Request body must be a JSON object")
    return data


def parse_history_length(v):
    if v is None or v == "":
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise bad("historyLength must be an integer")
    if n < 0:
        raise bad("historyLength must be non-negative")
    return n


# ------------------------------------------------------------------ views
def task_view(task: dict, history_length=None, include_artifacts=True, include_history=True) -> dict:
    t = copy.deepcopy(task)
    t.get("metadata", {}).pop("_internal", None)
    if not include_history:
        t.pop("history", None)
    elif history_length is not None:
        if history_length == 0:
            t.pop("history", None)
        else:
            t["history"] = t.get("history", [])[-history_length:]
    if not include_artifacts:
        t.pop("artifacts", None)
    return t


def fit(obj_builder, task: dict, **kw) -> Response:
    resp = obj_builder(task_view(task, **kw))
    body = dumps(resp).encode("utf-8")
    if len(body) > MAX_BYTES and task.get("history"):
        # Keep the first (initial) and last message only.
        t = task_view(task, **kw)
        h = t.get("history", [])
        if len(h) > 2:
            t["history"] = [h[0], h[-1]]
        body = dumps(obj_builder(t)).encode("utf-8")
    return Response(body, status_code=200, media_type=A2A_MT)


def agent_message(text: str, task_id: str, context_id: str) -> dict:
    return {"messageId": f"agent-{uuid.uuid4()}", "role": "ROLE_AGENT", "taskId": task_id,
            "contextId": context_id, "parts": [{"text": text, "mediaType": "text/plain"}]}


# ------------------------------------------------------------------ agent card
def base_urls(request: Request) -> list:
    if PUBLIC_BASE_URL:
        first = PUBLIC_BASE_URL
    else:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
        host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
        first = f"{proto}://{host.split(',')[0].strip()}{BASE_PATH}/"
    alt = first[:-1] if first.endswith("/") else first + "/"
    return [first, alt]


def agent_card(request: Request) -> dict:
    urls = base_urls(request)
    origin = re.match(r"^(https?://[^/]+)", urls[0]).group(1) if re.match(r"^https?://", urls[0]) else ""
    modes_in = [BATCH_MT, RESULTS_MT]
    modes_out = [PROPOSALS_MT, RECEIPTS_MT]
    return {
        "name": "Invoice Action Agent",
        "description": ("Reads invoice claim packages, proposes exactly one evidence-backed business action "
                        "per package (settle, request approval, hold, reject duplicate, open exception), "
                        "and executes only actions the counterparty has accepted, with idempotent, "
                        "persisted task state so retries never repeat a payment."),
        "version": "1.0.0",
        "supportedInterfaces": [{"url": u, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
                                for u in urls],
        "provider": {"organization": "Invoice Action Agent (course project)", "url": origin or urls[0]},
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {
            "scheme": "Bearer", "description": "Each bearer token is a separate principal."}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": modes_in,
        "defaultOutputModes": modes_out,
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": ("Choose one typed action per invoice package with exact decisive evidence "
                            "references, wait for accepted results, then record receipts for accepted "
                            "actions only."),
            "tags": ["invoice", "accounts-payable", "payments", "approval", "duplicate-detection", "a2a"],
            "examples": ["Propose actions for an invoice claim batch",
                         "Execute accepted invoice actions and return receipts"],
            "inputModes": modes_in,
            "outputModes": modes_out,
        }],
    }


# ------------------------------------------------------------------ handlers
async def send_message(request: Request, principal: str, body: bytes) -> Response:
    data = await read_json(request, body)
    msg = data.get("message")
    if not isinstance(msg, dict):
        raise bad("message is required")
    mid = msg.get("messageId")
    if not isinstance(mid, str) or not mid.strip():
        raise bad("message.messageId is required")
    if msg.get("role") != "ROLE_USER":
        raise bad("message.role must be ROLE_USER")
    parts = msg.get("parts")
    if not isinstance(parts, list) or not parts or not all(isinstance(p, dict) for p in parts):
        raise bad("message.parts must be a non-empty array of Part objects")
    config = data.get("configuration") if isinstance(data.get("configuration"), dict) else {}
    hl = parse_history_length(config.get("historyLength"))
    mhash = hashlib.sha256(canonical(msg).encode()).hexdigest()

    async with lock_for(f"m:{principal}:{mid}"):
        rec = store.message_record(principal, mid)
        if rec:
            if rec[0] != mhash:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT",
                               "IDEMPOTENCY_CONFLICT: messageId was already used with different content",
                               "ABORTED")
            task = store.get_task(principal, rec[1])
            if task is None:
                raise not_found()
            return fit(lambda t: {"task": t}, task, history_length=hl)
        if msg.get("taskId"):
            task = await continue_task(principal, msg, mid, mhash)
        else:
            task = await start_task(principal, msg, mid, mhash, config)
        return fit(lambda t: {"task": t}, task, history_length=hl)


def find_part(parts, media_type):
    for p in parts:
        if isinstance(p, dict) and str(p.get("mediaType", "")).split(";")[0].strip().lower() == media_type:
            return p
    return None


async def start_task(principal, msg, mid, mhash, config) -> dict:
    accepted = config.get("acceptedOutputModes")
    if isinstance(accepted, list) and accepted and PROPOSALS_MT not in accepted and "*/*" not in accepted:
        raise ApiError(400, "CONTENT_TYPE_NOT_SUPPORTED",
                       "acceptedOutputModes must include the proposals media type", "INVALID_ARGUMENT")
    part = find_part(msg["parts"], BATCH_MT)
    if part is None:
        raise ApiError(400, "CONTENT_TYPE_NOT_SUPPORTED",
                       f"Initial message needs a {BATCH_MT} data part", "INVALID_ARGUMENT")
    batch = part.get("data")
    if not isinstance(batch, dict):
        raise bad("batch part must carry a data object")
    batch_id = batch.get("batchId")
    pkgs = batch.get("packages")
    if not isinstance(batch_id, str) or not batch_id:
        raise bad("batchId is required")
    if not isinstance(pkgs, list) or not pkgs or not all(isinstance(p, dict) for p in pkgs):
        raise bad("packages must be a non-empty array")
    from decide import package_id
    pids = [package_id(p) for p in pkgs]
    if any(not p for p in pids) or len(set(pids)) != len(pids):
        raise bad("every package needs a unique packageId")

    task_id = str(uuid.uuid4())
    ctx = msg.get("contextId") if isinstance(msg.get("contextId"), str) and msg.get("contextId") else str(uuid.uuid4())
    store.log_inbound(principal, "batch", task_id, batch)

    decisions = await decide_batch(batch, store)
    proposals = []
    for pid in pids:
        action_id = "act_" + hashlib.sha256(f"{principal}|{task_id}|{pid}".encode()).hexdigest()[:28]
        proposals.append(to_proposal(pid, action_id, decisions[pid]))
    sources = {}
    for pid in pids:
        sources[decisions[pid].get("source", "?")] = sources.get(decisions[pid].get("source", "?"), 0) + 1

    task = {
        "id": task_id,
        "contextId": ctx,
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED",
            "message": agent_message(
                f"Proposed {len(proposals)} actions for batch {batch_id}. Awaiting results; "
                "no action has been executed.", task_id, ctx),
            "timestamp": now_iso(),
        },
        "artifacts": [{
            "artifactId": f"proposals-{task_id}",
            "name": "invoice-action-proposals",
            "parts": [{"mediaType": PROPOSALS_MT, "data": {"batchId": batch_id, "proposals": proposals}}],
        }],
        "history": [msg],
        "metadata": {"batchId": batch_id},
    }
    store.save_task(principal, task, mid, mhash, create=True)
    store.log_inbound(principal, "proposals", task_id, {"batchId": batch_id, "sources": sources,
                                                         "model": LLM_MODEL, "proposals": proposals})
    return task


async def continue_task(principal, msg, mid, mhash) -> dict:
    task_id = msg.get("taskId")
    if not isinstance(task_id, str) or store.get_task(principal, task_id) is None:
        raise not_found()
    async with lock_for(f"t:{task_id}"):
        task = store.get_task(principal, task_id)
        if task is None:
            raise not_found()
        state = task["status"]["state"]
        if state in TERMINAL:
            raise ApiError(409, "UNSUPPORTED_OPERATION", "Task is in a terminal state", "FAILED_PRECONDITION")
        if state != "TASK_STATE_INPUT_REQUIRED":
            raise ApiError(409, "UNSUPPORTED_OPERATION", "Task is not awaiting input", "FAILED_PRECONDITION")
        ctx = msg.get("contextId")
        if ctx is not None and ctx != task["contextId"]:
            raise bad("contextId does not match the task")
        part = find_part(msg["parts"], RESULTS_MT)
        if part is None or not isinstance(part.get("data"), dict):
            raise ApiError(400, "CONTENT_TYPE_NOT_SUPPORTED",
                           f"Continuation needs a {RESULTS_MT} data part", "INVALID_ARGUMENT")
        data = part["data"]
        prop_data = task["artifacts"][0]["parts"][0]["data"]
        if data.get("batchId") != prop_data["batchId"]:
            raise bad("batchId does not match the task")
        results = data.get("results")
        if not isinstance(results, list) or not results:
            raise bad("results must be a non-empty array")
        props = {p["packageId"]: p for p in prop_data["proposals"]}
        seen = {}
        for r in results:
            if not isinstance(r, dict):
                raise bad("each result must be an object")
            pid = r.get("packageId")
            if not isinstance(pid, str) or pid not in props or pid in seen:
                raise bad("result packageId does not match a proposal")
            p = props[pid]
            if r.get("actionId") != p["actionId"]:
                raise bad("result actionId does not match the proposal")
            if r.get("action") != p["action"]:
                raise bad("result action does not match the proposal")
            outcome = r.get("outcome")
            if outcome not in ("ACCEPTED", "REJECTED"):
                raise bad("result outcome must be ACCEPTED or REJECTED")
            if outcome == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce.strip():
                    raise bad("accepted result needs a receiptNonce")
            seen[pid] = r
        if set(seen) != set(props):
            raise bad("results must cover every proposal exactly once")

        executions = []
        for p in prop_data["proposals"]:
            r = seen[p["packageId"]]
            if r["outcome"] != "ACCEPTED":
                continue  # rejected proposals are never executed
            prior = store.executed(principal, p["actionId"])
            if prior:  # ledger: never execute the same action twice
                executions.append(prior)
                continue
            executions.append({
                "packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                "receiptNonce": r["receiptNonce"], "facts": copy.deepcopy(p["facts"]),
                "evidenceRefs": list(p["evidenceRefs"]),
            })
        new = copy.deepcopy(task)
        new["history"].append(msg)
        new["artifacts"].append({
            "artifactId": f"receipts-{task_id}",
            "name": "invoice-action-receipts",
            "parts": [{"mediaType": RECEIPTS_MT,
                       "data": {"batchId": prop_data["batchId"], "executions": executions}}],
        })
        new["status"] = {
            "state": "TASK_STATE_COMPLETED",
            "message": agent_message(
                f"Executed {len(executions)} accepted action(s); "
                f"{len(results) - len(executions)} rejected proposal(s) were not executed.",
                task_id, task["contextId"]),
            "timestamp": now_iso(),
        }
        store.save_task(principal, new, mid, mhash, executions=executions)
        store.log_inbound(principal, "results", task_id, data)
        return new


async def cancel_task(principal: str, task_id: str) -> Response:
    if store.get_task(principal, task_id) is None:
        raise not_found()
    async with lock_for(f"t:{task_id}"):
        task = store.get_task(principal, task_id)
        if task is None:
            raise not_found()
        if task["status"]["state"] in TERMINAL:
            raise ApiError(409, "TASK_NOT_CANCELABLE", "Task is in a terminal state and cannot be canceled",
                           "FAILED_PRECONDITION")
        new = copy.deepcopy(task)
        new["status"] = {"state": "TASK_STATE_CANCELED",
                         "message": agent_message("Task canceled; no actions were executed.",
                                                  task_id, task["contextId"]),
                         "timestamp": now_iso()}
        store.save_task(principal, new)
        return fit(lambda t: t, new)


def list_tasks(request: Request, principal: str) -> Response:
    q = request.query_params
    status = q.get("status") or None
    if status and status not in STATES:
        raise bad("unknown status filter")
    try:
        page_size = int(q.get("pageSize") or 50)
    except ValueError:
        raise bad("pageSize must be an integer")
    if page_size < 1 or page_size > 100:
        raise bad("pageSize must be between 1 and 100")
    offset = 0
    tok = q.get("pageToken")
    if tok:
        try:
            offset = int(base64.urlsafe_b64decode(tok.encode()).decode())
            assert offset >= 0
        except Exception:
            raise bad("invalid pageToken")
    hl = parse_history_length(q.get("historyLength"))
    include_artifacts = (q.get("includeArtifacts") or "").lower() == "true"
    all_tasks = store.list_tasks(principal, q.get("contextId") or None, status)
    page, size = [], 200
    i = offset
    while i < len(all_tasks) and len(page) < page_size:
        v = task_view(all_tasks[i], history_length=hl, include_artifacts=include_artifacts,
                      include_history=hl is not None)
        b = len(dumps(v).encode())
        if page and size + b > MAX_BYTES - 4096:
            break
        page.append(v)
        size += b + 1
        i += 1
    nxt = base64.urlsafe_b64encode(str(i).encode()).decode() if i < len(all_tasks) else ""
    return a2a_response({"tasks": page, "nextPageToken": nxt, "pageSize": len(page),
                         "totalSize": len(all_tasks)})


# ------------------------------------------------------------------ admin (debug)
def admin(request: Request, path: str) -> Response:
    key = request.query_params.get("key") or request.headers.get("x-admin-key")
    if not ADMIN_TOKEN or key != ADMIN_TOKEN:
        return Response(b'{"error":"not found"}', 404, media_type="application/json")
    if path == "/admin/inbound":
        limit = int(request.query_params.get("limit") or 20)
        return a2a_response(store.recent_inbound(limit, request.query_params.get("kind")),
                            media="application/json")
    if path == "/admin/clear-decisions" and request.method == "POST":
        store.clear_decisions()
        return a2a_response({"ok": True}, media="application/json")
    return Response(b'{"error":"not found"}', 404, media_type="application/json")


# ------------------------------------------------------------------ dispatcher
TASK_GET = re.compile(r"/tasks/([^/:]+)")
TASK_CANCEL = re.compile(r"/tasks/([^/:]+):cancel")
TASK_SUB = re.compile(r"/tasks/([^/:]+):subscribe")


async def dispatch(request: Request) -> Response:
    path = re.sub(r"/{2,}", "/", request.url.path)
    method = request.method
    try:
        if path.rstrip("/") in ("/.well-known/agent-card.json", BASE_PATH + "/.well-known/agent-card.json"):
            if method not in ("GET", "HEAD"):
                raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed", "UNIMPLEMENTED")
            return a2a_response(agent_card(request), media="application/json",
                                headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})
        if path in ("/", "/healthz") and method == "GET":
            return a2a_response({"status": "ok", "agentCard": "/.well-known/agent-card.json",
                                 "base": BASE_PATH + "/"}, media="application/json")
        if path.startswith("/admin/"):
            return admin(request, path.rstrip("/"))

        rel = path
        if BASE_PATH and (path == BASE_PATH or path.startswith(BASE_PATH + "/")):
            rel = path[len(BASE_PATH):]
        rel = rel.rstrip("/") or "/"

        route = None
        if rel in ("/message:send", "/message:stream"):
            route = ("send" if rel == "/message:send" else "stream", None, "POST")
        elif rel == "/tasks":
            route = ("list", None, "GET")
        elif TASK_CANCEL.fullmatch(rel):
            route = ("cancel", TASK_CANCEL.fullmatch(rel).group(1), "POST")
        elif TASK_SUB.fullmatch(rel):
            route = ("subscribe", TASK_SUB.fullmatch(rel).group(1), None)
        elif TASK_GET.fullmatch(rel):
            route = ("get", TASK_GET.fullmatch(rel).group(1), "GET")
        if route is None:
            raise ApiError(404, "NOT_FOUND", "Not found", "NOT_FOUND")
        name, task_id, want = route

        principal = principal_of(request)
        check_version(request)
        check_accept(request)
        if want and method != want:
            raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed", "UNIMPLEMENTED")

        if name == "send":
            body = await request.body()
            check_content_type(request, body)
            return await send_message(request, principal, body)
        if name in ("stream", "subscribe"):
            raise ApiError(400, "UNSUPPORTED_OPERATION", "Streaming is not supported", "FAILED_PRECONDITION")
        if name == "list":
            return list_tasks(request, principal)
        if name == "get":
            task = store.get_task(principal, task_id)
            if task is None:
                raise not_found()
            hl = parse_history_length(request.query_params.get("historyLength"))
            return fit(lambda t: t, task, history_length=hl)
        if name == "cancel":
            body = await request.body()
            check_content_type(request, body, optional=True)
            if body.strip():
                await read_json(request, body)
            return await cancel_task(principal, task_id)
        raise ApiError(404, "NOT_FOUND", "Not found", "NOT_FOUND")
    except ApiError as e:
        return error_response(e)
    except Exception:  # noqa: BLE001
        log.exception("unhandled error")
        return error_response(ApiError(500, "INTERNAL", "Internal error", "INTERNAL"))


ALL = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
app = Starlette(routes=[Route("/{path:path}", dispatch, methods=ALL)])
