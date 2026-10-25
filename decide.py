"""AI decision layer: read invoice packages, choose one action each, cite exact evidence.

Pipeline per batch:
  1. Segment each package into labelled paragraphs and collect the bracketed refs in each.
  2. Look up the decision cache by canonical package content (packageId excluded).
  3. Send uncached packages to an OpenAI-compatible chat model in batched calls.
  4. Validate every model answer deterministically: the action must come from the closed set,
     the evidence refs must exist verbatim in one paragraph of the package, the amount must be
     an integer in minor units, and the rationale is rebuilt to satisfy the contract.
  5. A keyword heuristic covers model failure, so a typed proposal is always returned.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import httpx

log = logging.getLogger("decide")

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
REF_RE = re.compile(r"\[([^\[\]\n]{1,160})\]")
DECOY_RE = re.compile(
    r"archiv|training|example|illustrat|sample|template|historical|precedent|decoy|"
    r"for instance|hypothetical|tutorial|quiz|exercise|do not use|not applicable to this",
    re.I,
)
COVER_RE = re.compile(r"cover", re.I)

# ISO 4217 minor-unit exponents that are not 2.
CURRENCY_EXP = {
    "JPY": 0, "KRW": 0, "VND": 0, "CLP": 0, "ISK": 0, "UGX": 0, "XAF": 0, "XOF": 0, "PYG": 0,
    "KWD": 3, "BHD": 3, "OMR": 3, "JOD": 3, "TND": 3, "LYD": 3, "IQD": 3,
}

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
LLM_API_KEY = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("AIPIPE_TOKEN") or "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
LLM_CHUNK = max(1, int(os.environ.get("LLM_CHUNK", "4")))
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "22"))
DECIDE_BUDGET = float(os.environ.get("DECIDE_BUDGET", "34"))
REF_STYLE = os.environ.get("REF_STYLE", "inner")  # "inner" -> ABC-1 ; "bracketed" -> [ABC-1]
PROMPT_VERSION = "v5"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def package_id(pkg: dict) -> str:
    for k in ("packageId", "package_id", "id"):
        if isinstance(pkg.get(k), (str, int)):
            return str(pkg[k])
    return ""


def batch_context(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if k not in ("packages", "batchId")}


def decision_key(batch: dict, pkg: dict) -> str:
    content = {k: v for k, v in pkg.items() if k not in ("packageId", "package_id")}
    raw = canonical({"ctx": batch_context(batch), "pkg": content, "v": PROMPT_VERSION,
                     "m": LLM_MODEL, "r": REF_STYLE})
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------- segmentation
class Segmented:
    def __init__(self, pkg: dict):
        self.paras = []  # {"pid","path","text","refs"}
        self.lines = []
        self._walk(pkg, "", 0)
        self.text = "\n".join(self.lines)
        self.all_refs = []
        for p in self.paras:
            for r in p["refs"]:
                if r not in self.all_refs:
                    self.all_refs.append(r)

    def _add_para(self, chunk: str, path: str, ind: str):
        refs = []
        for r in REF_RE.findall(chunk):
            r = r.strip()
            if r and r not in refs:
                refs.append(r)
        if not refs:
            self.lines.append(ind + "  " + chunk.replace("\n", "\n" + ind + "  "))
            return
        pid = f"P{len(self.paras) + 1}"
        self.paras.append({"pid": pid, "path": path, "text": chunk, "refs": refs})
        self.lines.append(f"{ind}  <<{pid}>> " + chunk.replace("\n", "\n" + ind + "    "))

    def _string(self, s: str, path: str, ind: str, label: str):
        chunks = [c.strip() for c in re.split(r"\n\s*\n", s) if c.strip()]
        if len(chunks) == 1 and not REF_RE.search(chunks[0]) and len(chunks[0]) < 200:
            self.lines.append(f"{ind}{label}: {chunks[0]}")
            return
        self.lines.append(f"{ind}{label}:")
        for c in chunks:
            sub = [c]
            if len(REF_RE.findall(c)) > 3 and "\n" in c:
                sub = [x.strip() for x in c.split("\n") if x.strip()]
            for x in sub:
                self._add_para(x, path, ind)

    def _walk(self, obj, path, depth):
        ind = "  " * depth
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                if isinstance(v, (dict, list)):
                    self.lines.append(f"{ind}{k}:")
                    self._walk(v, p, depth + 1)
                elif isinstance(v, str):
                    self._string(v, p, ind, str(k))
                else:
                    self.lines.append(f"{ind}{k}: {json.dumps(v)}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                p = f"{path}[{i}]"
                if isinstance(v, (dict, list)):
                    self.lines.append(f"{ind}- [{i}]")
                    self._walk(v, p, depth + 1)
                elif isinstance(v, str):
                    self._string(v, p, ind, f"- [{i}]")
                else:
                    self.lines.append(f"{ind}- {json.dumps(v)}")
        elif isinstance(obj, str):
            self._string(obj, path, ind, "text")
        else:
            self.lines.append(f"{ind}{json.dumps(obj)}")


# ---------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You are an accounts-payable control agent. For every invoice package you choose exactly ONE action and cite the decisive evidence.

ACTIONS (use these exact strings):
- settle_invoice: the invoice is valid, reconciled (invoice, purchase order and receipt agree), not already paid, no verification is outstanding, and the amount is WITHIN the agent's autonomous/delegated settlement authority.
- request_approval: commercially valid and reconciled, but OUTSIDE the agent's delegated authority (e.g. the amount exceeds the autonomous limit in the current policy, or policy requires a named human approver). Nothing is wrong with the invoice itself.
- hold_invoice: payment must PAUSE until a specific stated verification completes (e.g. bank-detail change call-back, tax/KYC document, inspection or certificate pending). The verification is still outstanding.
- reject_duplicate: this SAME commercial invoice (same vendor and same invoice number for the same charge) has ALREADY BEEN PAID. A payment for a DIFFERENT invoice (different invoice number, different period/charge) is NOT a duplicate, even if vendor or amount match.
- open_exception: material records CONFLICT (e.g. invoice vs PO vs receipt disagree on quantity, price, amount, vendor, bank account or tax) and cannot be reconciled without an exception workflow.

HOW TO READ THE DOCUMENTS:
- Documents mix current operational facts with archived examples, training material, templates, old cases, other invoices and decoys. IGNORE any paragraph that is an archive/historical example, training or illustrative text, a template, or about a different invoice. They often contain tempting action words.
- Watch negation and resolution: "no duplicate exists", "not a duplicate", "verification completed", "hold released", "does not exceed", "no discrepancy" mean the corresponding action does NOT apply.
- Action words used incidentally (in names, headings, examples, glossary text) are irrelevant.
- The cover sheet only introduces the package; its reference is never decisive.
- Exactly one current paragraph about THIS invoice determines the action. Paragraphs that carry bracketed references are labelled <<P#>>. Pick the decisive one and copy its bracketed references exactly (normally exactly three).
- Apply the current policy revision and its authority limits. Compare amounts carefully in the same currency.
- If several issues truly apply, priority is: reject_duplicate > open_exception > hold_invoice > request_approval > settle_invoice.
- Some packages instead give each fact its own paragraph starting with one reference like [R_ABC]. Then the decisive evidence is the THREE current-case records (for example the payment ledger entry, the signed instrument / order record, the vendor-master / callback / delegation record, the payment-controls record) and NOT the decoys (archive examples, training cases, previous-quarter thresholds, historical transfers for other orders, closed incidents for other suppliers, archived memos for unrelated orders, cover sheets).
  * A ledger entry that previously settled the SAME order/invoice now submitted => reject_duplicate. If it settled a DIFFERENT order, it is not a duplicate.
  * Order/receipt/invoice totals or units that disagree materially => open_exception.
  * Remittance account differs from the vendor master and the callback is NOT confirmed => hold_invoice. If the callback is confirmed, there is no hold.
  * Payable total greater than the autonomous ceiling => request_approval; otherwise, with everything reconciled, settle_invoice.
- Never invent identifiers. If no invoice number is written anywhere in the package, return an empty string for invoiceNumber.

FACTS: take vendorName, invoiceNumber, amount and currency from the CURRENT invoice in this package (not from examples). Copy names and numbers exactly as written. "amount" is the invoice total in MAJOR units as written (e.g. "12,345.50"); "amountMinor" is that total as an integer in minor units (paise/cents: x100; JPY x1). If the documents state the amount only in minor units, give it as amountMinor and set amount to null.

Return ONLY JSON of this shape:
{"decisions":[{"packageId":"...","decisiveParagraph":"P#","evidenceRefs":["ref1","ref2","ref3"],"action":"one of the five","vendorName":"...","invoiceNumber":"...","amount":"...","amountMinor":0,"currency":"ISO code","reason":"one or two sentences explaining why this action, referring to the decisive facts"}]}
Return one decision per package, in the same order, with the packageId given in the PACKAGE header."""


def build_user_prompt(batch: dict, items: list) -> str:
    ctx = batch_context(batch)
    parts = ["BATCH CONTEXT:\n" + json.dumps(ctx, ensure_ascii=False, indent=1)]
    for pid, seg in items:
        parts.append(f"\n===== PACKAGE {pid} =====\n{seg.text}\n===== END PACKAGE {pid} =====")
    parts.append(f"\nReturn exactly {len(items)} decisions as JSON.")
    return "\n".join(parts)


# ---------------------------------------------------------------- LLM
_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT, connect=10))
    return _client


def _extract_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


_sem: asyncio.Semaphore | None = None
LAST_LLM_ERROR = {"error": None}


def sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(int(os.environ.get("LLM_CONCURRENCY", "4")))
    return _sem


async def call_llm(system: str, user: str, deadline: float | None = None) -> dict:
    """Call the model; retry rate limits, 5xx and timeouts while time remains."""
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not configured")
    loop = asyncio.get_running_loop()
    deadline = deadline or (loop.time() + DECIDE_BUDGET)
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    base = {"model": LLM_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        extra = json.loads(os.environ.get("LLM_EXTRA", "") or "{}")
    except ValueError:
        extra = {}
    variants = [dict(base, temperature=0, response_format={"type": "json_object"}, **extra),
                dict(base, response_format={"type": "json_object"}, **extra),
                dict(base, **extra),
                dict(base)]
    models = [LLM_MODEL] + [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",")
                            if m.strip() and m.strip() != LLM_MODEL]
    mi, vi, attempt, last = 0, 0, 0, None
    while vi < len(variants):
        remaining = deadline - loop.time()
        if remaining < 4:
            break
        attempt += 1
        try:
            async with sem():
                remaining = deadline - loop.time()
                if remaining < 4:
                    break
                r = await client().post(url, headers=headers, json=dict(variants[vi], model=models[mi % len(models)]),
                                        timeout=min(LLM_TIMEOUT, remaining - 1))
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = RuntimeError(f"LLM transport error ({models[mi % len(models)]}): {e!r}")
            log.warning("%s", last)
            LAST_LLM_ERROR["error"] = str(last)
            mi += 1  # slow or unreachable model: try the next one
            await asyncio.sleep(min(1.0 * attempt, max(0, deadline - loop.time() - 5)))
            continue
        if r.status_code == 400:
            last = RuntimeError(f"LLM 400: {r.text[:300]}")
            log.warning("%s", last)
            vi += 1
            continue
        if r.status_code == 404 and mi + 1 < len(models):
            last = RuntimeError(f"LLM 404 for {models[mi]}: {r.text[:200]}")
            log.warning("%s", last)
            mi += 1
            continue
        if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
            last = RuntimeError(f"LLM {r.status_code} ({models[mi % len(models)]}): {r.text[:300]}")
            mi += 1  # rotate to the next model, if any
            log.warning("%s", last)
            LAST_LLM_ERROR["error"] = str(last)
            try:
                ra = float(r.headers.get("retry-after", "0"))
            except ValueError:
                ra = 0
            await asyncio.sleep(min(max(ra, 1.5 * attempt), 6, max(0, deadline - loop.time() - 5)))
            continue
        if r.status_code >= 300:
            last = RuntimeError(f"LLM {r.status_code}: {r.text[:300]}")
            log.warning("%s", last)
            LAST_LLM_ERROR["error"] = str(last)
            raise last
        try:
            content = r.json()["choices"][0]["message"]["content"]
            return _extract_json(content)
        except Exception as e:  # noqa: BLE001
            last = RuntimeError(f"LLM bad output: {e!r}")
            log.warning("%s", last)
            await asyncio.sleep(0.5)
            continue
    LAST_LLM_ERROR["error"] = str(last)
    raise last or RuntimeError("LLM failed: out of time")


# ---------------------------------------------------------------- validation
def _fmt_ref(r: str) -> str:
    return f"[{r}]" if REF_STYLE == "bracketed" else r


def _clean_ref(r) -> str:
    r = str(r).strip()
    if r.startswith("[") and r.endswith("]"):
        r = r[1:-1].strip()
    return r


def amount_to_minor(amount, currency: str):
    if amount is None:
        return None
    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
        s = str(amount)
    else:
        s = str(amount)
    s = re.sub(r"[^0-9.,\-]", "", s)
    if not s or not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        tail = s.split(",")[-1]
        s = s.replace(",", "." if len(tail) in (1, 2) and s.count(",") == 1 else "")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    exp = CURRENCY_EXP.get((currency or "").upper(), 2)
    return int((d * (Decimal(10) ** exp)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


STRUCT_KEYS = {
    "vendorName": ("vendorname", "vendor", "suppliername", "supplier", "payee"),
    "invoiceNumber": ("invoicenumber", "invoiceno", "invoice_number", "invoiceid", "invoiceref"),
    "amountMinor": ("amountminor", "totalminor", "amount_minor", "grossminor"),
    "currency": ("currency", "currencycode"),
}


def structured_facts(pkg: dict) -> dict:
    """Pick explicit fact fields at shallow depth outside archive/example sections."""
    found = {}

    def walk(obj, path, depth):
        if depth > 3 or DECOY_RE.search(path):
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                nk = re.sub(r"[^a-z_]", "", str(k).lower()).replace("_", "")
                for fact, names in STRUCT_KEYS.items():
                    if fact in found:
                        continue
                    if nk in [n.replace("_", "") for n in names]:
                        if fact == "amountMinor" and isinstance(v, int) and not isinstance(v, bool):
                            found[fact] = (depth, v)
                        elif fact != "amountMinor" and isinstance(v, str) and 0 < len(v) < 120:
                            found[fact] = (depth, v)
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(v, f"{path}.{k}", depth + 1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", depth + 1)

    walk(pkg, "", 0)
    return {k: v[1] for k, v in found.items()}


def choose_refs(seg: Segmented, para_id, llm_refs) -> list:
    valid = set(seg.all_refs)
    refs = []
    for r in llm_refs or []:
        r = _clean_ref(r)
        if r in valid and r not in refs:
            refs.append(r)
    by_id = {p["pid"]: p for p in seg.paras}
    para = by_id.get(str(para_id).strip("<> ")) if para_id else None
    # 1) LLM refs that all sit together in one paragraph.
    if len(refs) == 3:
        for p in seg.paras:
            if all(r in p["refs"] for r in refs):
                return refs
    # 2) The named paragraph's refs.
    if para:
        if len(para["refs"]) == 3:
            return list(para["refs"])
        inside = [r for r in refs if r in para["refs"]]
        if len(para["refs"]) > 3:
            for r in para["refs"]:
                if len(inside) >= 3:
                    break
                if r not in inside:
                    inside.append(r)
            return inside[:3]
        return list(para["refs"])
    # 3) A paragraph containing most LLM refs.
    if refs:
        best = max(seg.paras, key=lambda p: sum(r in p["refs"] for r in refs))
        if len(best["refs"]) == 3 and sum(r in best["refs"] for r in refs) >= 2:
            return list(best["refs"])
        return refs[:3]
    return []


def evidence_snippets(seg, refs: list) -> dict:
    """For each ref, the sentence (or paragraph) of the documents that carries it."""
    out = {}
    for r in refs:
        for p in seg.paras:
            if r in p["refs"]:
                text = p["text"]
                tag = f"[{r}]"
                idx = text.find(tag)
                start = max(text.rfind(". ", 0, idx) + 2 if text.rfind(". ", 0, idx) >= 0 else 0, 0)
                if text.startswith(tag) or idx < 3:
                    snippet = text[idx + len(tag):].strip()
                    snippet = re.split(r"(?<=\.)\s", snippet, maxsplit=3)
                    snippet = " ".join(snippet[:3])
                else:
                    snippet = text[start:idx].strip()
                snippet = re.sub(r"\s*\[[^\]]+\]\s*", " ", snippet).strip().rstrip(".")
                out[r] = snippet[:320]
                break
    return out


ACTION_MEANING = {
    "settle_invoice": "the invoice is valid, reconciled, not previously paid and within autonomous authority, so it can be settled",
    "request_approval": "the invoice is commercially valid but its total is outside delegated authority, so a named approver must release it",
    "hold_invoice": "payment must pause until the stated verification is completed",
    "reject_duplicate": "the same commercial invoice has already been paid, so a second payment must be rejected",
    "open_exception": "the controlling records materially conflict, so the case must go to the exception workflow",
}


def build_rationale(action: str, reason: str, refs: list, support: dict | None = None) -> str:
    support = support or {}
    parts = [f"Action {action}: {ACTION_MEANING.get(action, '')}."]
    for r in refs:
        snip = support.get(r)
        parts.append(f"[{r}] {snip}." if snip else f"[{r}] supports this.")
    reason = re.sub(r"\s+", " ", (reason or "").strip())
    if reason:
        parts.append(reason if reason.endswith(".") else reason + ".")
    text = " ".join(parts)
    if len(text) > 1500:
        text = " ".join(parts[:1 + len(refs)])[:1500]
    return text


def _old_build_rationale(action: str, reason: str, refs: list) -> str:
    reason = re.sub(r"\s+", " ", (reason or "").strip())
    cited = ", ".join(f"[{r}]" for r in refs) if refs else "the package documents"
    text = (f"Action {action}: {reason} " if reason else f"Action {action} is required by the decisive record. ")
    text += f"Decisive evidence: {cited}."
    if len(text) < 60:
        text += " The decision follows the current policy revision and the current invoice record."
    if len(text) > 1500:
        keep = 1500 - len(f" Decisive evidence: {cited}.") - 20
        text = f"Action {action}: {reason[:max(0, keep)]}… Decisive evidence: {cited}."
    return text[:1500]


def finalize(seg: Segmented, pkg: dict, raw: dict, source: str) -> dict:
    action = str(raw.get("action", "")).strip()
    if action not in ACTIONS:
        raise ValueError(f"bad action {action!r}")
    det_refs, det_paras = decisive_refs(seg)
    refs = det_refs or choose_refs(seg, raw.get("decisiveParagraph"), raw.get("evidenceRefs"))
    reason_text = str(raw.get("reason") or raw.get("rationale") or "")
    if det_paras and source == "llm":
        rule_action = classify_text("\n".join(p["text"] for p in det_paras))
        if rule_action != action and rule_action != "settle_invoice":
            log.warning("rules override model: %s -> %s", action, rule_action)
            action = rule_action
            reason_text = ""
            source = "llm+rules"
    currency = str(raw.get("currency") or "").strip().upper()[:3]
    sf = structured_facts(pkg)
    vendor = str(raw.get("vendorName") or sf.get("vendorName") or "").strip()
    inv = str(raw.get("invoiceNumber") or sf.get("invoiceNumber") or "").strip()
    if sf.get("currency") and len(str(sf["currency"]).strip()) == 3:
        currency = currency or str(sf["currency"]).strip().upper()
    minor = None
    if isinstance(sf.get("amountMinor"), int):
        minor = sf["amountMinor"]
    if minor is None:
        minor = amount_to_minor(raw.get("amount"), currency)
    if minor is None:
        am = raw.get("amountMinor")
        if isinstance(am, (int, float)) and not isinstance(am, bool):
            minor = int(round(am))
        elif isinstance(am, str) and re.fullmatch(r"-?\d+", am.strip()):
            minor = int(am.strip())
    if minor is None:
        minor = 0
    df = doc_facts(seg)
    if df.get("vendorName"):
        vendor = df["vendorName"]
    if df.get("invoiceNumber"):
        inv = df["invoiceNumber"]
    if df.get("currency"):
        currency = df["currency"]
    if df.get("amount"):
        m2 = amount_to_minor(df["amount"], currency)
        if m2 is not None:
            minor = m2
    if inv and not _present(inv, seg.text):
        inv = ""  # never return an identifier that is not in the documents
    return {
        "action": action,
        "facts": {"vendorName": vendor, "invoiceNumber": inv, "amountMinor": minor,
                  "currency": currency or "INR"},
        "refs": refs,
        "reason": reason_text,
        "support": evidence_snippets(seg, refs),
        "source": source,
    }



# ---------------------------------------------------------------- deterministic evidence & facts
REF_DECOY_RE = re.compile(
    r"archive|archived|training|previous quarter|expired|historical|closed reconciliation|"
    r"separate supplier|unrelated|hypothetical|cover sheet|not this vendor|govern no live case|"
    r"vocabulary|non-operative|superseded|illustrat|example", re.I)
COVER_FACTS_RE = re.compile(
    r"Supplier (?P<v>.+?); invoice (?P<i>[^\s;]+); stated total (?P<c>[A-Z]{3}) (?P<a>[\d][\d,]*(?:\.\d+)?)")
INSTRUMENT_RE = re.compile(
    r"(?:Signed )?commercial instrument (?P<i>[A-Z][A-Z0-9/_.-]*\d[A-Z0-9/_.-]*) covers order (?P<o>[\w-]+),"
    r"(?: payee (?P<p>[\w-]+))? and (?P<c>[A-Z]{3}) (?P<a>\d[\d,]*(?:\.\d+)?)", re.I)
LEDGER_RE = re.compile(
    r"disbursement for (?P<c>[A-Z]{3}) (?P<a>\d[\d,]*(?:\.\d+)?) to (?P<v>.+?) previously settled order "
    r"(?P<o1>[\w-]+)\. The currently submitted order is (?P<o2>[\w-]+)", re.I)
ORDER_TOTAL_RE = re.compile(r"submitted invoice total is (?P<c>[A-Z]{3}) (?P<a>\d[\d,]*(?:\.\d+)?)", re.I)


def _is_decoy_para(p) -> bool:
    return bool(REF_DECOY_RE.search(p["text"]))


def decisive_refs(seg: "Segmented"):
    """Return (refs, paragraphs) when the evidence can be identified without the model."""
    multi = [p for p in seg.paras if len(p["refs"]) >= 3 and not _is_decoy_para(p)]
    if len(multi) == 1 and len(multi[0]["refs"]) == 3:
        return list(multi[0]["refs"]), multi
    if not multi:
        single = [p for p in seg.paras if len(p["refs"]) == 1 and not _is_decoy_para(p)]
        if len(single) == 3:
            return [p["refs"][0] for p in single], single
    return None, None


def doc_facts(seg: "Segmented") -> dict:
    text = seg.text
    out = {}
    m = COVER_FACTS_RE.search(text)
    if m:
        out = {"vendorName": m.group("v").strip(), "invoiceNumber": m.group("i").strip().rstrip("."),
               "currency": m.group("c"), "amount": m.group("a")}
        return out
    m = INSTRUMENT_RE.search(text)
    if m:
        out.update({"invoiceNumber": m.group("i").rstrip("."), "currency": m.group("c").upper(),
                    "amount": m.group("a")})
    led = LEDGER_RE.search(text)
    if led:
        out.setdefault("vendorName", led.group("v").strip())
    if "amount" not in out:
        m = ORDER_TOTAL_RE.search(text)
        if m:
            out.update({"currency": m.group("c").upper(), "amount": m.group("a")})
    return out


def classify_text(text: str):
    """Rule classifier for a decisive passage (used when the model is unavailable)."""
    t = re.sub(r"\[[^\]]+\]", "", text)
    # Generic policy wording lists every control; it never decides a case by itself.
    t = re.sub(r"Payment controls:[^\n]*?Otherwise follow the delegation limit\.?", " ", t, flags=re.I)
    t = re.sub(r"Payment controls:[^\n]*", " ", t, flags=re.I)
    led = LEDGER_RE.search(t)
    if led and led.group("o1").lower() == led.group("o2").lower():
        return "reject_duplicate"
    if re.search(r"earlier posting for the same|same commercial key|earlier settled entry|second disbursement|"
                 r"exact commercial duplicate|already (been )?(paid|discharged|settled)|another scan of the same", t, re.I) \
            and not re.search(r"\bno (earlier|prior) (posting|settlement)|no paid item", t, re.I):
        return "reject_duplicate"
    if re.search(r"outside the permitted reconciliation tolerance|conflicts with|does not reconcile|incompatible|"
                 r"contradict|exception (queue|case|workflow)|documented exception", t, re.I):
        return "open_exception"
    m = re.search(r"Order [\w-]+ is signed for [A-Z]{3} (\d[\d,]*(?:\.\d+)?).*?submitted invoice total is [A-Z]{3} (\d[\d,]*(?:\.\d+)?)", t, re.I | re.S)
    if m and abs(float(m.group(1).replace(",", "")) - float(m.group(2).replace(",", ""))) > 0.5:
        return "open_exception"
    if re.search(r"(callback|call-back|known-channel|out-of-band|pre-existing contact)[^.]*"
                 r"(not completed|has not|neither confirmed|incomplete|pending|until)", t, re.I) or \
            re.search(r"(beneficiary details|destination-account change|newly supplied bank account|"
                      r"pauses release|freezes payment)", t, re.I):
        if not re.search(r"callback[^.]*\bconfirmed\b(?! nor)", t, re.I):
            return "hold_invoice"
    vm = re.search(r"Vendor master lists account ([\w-]+)\. Remittance instructions now specify ([\w-]+)", t, re.I)
    if vm and vm.group(1) != vm.group(2) and not re.search(r"callback[^.]*\bconfirmed\b(?! nor)", t, re.I):
        return "hold_invoice"
    if re.search(r"outside the operator's|delegation ceiling|exceeds this queue|without escalation only up to|"
                 r"financial-approval|named financial approver", t, re.I):
        return "request_approval"
    ceil = re.search(r"autonomous payment ceiling for [A-Z]{3} is [A-Z]{3} (\d[\d,]*(?:\.\d+)?)", t, re.I)
    tot = ORDER_TOTAL_RE.search(t) or INSTRUMENT_RE.search(t)
    if ceil and tot:
        if float(tot.group("a").replace(",", "")) > float(ceil.group(1).replace(",", "")):
            return "request_approval"
    return "settle_invoice"


def _present(value: str, text: str) -> bool:
    return bool(value) and value.lower() in text.lower()

# ---------------------------------------------------------------- heuristic fallback
PATTERNS = [
    ("reject_duplicate", r"already (been )?paid|duplicate|previously paid|paid in full|remitted on|was paid"),
    ("open_exception", r"conflict|mismatch|discrepan|does not (match|agree|reconcile)|disagree|inconsisten|variance"),
    ("hold_invoice", r"\bhold\b|pending|until .{0,60}verif|awaiting|call-?back|re-?verif"),
    ("request_approval", r"exceed|above .{0,40}(limit|authority)|outside .{0,40}authority|beyond .{0,40}(limit|authority)|requires? .{0,30}approv"),
    ("settle_invoice", r"within .{0,40}(limit|authority)|reconciled|three-way match|matched|cleared for payment"),
]
NEG_RE = r"\b(no|not|without|never|nor|cleared|resolved|released|completed)\b[^.;]{0,25}$"


def text_facts(text: str) -> dict:
    """Best-effort facts from labelled lines, used only by the fallback path."""
    out = {}
    m = re.search(r"(?im)^\s*(?:-\s*)?(?:vendor|supplier|payee)(?:\s*name)?\s*[:=]\s*(.+?)\s*$", text)
    if m:
        out["vendorName"] = m.group(1).strip().strip('"')
    m = re.search(r"(?im)invoice\s*(?:number|no\.?|#|id)\s*[:=]?\s*([A-Z0-9][A-Z0-9/_.-]{2,})", text)
    if m:
        out["invoiceNumber"] = m.group(1).strip().rstrip(".")
    m = re.search(r"(?im)(?:total|amount(?:\s*due)?|invoice\s*total)\s*[:=]\s*([A-Z]{3})\s*([0-9][0-9,]*(?:\.[0-9]+)?)", text)
    if m:
        out["currency"], out["amount"] = m.group(1), m.group(2)
    return out


def heuristic(seg: Segmented, pkg: dict) -> dict:
    refs, paras = decisive_refs(seg)
    if paras:
        action = classify_text("\n".join(p["text"] for p in paras))
        raw = {"action": action, "evidenceRefs": refs,
               "reason": "Read together, these current-case records determine the action under the payment controls."}
        return finalize(seg, pkg, raw, "rules")
    return legacy_heuristic(seg, pkg)


def legacy_heuristic(seg: Segmented, pkg: dict) -> dict:
    best = None
    for p in seg.paras:
        ctx = p["path"] + " " + p["text"]
        if DECOY_RE.search(ctx) or COVER_RE.search(p["path"]):
            continue
        for rank, (action, pat) in enumerate(PATTERNS):
            for m in re.finditer(pat, p["text"], re.I):
                if re.search(NEG_RE, p["text"][:m.start()], re.I):
                    continue
                score = (rank, 0 if len(p["refs"]) == 3 else 1)
                if best is None or score < best[0]:
                    best = (score, action, p)
                break
    if best is None:
        cand = [p for p in seg.paras if not COVER_RE.search(p["path"])] or seg.paras
        action, para = "request_approval", (cand[0] if cand else None)
    else:
        action, para = best[1], best[2]
    sf = dict(text_facts(seg.text))
    sf.update(structured_facts(pkg))
    raw = {
        "action": action,
        "decisiveParagraph": para["pid"] if para else None,
        "evidenceRefs": para["refs"][:3] if para else [],
        "vendorName": sf.get("vendorName", ""),
        "invoiceNumber": sf.get("invoiceNumber", ""),
        "amountMinor": sf.get("amountMinor"),
        "amount": sf.get("amount"),
        "currency": sf.get("currency", ""),
        "reason": "Keyword review of the current, non-archival record determined this action.",
    }
    return finalize(seg, pkg, raw, "heuristic")


# ---------------------------------------------------------------- orchestration
_inflight: dict[str, asyncio.Future] = {}


def _match_decisions(out, chunk):
    decs = None
    if isinstance(out, dict):
        decs = out.get("decisions")
        if decs is None:
            for v in out.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    decs = v
                    break
            if decs is None and "action" in out:
                decs = [out]
    elif isinstance(out, list):
        decs = out
    if not isinstance(decs, list):
        raise ValueError("no decisions list")
    decs = [d for d in decs if isinstance(d, dict)]
    found = {}
    for d in decs:
        raw = str(d.get("packageId", "")).strip()
        for key, pid, pkg, seg in chunk:
            if raw == pid or (raw and (raw.endswith(pid) or pid in raw)):
                found.setdefault(key, d)
                break
    if not found and len(decs) == len(chunk):
        for (key, *_), d in zip(chunk, decs):
            found[key] = d
    return found


async def _decide_chunk(batch, chunk, store, futures, deadline=None):
    """chunk: list of (key, pid, pkg, seg). Re-asks the model for packages it skipped."""
    results = {}
    loop = asyncio.get_running_loop()
    deadline = deadline or (loop.time() + DECIDE_BUDGET)
    pending = list(chunk)
    for attempt in range(4):
        if not pending or deadline - loop.time() < 8:
            break
        try:
            user = build_user_prompt(batch, [(pid, seg) for _, pid, _, seg in pending])
            out = await call_llm(SYSTEM_PROMPT, user, deadline)
            found = _match_decisions(out, pending)
            for key, pid, pkg, seg in pending:
                d = found.get(key)
                if d is None:
                    continue
                try:
                    dec = finalize(seg, pkg, d, "llm")
                    store.put_decision(key, dec)
                    results[key] = dec
                except Exception as e:  # noqa: BLE001
                    log.warning("invalid model decision for %s: %s", pid, e)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM chunk failed (attempt %d): %s", attempt + 1, e)
            await asyncio.sleep(min(1.0, max(0, deadline - loop.time() - 8)))
        pending = [c for c in pending if c[0] not in results]
        if pending:
            log.warning("model left %d package(s) undecided; retrying", len(pending))
    for key, pid, pkg, seg in chunk:
        if key not in results:
            results[key] = heuristic(seg, pkg)  # not cached: a later run may retry the model
        fut = futures[key]
        if not fut.done():
            fut.set_result(results[key])


async def decide_batch(batch: dict, store) -> dict:
    """Return {packageId: decision}."""
    packages = batch["packages"]
    segs, keys, waits, todo = {}, {}, {}, []
    futures = {}
    loop = asyncio.get_running_loop()
    for pkg in packages:
        pid = package_id(pkg)
        seg = Segmented(pkg)
        key = decision_key(batch, pkg)
        segs[pid], keys[pid] = seg, key
        cached = store.get_decision(key)
        if cached:
            fut = loop.create_future()
            fut.set_result(cached)
            waits[pid] = fut
        elif key in _inflight:
            waits[pid] = _inflight[key]
        else:
            fut = loop.create_future()
            _inflight[key] = fut
            futures[key] = fut
            waits[pid] = fut
            todo.append((key, pid, pkg, seg))
    deadline = loop.time() + DECIDE_BUDGET - 1
    jobs = [asyncio.create_task(_decide_chunk(batch, todo[i:i + LLM_CHUNK], store, futures, deadline))
            for i in range(0, len(todo), LLM_CHUNK)]
    try:
        if jobs:
            done, pending = await asyncio.wait(jobs, timeout=DECIDE_BUDGET)
            for j in pending:
                j.cancel()
        for key, pid, pkg, seg in todo:
            fut = futures[key]
            if not fut.done():
                fut.set_result(heuristic(seg, pkg))
        out = {}
        for pkg in packages:
            pid = package_id(pkg)
            out[pid] = await waits[pid]
        return out
    finally:
        for key, *_ in todo:
            _inflight.pop(key, None)


def to_proposal(pid: str, action_id: str, dec: dict) -> dict:
    refs = [_fmt_ref(r) for r in dec["refs"]]
    return {
        "packageId": pid,
        "actionId": action_id,
        "action": dec["action"],
        "facts": dict(dec["facts"]),
        "evidenceRefs": refs,
        "rationale": build_rationale(dec["action"], dec.get("reason", ""), dec["refs"], dec.get("support")),
    }
