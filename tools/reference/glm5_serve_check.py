"""What spec 045 asks of GLM served on two ranks, asked of a running server: a streamed reply, a tool call in
GLM's form, a stop string, and a client that hangs up in the middle of a reply, after which the next request is
answered. Thinking is off, so a reply is its content. The standard library only; `glm5_client_bench.sh`'s AFTER runs it on the load it timed.

    python3 tools/reference/glm5_serve_check.py http://127.0.0.1:18888 glm-5.3-flash

It prints what it received and ends with OK, or names what was not as it should be and exits 1.
"""

import http.client, json, sys, time
from urllib.parse import urlparse

URL, MODEL = sys.argv[1], sys.argv[2]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                         "required": ["city"]}}}]


def stream(body, hang_up_after=None):
    """The streamed reply's content, tool calls, finish reason and seconds; `hang_up_after` deltas, the socket is closed."""

    u = urlparse(URL)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    started = time.time()
    conn.request("POST", "/v1/chat/completions", json.dumps({"model": MODEL, "stream": True, "temperature": 0,
                                                       "chat_template_kwargs": {"enable_thinking": False}, **body}),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        sys.exit(f"status {r.status}: {r.read()[:300]!r}")
    content, calls, finish, deltas = "", {}, None, 0
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        for choice in json.loads(line[6:]).get("choices") or []:
            d = choice.get("delta") or {}
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                e = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                e["name"] += (tc.get("function") or {}).get("name") or ""
                e["arguments"] += (tc.get("function") or {}).get("arguments") or ""
            finish = choice.get("finish_reason") or finish
            deltas += 1
        if hang_up_after is not None and deltas >= hang_up_after:
            r.close()
            conn.close()
            return content, calls, "hung up", time.time() - started
    conn.close()
    return content, calls, finish, time.time() - started


failed = []


def ask(text):
    return [{"role": "user", "content": text}]


def need(what, ok):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    failed.append(what) if not ok else None


content, _, finish, s = stream({"messages": ask("Name three primary colours, one a line."), "max_tokens": 64})
print(repr(content[:200]))
need(f"a streamed reply: {len(content)} characters, finish {finish}, {s:.2f} s", bool(content.strip()) and finish in ("stop", "length"))

_, calls, finish, s = stream({"messages": ask("What is the weather in Paris for the next 3 days? Use the tool."),
                              "tools": TOOLS, "max_tokens": 256})
print(calls)
call = calls.get(0) or {"name": "", "arguments": "null"}
try:
    arguments = json.loads(call["arguments"] or "null")
except ValueError:
    arguments = None
need(f"a tool call: {call['name']} {arguments}, finish {finish}, {s:.2f} s",
     call["name"] == "get_weather" and isinstance(arguments, dict) and "city" in arguments and finish == "tool_calls")

content, _, finish, s = stream({"messages": ask("Count from 1 to 20, the numbers separated by commas."), "max_tokens": 128,
                                "stop": [" 7"]})
print(repr(content))
need(f"a stop string: finish {finish}, {s:.2f} s", finish == "stop" and "7" not in content and "6" in content)

content, _, finish, s = stream({"messages": ask("Write four hundred words about the sea."), "max_tokens": 1024}, hang_up_after=8)
need(f"hung up after {len(content)} characters, {s:.2f} s", finish == "hung up")
time.sleep(1.0)
content, _, finish, s = stream({"messages": ask("Say hello in one short sentence."), "max_tokens": 32})
print(repr(content))
need(f"the next request after a hang-up: finish {finish}, {s:.2f} s", bool(content.strip()) and s < 20)
sys.exit(f"{len(failed)} failed") if failed else print("OK")
