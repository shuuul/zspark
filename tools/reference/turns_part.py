"""Where a later request parts from what `zspark serve` holds (spec 030): conversations through the three APIs,
each request's prompt against the sequence the session held before it.

Not a fixture and never runtime: a measurement, run by hand against a server on this machine whose log it reads.
The standard library only.

    zspark serve CHECKPOINT --port 8011 > server.log 2>&1 &
    python3 tools/reference/turns_part.py --url http://127.0.0.1:8011 --log server.log --out turns.json

A conversation is a first message, its reply, a second and a third turn, the third turn's message edited, the
second turn's message edited (the turns after it gone, as a client that edits sends it) and the first message
edited; with tools, a call, its result, the answer, a new turn and that turn edited. Each is sent with thinking
off, with thinking on and the reasoning sent back as the API carries it, and with the reasoning left out, through
chat completions, Responses (the items sent again, and `previous_response_id`) and Messages. A required and a
named call after thinking go through all of them too (spec 034), and replies held to a schema after thinking
through chat completions. Every conversation
opens with a salt, so it shares nothing with the one before it but the template's opening.

`--set shapes` (spec 035) is every way a reply was found to differ from its rendering, each a reply and the
turn after it through the APIs that can ask for it: a schema's reply, a JSON object, a document beside tools,
several calls, a call after content, a call whose values are JSON, a stop string, the length reached in content,
in reasoning and inside a call, a client that hangs up (a streamed reply closed after some deltas: the turn sent
again, and gone on from with what arrived), whitespace a client adds around the content it sends back, thinking switched either way, a
system prompt and the tools changed, and a turn sent again as it was.

`--only` takes names between commas (any of them, of an API or a conversation) and `--api` one API, so a few
rows can be taken again without the half hour the whole set holds the GPU for.

For each request the server's log gives the prompt's tokens and the position of the state it went on from.
`run again` is the prompt less that position.
Each row also holds the states on the reply's stride after the request (`strides`, spec 036) and the seconds the
server took to the reply's first token (`prompt_s`), which are a timing only when the machine was held alone.
"""

import argparse, hashlib, http.client, json, re, sys, time
from urllib.parse import urlparse

WEATHER = {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                                          "units": {"type": "string", "enum": ["c", "f"]}},
                          "required": ["city"]}}
# A property without one `type`: a required call to it is written as the JSON payload (spec 023).
LOOKUP = {"name": "lookup", "description": "Look a key up in the store",
          "parameters": {"type": "object", "properties": {"key": {"type": ["string", "integer"]}}, "required": ["key"]}}

# A reply held to a schema (`response_format`), for the conversation that asks whether it renders as decoded.
ANSWER = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}

DONE = re.compile(r"prompt (\d+) tokens, (\d+) held, in ([\d.]+) s; reply (\d+) tokens.*?snapshots, \d+ MiB, at \{([^}]*)\}(?:; on a reply's stride \{([^}]*)\})?")


def stream(url, path, body, headers, cut):
    """A streamed reply closed by the client after `cut[1]` deltas of `cut[0]` ("content" or "reasoning"): what
    had arrived, as a reply. The server learns of it when it next writes."""

    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    conn.request("POST", path, json.dumps({**body, "stream": True}).encode(), {"Content-Type": "application/json", **(headers or {})})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"{path}: {response.status} {response.read().decode()[:400]}")
    got, seen = {"content": "", "reasoning": ""}, 0
    while seen < cut[1]:
        line = response.readline()
        if not line:
            break
        if not line.startswith(b"data: {"):
            continue
        event = json.loads(line[6:])
        # the three APIs' deltas: chat completions, Responses, Messages
        if "choices" in event:
            delta = event["choices"][0]["delta"] if event["choices"] else {}
            pieces = [("content", delta.get("content")), ("reasoning", delta.get("reasoning_content"))]
        elif event.get("type") in ("response.output_text.delta", "response.reasoning_text.delta"):
            pieces = [("content" if "output_text" in event["type"] else "reasoning", event["delta"])]
        elif event.get("type") == "content_block_delta":
            pieces = [("content", event["delta"].get("text")), ("reasoning", event["delta"].get("thinking"))]
        else:
            pieces = []
        for kind, text in pieces:
            if text:
                got[kind] += text
                seen += kind == cut[0]
    response.close()
    conn.close()
    return {"role": "assistant", "content": got["content"], "reasoning": got["reasoning"], "calls": [], "finish": "hung up"}


def post(url, path, body, headers=None):
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    conn.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json", **(headers or {})})
    response = conn.getresponse()
    text = response.read().decode()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"{path}: {response.status} {text[:400]}")
    return json.loads(text)


class Chat:
    """Chat completions. `kept`: an assistant message goes back with its `reasoning_content`."""

    name = "chat"

    def __init__(self, url, model):
        self.url, self.model = url, model

    def messages(self, turns, kept):
        out = []
        for turn in turns:
            if turn["role"] == "assistant":
                message = {"role": "assistant", "content": turn["content"]}
                if kept and turn["reasoning"]:
                    message["reasoning_content"] = turn["reasoning"]
                if turn["calls"]:
                    message["tool_calls"] = [{"id": c["id"], "type": "function",
                                              "function": {"name": c["name"], "arguments": c["arguments"]}} for c in turn["calls"]]
                out.append(message)
            elif turn["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": turn["id"], "content": turn["content"]})
            else:
                out.append({"role": "user", "content": turn["content"]})
        return out

    def ask(self, turns, *, thinking, kept, max_tokens, tools=None, required=False, schema=None, stop=None, system=None, cut=None):
        body = {"model": self.model, "messages": self.messages(turns, kept), "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        if system:
            body["messages"].insert(0, {"role": "system", "content": system})
        if stop:
            body["stop"] = stop
        if schema:
            # `schema` is a schema, or "object" for any JSON object
            body["response_format"] = {"type": "json_object"} if schema == "object" else \
                {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            if required:
                # `required` is True, or "named": the one function, by its name
                body["tool_choice"] = {"type": "function", "function": {"name": tools[0]["name"]}} if required == "named" else "required"
        if cut:
            return stream(self.url, "/v1/chat/completions", body, None, cut)
        message = post(self.url, "/v1/chat/completions", body)["choices"][0]
        said = message["message"]
        return {"role": "assistant", "content": said.get("content") or "", "reasoning": said.get("reasoning_content") or "",
                "calls": [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                          for c in said.get("tool_calls") or []], "finish": message["finish_reason"]}


class Responses:
    """The Responses API: every item sent again, or (`chained`) the new ones after `previous_response_id`."""

    def __init__(self, url, model, chained):
        self.url, self.model, self.chained = url, model, chained
        self.name = "responses, previous_response_id" if chained else "responses"

    def items(self, turns, kept):
        out = []
        for turn in turns:
            if turn["role"] == "assistant":
                if kept and turn["reasoning"]:
                    out.append({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": turn["reasoning"]}]})
                if turn["content"]:
                    out.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": turn["content"]}]})
                for c in turn["calls"]:
                    out.append({"type": "function_call", "call_id": c["id"], "name": c["name"], "arguments": c["arguments"]})
            elif turn["role"] == "tool":
                out.append({"type": "function_call_output", "call_id": turn["id"], "output": turn["content"]})
            else:
                out.append({"type": "message", "role": "user", "content": turn["content"]})
        return out

    def ask(self, turns, *, thinking, kept, max_tokens, tools=None, required=False, schema=None, stop=None, system=None, cut=None):
        body = {"model": self.model, "max_output_tokens": max_tokens, "temperature": 0}
        if system:
            body["instructions"] = system
        if stop:
            body["stop"] = stop
        if schema:
            body["text"] = {"format": {"type": "json_object"} if schema == "object" else
                            {"type": "json_schema", "name": "answer", "schema": schema}}
        # A chain goes on from the last reply that is in `turns` (an edit drops the replies after the edit).
        last = max((i for i, t in enumerate(turns) if t["role"] == "assistant"), default=None)
        if self.chained and last is not None and turns[last].get("response"):
            body["previous_response_id"] = turns[last]["response"]
            body["input"] = self.items(turns[last + 1:], kept)
        else:
            body["input"] = self.items(turns, kept)
        if not thinking:
            body["reasoning"] = {"effort": "none"}
        if tools:
            body["tools"] = [{"type": "function", **t} for t in tools]
            if required:
                body["tool_choice"] = {"type": "function", "name": tools[0]["name"]} if required == "named" else "required"
        if cut:
            return stream(self.url, "/v1/responses", body, None, cut)
        response = post(self.url, "/v1/responses", body)
        reply = {"role": "assistant", "content": "", "reasoning": "", "calls": [], "finish": response.get("status"),
                 "response": response["id"]}
        for item in response["output"]:
            if item["type"] == "reasoning":
                reply["reasoning"] += "".join(p.get("text", "") for p in item.get("content") or [])
            elif item["type"] == "message":
                reply["content"] += "".join(p.get("text", "") for p in item["content"])
            elif item["type"] == "function_call":
                reply["calls"].append({"id": item["call_id"], "name": item["name"], "arguments": item["arguments"]})
        return reply


class Messages:
    """Anthropic's Messages: thinking, text and tool_use blocks sent back as they came."""

    name = "messages"

    def __init__(self, url, model):
        self.url, self.model = url, model

    def messages(self, turns, kept):
        out = []
        for turn in turns:
            if turn["role"] == "assistant":
                blocks = []
                if kept and turn["reasoning"]:
                    blocks.append({"type": "thinking", "thinking": turn["reasoning"], "signature": ""})
                if turn["content"]:
                    blocks.append({"type": "text", "text": turn["content"]})
                for c in turn["calls"]:
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": json.loads(c["arguments"])})
                out.append({"role": "assistant", "content": blocks})
            elif turn["role"] == "tool":
                out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": turn["id"], "content": turn["content"]}]})
            else:
                out.append({"role": "user", "content": turn["content"]})
        return out

    def ask(self, turns, *, thinking, kept, max_tokens, tools=None, required=False, schema=None, stop=None, system=None, cut=None):
        body = {"model": self.model, "messages": self.messages(turns, kept), "max_tokens": max_tokens, "temperature": 0}
        if system:
            body["system"] = system
        if stop:
            body["stop_sequences"] = stop
        if schema:
            if schema == "object":
                raise RuntimeError("Messages has no JSON object format")
            body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        if thinking:
            body["thinking"] = {"type": "adaptive"}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
            if required:
                body["tool_choice"] = {"type": "tool", "name": tools[0]["name"]} if required == "named" else {"type": "any"}
        if cut:
            return stream(self.url, "/v1/messages", body, {"anthropic-version": "2023-06-01"}, cut)
        response = post(self.url, "/v1/messages", body, {"anthropic-version": "2023-06-01"})
        reply = {"role": "assistant", "content": "", "reasoning": "", "calls": [], "finish": response.get("stop_reason")}
        for block in response["content"]:
            if block["type"] == "thinking":
                reply["reasoning"] += block["thinking"]
            elif block["type"] == "text":
                reply["content"] += block["text"]
            elif block["type"] == "tool_use":
                reply["calls"].append({"id": block["id"], "name": block["name"], "arguments": json.dumps(block["input"])})
        return reply


class Run:
    def __init__(self, log, seed):
        self.log, self.seed, self.rows, self.seen = log, seed, [], 0
        self.seen = len(open(log, encoding="utf-8", errors="replace").read().splitlines())

    def salt(self, *names):
        return hashlib.sha256(":".join([str(self.seed), *names]).encode()).hexdigest()[:12]

    def answered(self):
        """The server's lines for the request just answered, and the one that ends them."""

        for _ in range(1200):
            lines = open(self.log, encoding="utf-8", errors="replace").read().splitlines()
            fresh = lines[self.seen:]
            done = [m for m in map(DONE.search, fresh) if m]
            if done:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("the server's log has no line for the reply")
        self.seen = len(lines)
        return fresh, done

    def record(self, api, case, step, reply):
        fresh, done = self.answered()
        prompt, restored, reply_tokens = int(done[-1][1]), int(done[-1][2]), int(done[-1][4])
        row = {"api": api, "case": case, "request": step, "prompt": prompt, "restored": restored, "run_again": prompt - restored,
               "reply": reply_tokens, "finish": reply["finish"], "calls": len(reply["calls"]), "snapshots": done[-1][5].strip(),
               "strides": (done[-1][6] or "").strip(), "prompt_s": float(done[-1][3])}
        self.rows.append(row)
        print(f"{api:32s} {case:22s} {step:16s} prompt {prompt:5d} restored {restored:5d} again {prompt - restored:4d} "
              f"reply {reply_tokens:4d} {reply['finish']}", flush=True)

    def turns(self, client, case, *, thinking, kept, max_tokens=400, switch=False, schema=None):
        """Three turns, then the third, the second and the first message edited. `schema` (chat completions
        only): every reply is held to it."""

        salt = self.salt(client.name, case)
        extra = {"schema": schema} if schema else {}
        ask = lambda turns, on=thinking: client.ask(turns, thinking=on, kept=kept, max_tokens=max_tokens, **extra)
        user = lambda text: {"role": "user", "content": text}
        u1 = user(f"Session {salt}. In about sixty words, say how a bicycle stays upright when it moves.")
        a1 = ask([u1])
        self.record(client.name, case, "first", a1)
        u2 = user("Now, in about sixty words, say why it falls over when it stops.")
        # `switch`: thinking is turned on for the second turn alone
        a2 = ask([u1, a1, u2], (not thinking) if switch else thinking)
        self.record(client.name, case, "second turn", a2)
        if switch:
            return
        a3 = ask([u1, a1, u2, a2, user("And in about sixty words, what a tricycle does differently.")])
        self.record(client.name, case, "third turn", a3)
        self.record(client.name, case, "third edited", ask([u1, a1, u2, a2, user("And in about sixty words, what a unicycle demands of its rider.")]))
        self.record(client.name, case, "second edited", ask([u1, a1, user("Now, in about sixty words, say how its rider steers it.")]))
        self.record(client.name, case, "first edited", ask([user(f"Session {salt}. In about sixty words, say how a spinning top stays upright.")]))

    def calls(self, client, case, *, thinking, kept, required, tool=WEATHER, max_tokens=400):
        """A call, its result, the answer, a new turn, and that turn's message edited."""

        salt = self.salt(client.name, case)
        ask = lambda turns, must=False: client.ask(turns, thinking=thinking, kept=kept, max_tokens=max_tokens, tools=[tool], required=must)
        user = lambda text: {"role": "user", "content": text}
        question = "Use the tool to look up the key 'alpha-7', then tell me what it holds." if tool is LOOKUP else \
            "What's the weather in Paris for the next 3 days, in celsius? Use the tool."
        u1 = user(f"Session {salt}. {question}")
        a1 = ask([u1], required)
        self.record(client.name, case, "first (the call)", a1)
        if not a1["calls"]:
            print(f"  no call was made: {a1['content'][:120]!r}")
            return
        result = "{\"value\": \"forty-two crates of oranges\"}" if tool is LOOKUP else \
            "{\"city\": \"Paris\", \"units\": \"c\", \"days\": [{\"high\": 21, \"low\": 12}, {\"high\": 19, \"low\": 11}, {\"high\": 23, \"low\": 13}]}"
        results = [{"role": "tool", "id": c["id"], "content": result} for c in a1["calls"]]
        a2 = ask([u1, a1, *results])
        self.record(client.name, case, "the result", a2)
        u2 = user("Thanks. In about forty words, which day is best for a walk, and why?")
        a3 = ask([u1, a1, *results, a2, u2])
        self.record(client.name, case, "new turn", a3)
        self.record(client.name, case, "new turn edited", ask([u1, a1, *results, a2, user("Thanks. In about forty words, what should I wear?")]))
        self.record(client.name, case, "result edited", ask([u1, a1, *[{**r, "content": r["content"].replace("2", "3")} for r in results]]))

    def pair(self, client, case, *, question=None, first=None, then=None, back=None, again=False, **common):
        """A reply and the turn after it (spec 035). `common` is asked of both requests, `first` and `then` of
        one; `back` changes the reply as the client sends it back; `again` sends the first request once more
        before the next turn. `first["past"]`: the reply is cut by the length that many tokens after its
        reasoning ends, wherever that is (the model reasons for 350 to 1500 tokens on any question tried, so no
        fixed length falls in the content): the request is first sent streamed and hung up on at its first
        delta of content, which the server's log counts, then another conversation takes the session, so
        that the reply's row is a first-met prompt's as the other rows' are. Neither is a row."""

        salt = self.salt(client.name, case)
        common = {"thinking": False, "kept": True, "max_tokens": 400, **common}
        user = lambda text: {"role": "user", "content": text}
        u1 = user(f"Session {salt}. {question or 'In about sixty words, say how a bicycle stays upright when it moves.'}")
        if first and "past" in first:
            client.ask([u1], **{**common, "max_tokens": 2400, "cut": ("content", 1)})
            reasoned = int(self.answered()[1][-1][4])
            client.ask([user(f"Session {salt}, aside. Say hello.")], **{**common, "thinking": False, "max_tokens": 1})
            self.answered()
            first = {**{k: v for k, v in first.items() if k != "past"}, "max_tokens": reasoned + first["past"]}
        a1 = client.ask([u1], **{**common, **(first or {})})
        self.record(client.name, case, "the reply", a1)
        self.rows[-1]["said"] = (a1["reasoning"][-40:], a1["content"][:60], a1["content"][-40:], [c["arguments"] for c in a1["calls"]])
        if again:
            # whole this time, where the first was hung up on
            a1 = client.ask([u1], **{**common, **{k: v for k, v in (first or {}).items() if k != "cut"}})
            self.record(client.name, case, "sent again", a1)
        results = [{"role": "tool", "id": c["id"], "content": "{\"city\": \"Paris\", \"days\": [{\"high\": 21, \"low\": 12}]}"} for c in a1["calls"]]
        if back:
            a1 = back(dict(a1))
        u2 = [] if results else [user("Now, in about sixty words, say why it falls over when it stops.")]
        a2 = client.ask([u1, a1, *results, *u2], **{**common, **(then or {})})
        self.record(client.name, case, "the next turn", a2)


def table(rows):
    print("\n| API | Conversation | Request | Prompt | Restored at | Run again |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['api']} | {r['case']} | {r['request']} | {r['prompt']} | {r['restored']} | {r['run_again']} |")


def shapes(run, clients):
    """Spec 035's plan: every reply's shape with the turn after it, through each API that can ask for it."""

    chat, items, chained, messages = clients
    think = {"thinking": True, "max_tokens": 1200}
    weather, both = {"tools": [WEATHER]}, {"tools": [WEATHER, LOOKUP]}
    cities = "What's the weather in Paris and in Tokyo tomorrow, in celsius? Use the tool once for each city."
    after = "Say in one short sentence what you are about to do, then use the tool for the weather in Paris tomorrow."
    listed = {"name": "note", "description": "Store a note", "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}, "pinned": {"type": "boolean"},
        "meta": {"type": "object", "properties": {"priority": {"type": "integer"}, "owner": {"type": "string"}}}},
        "required": ["title", "tags", "pinned", "meta"]}}
    note = "Store a note titled 'Café menu' with the tags 'food' and 'paris', pinned, priority 2, owner 'Zoë'. Use the tool."
    story = "Write a story of about seven hundred words about a bicycle that is passed down through three generations."
    spaced = lambda before, behind: lambda reply: {**reply, "content": before + reply["content"] + behind}
    plan = []
    for client in clients:
        cases = [
            ("a schema's reply", dict(schema=ANSWER)),
            ("a schema's reply, thinking", dict(schema=ANSWER, thinking=True, max_tokens=2400)),
            ("a JSON object, thinking", dict(schema="object", question="As a JSON object with the keys 'answer' and 'why', say whether a bicycle stays upright by itself.", **think)),
            ("a document beside tools, thinking", dict(schema=ANSWER, **weather, **think)),
            ("two calls", dict(question=cities, **weather)),
            ("two calls, thinking", dict(question=cities, **weather, **think)),
            ("two required calls, thinking", dict(question=cities, required=True, **weather, **think)),
            ("a call after content", dict(question=after, **weather)),
            ("a call after content, thinking", dict(question=after, **weather, **think)),
            ("a call with JSON values", dict(question=note, tools=[listed])),
            ("a required call with JSON values", dict(question=note, tools=[listed], required=True)),
            ("a required call with JSON values, thinking", dict(question=note, tools=[listed], required=True, **think)),
            ("a stop string", dict(first=dict(stop=["."]))),
            ("a stop string, thinking", dict(first=dict(stop=["."]), **think)),
            ("a stop string after a blank line", dict(question="Write two short paragraphs about bicycles; begin the second with the word Second.", first=dict(stop=["Second"]))),
            ("the length, in content", dict(first=dict(max_tokens=24))),
            ("the length, in content, thinking", dict(thinking=True, max_tokens=1200, first=dict(past=16))),
            ("the length, in reasoning", dict(thinking=True, first=dict(max_tokens=24), then=dict(max_tokens=1200))),
            ("the length, in a call", dict(question=cities, **weather, first=dict(max_tokens=14))),
            ("the length, in a required call", dict(question=cities, required=True, **weather, first=dict(max_tokens=14), then=dict(required=False))),
            ("hung up in content", dict(first=dict(cut=("content", 12)))),
            ("hung up in content, sent again", dict(first=dict(cut=("content", 12)), then=None, again=True)),
            ("hung up in reasoning", dict(thinking=True, max_tokens=1200, first=dict(cut=("reasoning", 12)))),
            ("hung up in content, thinking", dict(thinking=True, max_tokens=2400, first=dict(cut=("content", 12)))),
            ("hung up late in a long reply", dict(question=story, max_tokens=1600, first=dict(cut=("content", 600)))),
            ("blanks before the content", dict(back=spaced("\n\n", ""))),
            ("blanks after the content", dict(back=spaced("", "\n\n"))),
            ("blanks around the content, thinking", dict(back=spaced(" ", " \n"), **think)),
            ("thinking switched off", dict(max_tokens=1200, first=dict(thinking=True))),
            ("thinking switched on", dict(max_tokens=1200, then=dict(thinking=True))),
            ("a system prompt", dict(system="Answer plainly.")),
            ("a system prompt changed", dict(first=dict(system="Answer plainly."), then=dict(system="Answer plainly and briefly."))),
            ("tools changed", dict(first=weather, then=both)),
            ("sent again", dict(again=True)),
            ("sent again, thinking", dict(again=True, **think)),
            ("a call sent again", dict(question=cities, again=True, **weather)),
        ]
        for name, options in cases:
            if client is messages and options.get("schema") == "object":
                continue
            plan.append((client, name, lambda c, n, options=options: run.pair(c, n, **options)))
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="m")
    parser.add_argument("--log", required=True)
    parser.add_argument("--seed", default="1")
    parser.add_argument("--only", default="", help="substrings of the API's or the conversation's name, between commas: any of them")
    parser.add_argument("--api", default="", help="the API's name, whole")
    parser.add_argument("--set", default="turns", choices=["turns", "shapes", "all"],
                        help="turns: spec 030's conversations; shapes: spec 035's replies, each with its next turn")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out and args.only == "table":
        return table(json.load(open(args.out)))
    run = Run(args.log, args.seed)
    chat, items, chained, messages = Chat(args.url, args.model), Responses(args.url, args.model, False), \
        Responses(args.url, args.model, True), Messages(args.url, args.model)
    plan = []
    for client in (chat, items, chained, messages):
        plan += [
            (client, "thinking off", lambda c, n: run.turns(c, n, thinking=False, kept=True)),
            (client, "thinking, sent back", lambda c, n: run.turns(c, n, thinking=True, kept=True, max_tokens=1200)),
            (client, "tools, thinking off", lambda c, n: run.calls(c, n, thinking=False, kept=True, required=False)),
            (client, "tools, required call", lambda c, n: run.calls(c, n, thinking=False, kept=True, required=True)),
            (client, "tools, thinking", lambda c, n: run.calls(c, n, thinking=True, kept=True, required=False, max_tokens=1200)),
            # spec 034: the gate's call after the reasoning, and the tool result's turn after it
            (client, "tools, required, thinking", lambda c, n: run.calls(c, n, thinking=True, kept=True, required=True, max_tokens=1200)),
            (client, "tools, named, thinking", lambda c, n: run.calls(c, n, thinking=True, kept=True, required="named", max_tokens=1200)),
        ]
    plan += [
        (chat, "thinking, left out", lambda c, n: run.turns(c, n, thinking=True, kept=False, max_tokens=1200)),
        (messages, "thinking, left out", lambda c, n: run.turns(c, n, thinking=True, kept=False, max_tokens=1200)),
        (chat, "cut by the length", lambda c, n: run.turns(c, n, thinking=False, kept=True, max_tokens=24)),
        (chat, "thinking cut by the length", lambda c, n: run.turns(c, n, thinking=True, kept=True, max_tokens=24)),
        (chat, "thinking switched on", lambda c, n: run.turns(c, n, thinking=False, kept=True, switch=True)),
        (chat, "tools, a JSON call", lambda c, n: run.calls(c, n, thinking=False, kept=True, required=True, tool=LOOKUP)),
        (chat, "a schema's reply, thinking", lambda c, n: run.turns(c, n, thinking=True, kept=True, max_tokens=1200, schema=ANSWER)),
    ]
    if args.set == "shapes":
        plan = []
    if args.set != "turns":
        plan += shapes(run, (chat, items, chained, messages))
    for client, case, go in plan:
        if args.only and not any(name in client.name or name in case for name in args.only.split(",")):
            continue
        if args.api and args.api != client.name:
            continue
        try:
            go(client, case)
        except Exception as error:  # noqa: BLE001 - one conversation's failure is reported, the rest still run
            print(f"{client.name} / {case}: {error}", file=sys.stderr, flush=True)
            # whatever the failed request logged belongs to no row
            run.seen = len(open(args.log, encoding="utf-8", errors="replace").read().splitlines())
    if args.out:
        json.dump(run.rows, open(args.out, "w"), indent=1)
    table(run.rows)


if __name__ == "__main__":
    main()
