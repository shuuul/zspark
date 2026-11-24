"""The `anthropic` package against `zspark serve`: a reply whole and streamed, a streamed tool use and its result
answered, `count_tokens` against the reply's own usage, a stop sequence, then structured output
(`output_config.format` whole and streamed, `tool_choice` any and tool, a Pydantic model's schema, refusals),
then tool uses as they stream (spec 027): a free use with thinking and without, `any` and a named tool, two uses in
a reply, each streamed Message the one sent whole, and a block open with its input under way before the reply ends.

Not a fixture and never runtime: a check of the server by the `anthropic` package, run by hand. With
`zspark serve DIR --port 8011 --name qwen3.8-flash-next` listening (`ZSPARK_URL` names another address):

    uv run --with anthropic --with jsonschema --with pydantic python tools/reference/anthropic_exchange.py

It prints what it received and ends with OK, or fails on the first thing that is not as it should be.
"""

import json, os, sys, time
from typing import Optional
import anthropic, jsonschema
from anthropic import Anthropic
from pydantic import BaseModel

c = Anthropic(base_url=os.environ.get("ZSPARK_URL", "http://127.0.0.1:8011"), api_key="none")
M = "qwen3.8-flash-next"
tools = [{"name": "get_weather", "description": "Current weather for a city",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
          "units": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}]


def prompt_tokens(usage):
    """A reply's prompt in tokens: `input_tokens` leaves out the ones the server held from the request before."""
    return usage.input_tokens + (usage.cache_read_input_tokens or 0)


def streamed(**request):
    """The stream's event types in order, and the message its events add up to."""
    kinds, blocks, stop, usage = [], [], None, None
    for event in c.messages.create(stream=True, **request):
        kinds.append(event.type)
        if event.type == "message_start": assert event.message.role == "assistant" and event.message.content == []
        elif event.type == "content_block_start":
            assert event.index == len(blocks), (event.index, len(blocks))
            block = event.content_block
            blocks.append({"type": block.type, "body": "", "id": getattr(block, "id", None), "name": getattr(block, "name", None)})
        elif event.type == "content_block_delta":
            d = event.delta
            blocks[event.index]["body"] += {"thinking_delta": lambda: d.thinking, "text_delta": lambda: d.text,
                                            "input_json_delta": lambda: d.partial_json, "signature_delta": lambda: ""}[d.type]()
        elif event.type == "message_delta": stop, usage = event.delta, event.usage
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"], kinds
    assert kinds.count("content_block_start") == kinds.count("content_block_stop") == len(blocks)
    return kinds, blocks, stop, usage


# a reply whole, with a system prompt; thinking is off unless asked for
plain = dict(model=M, max_tokens=200, system="Answer in one short sentence.",
             messages=[{"role": "user", "content": "What is the capital of France?"}])
r = c.messages.create(**plain)
print("whole:", r.stop_reason, [b.type for b in r.content], repr(r.content[-1].text)); print("usage:", r.usage)
assert r.stop_reason == "end_turn" and [b.type for b in r.content] == ["text"] and "Paris" in r.content[0].text

# count_tokens for the same request is the prompt the reply's usage reports
counted = c.messages.count_tokens(**{k: v for k, v in plain.items() if k != "max_tokens"})
print("count_tokens:", counted.input_tokens, "| the reply's prompt:", prompt_tokens(r.usage))
assert counted.input_tokens == prompt_tokens(r.usage)

# the same streamed: the deltas are the whole reply's text
kinds, blocks, stop, usage = streamed(**plain)
print("streamed:", stop.stop_reason, blocks, "| events:", [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]])
print("usage:", usage)
assert stop.stop_reason == "end_turn" and len(blocks) == 1 and blocks[0]["body"] == r.content[0].text
assert prompt_tokens(usage) == counted.input_tokens and usage.output_tokens == r.usage.output_tokens

# a streamed tool use, with thinking
asking = dict(model=M, max_tokens=1024, tools=tools, thinking={"type": "enabled", "budget_tokens": 512}, extra_body={"temperature": 0.6},
              messages=[{"role": "user", "content": "What's the weather in Paris for the next 3 days, in celsius? Use the tool."}])
kinds, blocks, stop, usage = streamed(**asking)
print("events:", [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]])
print("blocks:", [(b["type"], b["name"], b["body"][:200].replace("\n", " ")) for b in blocks])
print("stop:", stop.stop_reason, "| usage:", usage)
uses = [b for b in blocks if b["type"] == "tool_use"]
assert stop.stop_reason == "tool_use" and len(uses) == 1 and uses[0]["name"] == "get_weather", "no tool use"
assert blocks[0]["type"] == "thinking" and blocks[0]["body"], "no thinking block"
args = json.loads(uses[0]["body"]); assert args["city"].lower().startswith("paris") and args.get("days") == 3, args
counted = c.messages.count_tokens(**{k: v for k, v in asking.items() if k not in ("max_tokens", "extra_body")})
print("count_tokens with tools and thinking:", counted.input_tokens, "| the reply's prompt:", prompt_tokens(usage))
assert counted.input_tokens == prompt_tokens(usage)

# the package's own accumulation of the same stream (`messages.stream`) ends with the same blocks
with c.messages.stream(**asking) as helper: built = helper.get_final_message()
print("helper:", built.stop_reason, [(b.type, getattr(b, "input", None)) for b in built.content])
assert built.stop_reason == "tool_use" and [b.type for b in built.content] == [b["type"] for b in blocks]
assert built.content[-1].input == args and built.content[0].thinking == blocks[0]["body"]

# its result answered, the assistant's turn sent back as the blocks it came in
turn = [{"type": "thinking", "thinking": b["body"], "signature": ""} if b["type"] == "thinking" else
        {"type": "text", "text": b["body"]} if b["type"] == "text" else
        {"type": "tool_use", "id": b["id"], "name": b["name"], "input": json.loads(b["body"])} for b in blocks]
result = {"type": "tool_result", "tool_use_id": uses[0]["id"], "content": json.dumps({"forecast": [
    {"day": 1, "high": 21, "sky": "sunny"}, {"day": 2, "high": 18, "sky": "rain"}, {"day": 3, "high": 19, "sky": "cloudy"}]})}
answering = dict(asking, messages=asking["messages"] + [{"role": "assistant", "content": turn}, {"role": "user", "content": [result]}])
second = c.messages.create(**answering)
said = "".join(b.text for b in second.content if b.type == "text")
print("second:", second.stop_reason, [b.type for b in second.content], "|", said); print("usage:", second.usage)
assert second.stop_reason == "end_turn" and "21" in said
assert second.usage.cache_read_input_tokens > 0, "the second turn held nothing of the first"
counted = c.messages.count_tokens(**{k: v for k, v in answering.items() if k not in ("max_tokens", "extra_body")})
print("count_tokens of the second turn:", counted.input_tokens, "| the reply's prompt:", prompt_tokens(second.usage))
assert counted.input_tokens == prompt_tokens(second.usage)

# a stop sequence is reported, whole and streamed
counting = dict(model=M, max_tokens=200, stop_sequences=["7"], extra_body={"temperature": 0},
                messages=[{"role": "user", "content": "Count from 1 to 20, one number a line."}])
s = c.messages.create(**counting)
print("stop:", repr(s.content[0].text), s.stop_reason, repr(s.stop_sequence))
assert s.stop_reason == "stop_sequence" and s.stop_sequence == "7" and "7" not in s.content[0].text and "6" in s.content[0].text
kinds, blocks, stop, usage = streamed(**counting)
assert stop.stop_reason == "stop_sequence" and stop.stop_sequence == "7" and blocks[0]["body"] == s.content[0].text

# a length cut
cut = c.messages.create(model=M, max_tokens=16, messages=[{"role": "user", "content": "Count from 1 to 200, one number a line."}])
print("cut:", cut.stop_reason, cut.usage.output_tokens)
assert cut.stop_reason == "max_tokens" and cut.usage.output_tokens == 16


# -- structured output --------------------------------------------------------------------------------------------

class Address(BaseModel):
    city: str
    country: str


class Person(BaseModel):
    """A Pydantic model: its schema has `$defs`, a `$ref`, and an `anyOf` with null for the Optional."""
    name: str
    age: int
    address: Address
    nickname: Optional[str] = None
    languages: list[str]


city = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
        "landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 3}, "coastal": {"type": "boolean"}},
        "required": ["city", "population", "landmarks", "coastal"], "additionalProperties": False}
patterned = {"type": "object", "properties": {"code": {"type": "string", "pattern": "^[A-Z]{3}$"}}, "required": ["code"]}
about_city = "Describe Lisbon as JSON: its name, population, up to three landmarks, and whether it is coastal."
about_person = "Invent a person who lives in Kyoto and speaks two languages, as JSON."
clock = {"name": "get_time", "description": "The time in a timezone",
         "input_schema": {"type": "object", "properties": {"timezone": {"type": "string"}, "hours24": {"type": "boolean"}},
                          "required": ["timezone", "hours24"], "additionalProperties": False}}
parameters = {t["name"]: t["input_schema"] for t in (tools[0], clock)}


def reply(stream, text, **request):
    """A Message's text, its tool uses as (name, input) and its stop reason, whole or from a stream. The body's
    fields go as `extra_body`, so the package's version does not decide what is sent."""
    request = dict(model=M, max_tokens=600, messages=[{"role": "user", "content": text}], extra_body=dict(request, temperature=0.6, seed=3))
    if not stream:
        r = c.messages.create(**request)
        return "".join(b.text for b in r.content if b.type == "text"), [(b.name, b.input) for b in r.content if b.type == "tool_use"], r.stop_reason
    _, blocks, stop, _ = streamed(**request)
    return ("".join(b["body"] for b in blocks if b["type"] == "text"),
            [(b["name"], json.loads(b["body"])) for b in blocks if b["type"] == "tool_use"], stop.stop_reason)


def formatted(schema): return {"output_config": {"format": {"type": "json_schema", "schema": schema}}}


# output_config.format: a schema, whole and streamed, with thinking and without
for stream in (False, True):
    for more in ({}, {"thinking": {"type": "enabled", "budget_tokens": 256}}):
        text, uses, stop = reply(stream, about_city, **formatted(city), **more)
        print(f"json_schema (stream {stream}, {'thinking' if more else 'no thinking'}):", stop, text)
        jsonschema.validate(json.loads(text), city); assert stop == "end_turn" and not uses

# a Pydantic model's own schema ($defs, $ref, anyOf with null)
for stream in (False, True):
    text, _, stop = reply(stream, about_person, **formatted(Person.model_json_schema()))
    print(f"pydantic (stream {stream}):", stop, text); print("  ->", Person.model_validate_json(text))

# `any` where the model would not have used a tool, and `tool` where it would have used the other
for stream in (False, True):
    text, uses, stop = reply(stream, "Say hello to Ada.", tools=[tools[0], clock], tool_choice={"type": "any"})
    print(f"any (stream {stream}):", stop, uses, repr(text))
    assert stop == "tool_use" and uses and not text
    for name, given in uses: jsonschema.validate(given, parameters[name])
    text, uses, stop = reply(stream, "What's the weather in Paris?", tools=[tools[0], clock],
                             tool_choice={"type": "tool", "name": "get_time", "disable_parallel_tool_use": True})
    print(f"tool (stream {stream}):", stop, uses, repr(text))
    assert stop == "tool_use" and len(uses) == 1 and uses[0][0] == "get_time" and not text
    jsonschema.validate(uses[0][1], parameters["get_time"])

# A required call whose arguments are a text of several lines, a nested object, an array and an enum. The server
# has the model write such a call as the chat template does (each parameter apart, a string as it is), and what
# arrives is the arguments as JSON, each of its type.
note = {"type": "object", "properties": {
    "title": {"type": "string"}, "body": {"type": "string", "description": "The note's whole text"},
    "meta": {"type": "object", "properties": {"lang": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["lang", "lines"], "additionalProperties": False},
    "tags": {"type": "array", "items": {"type": "string"}}, "priority": {"type": "string", "enum": ["low", "high"]}, "score": {"type": "number"}},
    "required": ["title", "body", "meta", "tags", "priority"], "additionalProperties": False}
about_note = 'Save a note titled "Rain" whose body is a three-line haiku about rain, each line on its own line, tagged poem and weather, high priority.'
for stream in (False, True):
    text, uses, stop = reply(stream, about_note, tool_choice={"type": "any", "disable_parallel_tool_use": True},
                             tools=[{"name": "save_note", "description": "Save a note", "input_schema": note}])
    print(f"any, a text of several lines (stream {stream}):", stop, uses, repr(text))
    assert stop == "tool_use" and len(uses) == 1 and uses[0][0] == "save_note" and not text
    given = uses[0][1]; jsonschema.validate(given, note)
    assert isinstance(given["body"], str) and "\n" in given["body"].strip() and isinstance(given["meta"]["lines"], int) and isinstance(given["tags"], list), given

# count_tokens reads the same fields and compiles nothing: a count is the prompt's, whatever the schema
counted = c.messages.count_tokens(model=M, messages=[{"role": "user", "content": "hi"}], extra_body=formatted(patterned))
print("count_tokens beside a schema that would be refused:", counted.input_tokens)

# refusals, in Anthropic's error shape: what cannot be enforced names its keyword
fussy = {"name": "lookup", "input_schema": patterned}
for extra in (formatted(patterned), dict(formatted(patterned), stream=True), {"tools": [fussy], "tool_choice": {"type": "any"}},
              {"tools": [fussy], "tool_choice": {"type": "tool", "name": "lookup"}},
              dict(formatted(city), tools=tools, tool_choice={"type": "any"}),
              {"output_config": {"format": {"type": "json_object"}}}):
    try:
        r = c.messages.create(model=M, max_tokens=32, messages=[{"role": "user", "content": "hi"}], extra_body=extra)
        if extra.get("stream"): list(r)
        sys.exit(f"{extra} was not refused")
    except anthropic.BadRequestError as e:
        print("refused:", e.status_code, e.body)
        assert e.body["type"] == "error" and e.body["error"]["type"] == "invalid_request_error", e.body
        assert any(word in e.body["error"]["message"] for word in ("pattern", "cannot be combined", "must be json_schema")), e.body

# -- a tool use streamed as it is written (spec 027) --------------------------------------------------------------
save = {"name": "save_note", "description": "Save a note", "input_schema": note}
about_two = "What's the weather in Paris and in Tokyo, in celsius? Call the tool once for each city, both calls in this one turn."
about_long = ('Save a note titled "Rain" whose body is a poem of eight lines about rain, each line on its own line, '
              "tagged poem and weather, high priority. Call the tool.")


def shape(message):
    """What a client reads of a Message: its thinking, its text, its tool uses, its stop reason and its tokens. A
    Message sent whole has its thinking trimmed, a stream sends it as it came: the two are compared trimmed."""
    return ("".join(b.thinking for b in message.content if b.type == "thinking").strip(), "".join(b.text for b in message.content if b.type == "text"),
            [(b.name, b.input) for b in message.content if b.type == "tool_use"], message.stop_reason,
            (prompt_tokens(message.usage), message.usage.output_tokens))


def as_written(**request):
    """A streamed Message as `shape` gives it, from its events, and when they came: for each tool use the number
    of `input_json_delta`s and the seconds to its block's start, to its last delta and to the stream's end."""
    began = time.perf_counter()
    blocks, stop, usage = [], None, None
    for event in c.messages.create(stream=True, **request):
        now = time.perf_counter() - began
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use": assert block.input == {} and block.name and block.id, block
            blocks.append({"type": block.type, "body": "", "name": getattr(block, "name", None), "deltas": 0, "named": now, "last": None, "open": True})
        elif event.type == "content_block_delta":
            d, b = event.delta, blocks[event.index]
            assert event.index == len(blocks) - 1 and b["open"], "a delta for a block that is not the open one"
            piece = {"thinking_delta": lambda: d.thinking, "text_delta": lambda: d.text, "input_json_delta": lambda: d.partial_json,
                     "signature_delta": lambda: ""}[d.type]()
            b["body"] += piece
            if d.type == "input_json_delta" and piece: b["deltas"] += 1; b["last"] = now
        elif event.type == "content_block_stop": blocks[event.index]["open"] = False
        elif event.type == "message_delta": stop, usage = event.delta, event.usage
    ended = time.perf_counter() - began
    uses = [b for b in blocks if b["type"] == "tool_use"]
    built = ("".join(b["body"] for b in blocks if b["type"] == "thinking").strip(), "".join(b["body"] for b in blocks if b["type"] == "text"),
             [(b["name"], json.loads(b["body"])) for b in uses], stop.stop_reason, (prompt_tokens(usage), usage.output_tokens))
    return built, [(b["deltas"], b["named"], b["last"], b["body"]) for b in uses], ended


thinks = {"thinking": {"type": "enabled", "budget_tokens": 512}}
cases = [("a free use, a long string", dict(text=about_long, tools=[save]), 1),
         ("a free use, with thinking", dict(text=about_long, tools=[save], **thinks), 1),
         ("two uses in a reply", dict(text=about_two, tools=tools), 2),
         ("two asked, one allowed", dict(text=about_two, tools=tools, tool_choice={"type": "auto", "disable_parallel_tool_use": True}), 1),
         ("any", dict(text=about_long, tools=[save], tool_choice={"type": "any", "disable_parallel_tool_use": True}), 1),
         ("a named tool", dict(text="What's the weather in Paris?", tools=[tools[0], clock],
                               tool_choice={"type": "tool", "name": "get_time", "disable_parallel_tool_use": True}), 1)]
for name, request, count in cases:
    text = request.pop("text")
    request = dict(model=M, max_tokens=1200, messages=[{"role": "user", "content": text}], extra_body=dict(request, temperature=0.6, seed=5))
    sent, (got, timing, ended) = shape(c.messages.create(**request)), as_written(**request)
    print(f"streamed, {name}: {got[3]}, {len(got[2])} use(s), text {got[1]!r}, thinking {len(got[0])} characters")
    for (use, given), (deltas, named, last, body) in zip(got[2], timing):
        print(f"  {use} in {deltas} deltas, block at {named:.2f} s, last delta {last:.2f} s, end {ended:.2f} s: {body[:120]}")
    assert got == sent, f"the stream is not the Message sent whole:\n{got}\n{sent}"
    assert got[3] == "tool_use" and len(got[2]) == count, got
    assert bool(got[0]) == ("thinking" in request["extra_body"]), "thinking where none was asked, or none where it was"
    for use, given in got[2]: jsonschema.validate(given, note if use == "save_note" else parameters[use])
    assert all(named < last for _, named, last, _ in timing), timing
    if "save_note" == got[2][0][0]: assert timing[0][0] >= 8, f"a long string came in {timing[0][0]} deltas"
print("OK")
