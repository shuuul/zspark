"""An OpenAI client against `zspark serve`: a streamed tool call, its result answered, a stop string, then structured
output (a reply held to a JSON schema, whole and streamed; a required and a named tool call, and one whose
arguments are a text of several lines beside nested values; a Pydantic model's schema; a schema that is refused),
then calls as they stream (spec 027): a free, a required and a named call, two calls in a reply, with thinking and
without, each streamed reply the same reply sent whole, and a call's arguments under way before the reply ends.

Not a fixture and never runtime: a check of the server by the `openai` package, run by hand. With
`zspark serve DIR --port 8011 --name qwen3.8-flash-next` listening (`ZSPARK_URL` names another address):

    uv run --with openai --with jsonschema --with pydantic python tools/reference/openai_exchange.py

It prints what it received and ends with OK, or fails on the first thing that is not as it should be.
"""

import json, os, sys, time
from typing import Optional
import jsonschema, openai
from openai import OpenAI
from pydantic import BaseModel
c = OpenAI(base_url=os.environ.get("ZSPARK_URL", "http://127.0.0.1:8011") + "/v1", api_key="none")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
          "units": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris for the next 3 days, in celsius? Use the tool."}]
stream = c.chat.completions.create(model="qwen3.8-flash-next", messages=msgs, tools=tools, stream=True,
                                   stream_options={"include_usage": True}, max_tokens=512, seed=1, temperature=0.6)
calls, content, reasoning, finish, usage = {}, "", "", None, None
for chunk in stream:
    if chunk.usage: usage = chunk.usage
    for ch in chunk.choices:
        d = ch.delta
        content += d.content or ""
        reasoning += getattr(d, "reasoning_content", None) or ""
        for tc in d.tool_calls or []:
            e = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id: e["id"] = tc.id
            if tc.function and tc.function.name: e["name"] += tc.function.name
            if tc.function and tc.function.arguments: e["arguments"] += tc.function.arguments
        if ch.finish_reason: finish = ch.finish_reason
print("reasoning:", reasoning[:200].replace("\n", " "))
print("content:", repr(content)); print("calls:", calls); print("finish:", finish); print("usage:", usage)
assert finish == "tool_calls" and calls[0]["name"] == "get_weather", "no tool call"
args = json.loads(calls[0]["arguments"]); assert args["city"].lower().startswith("paris") and args.get("days") == 3, args
msgs.append({"role": "assistant", "content": content or None, "tool_calls": [{"id": calls[0]["id"], "type": "function",
             "function": {"name": calls[0]["name"], "arguments": calls[0]["arguments"]}}]})
msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"forecast": [
    {"day": 1, "high": 21, "sky": "sunny"}, {"day": 2, "high": 18, "sky": "rain"}, {"day": 3, "high": 19, "sky": "cloudy"}]})})
r = c.chat.completions.create(model="qwen3.8-flash-next", messages=msgs, tools=tools, max_tokens=512, seed=1, temperature=0.6)
m = r.choices[0]
print("second finish:", m.finish_reason, "| content:", m.message.content); print("usage:", r.usage)
assert m.finish_reason == "stop" and "21" in m.message.content
s = c.chat.completions.create(model="qwen3.8-flash-next", messages=[{"role": "user", "content": "Count from 1 to 20, one number a line."}],
                              stop=["7"], max_tokens=200, extra_body={"enable_thinking": False}, temperature=0)
print("stop:", repr(s.choices[0].message.content), s.choices[0].finish_reason)
assert "7" not in s.choices[0].message.content and "6" in s.choices[0].message.content

# -- structured output --------------------------------------------------------------------------------------------
M = "qwen3.8-flash-next"

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
clock = {"type": "function", "function": {"name": "get_time", "description": "The time in a timezone", "strict": True,
         "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}, "hours24": {"type": "boolean"}},
                        "required": ["timezone", "hours24"], "additionalProperties": False}}}
parameters = {t["function"]["name"]: t["function"]["parameters"] for t in (tools[0], clock)}


def reply(stream, **request):
    """A reply's content, its calls as (name, arguments) and its finish reason, whole or added up from a stream."""
    request = dict(model=M, max_tokens=600, temperature=0.6, seed=3, **request)
    if not stream:
        choice = c.chat.completions.create(**request).choices[0]
        return choice.message.content or "", [(t.function.name, t.function.arguments) for t in choice.message.tool_calls or []], choice.finish_reason
    text, made, finish = "", {}, None
    for chunk in c.chat.completions.create(stream=True, **request):
        for ch in chunk.choices:
            text += ch.delta.content or ""
            for tc in ch.delta.tool_calls or []:
                e = made.setdefault(tc.index, ["", ""])
                e[0] += (tc.function.name or "") if tc.function else ""
                e[1] += (tc.function.arguments or "") if tc.function else ""
            finish = ch.finish_reason or finish
    return text, [tuple(made[i]) for i in sorted(made)], finish


def ask(text): return [{"role": "user", "content": text}]


# a reply held to a schema, whole and streamed, thinking and not; `strict` either way
for stream in (False, True):
    for more in ({}, {"extra_body": {"enable_thinking": False}}):
        text, made, finish = reply(stream, messages=ask(about_city), **more,
                                   response_format={"type": "json_schema", "json_schema": {"name": "city", "strict": not stream, "schema": city}})
        print(f"json_schema (stream {stream}, {more or 'thinking'}):", finish, text)
        jsonschema.validate(json.loads(text), city); assert finish == "stop" and not made
text, _, finish = reply(False, messages=ask("Give me any small JSON object about tea."), response_format={"type": "json_object"})
print("json_object:", finish, text); assert isinstance(json.loads(text), dict)

# a Pydantic model's own schema ($defs, $ref, anyOf with null), as it is and through the package's `parse`
for stream in (False, True):
    text, _, finish = reply(stream, messages=ask(about_person),
                            response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": Person.model_json_schema()}})
    print(f"pydantic (stream {stream}):", finish, text); print("  ->", Person.model_validate_json(text))
parse = getattr(c.chat.completions, "parse", None) or c.beta.chat.completions.parse
parsed = parse(model=M, messages=ask(about_person), response_format=Person, max_tokens=600, temperature=0.6, seed=4)
print("parse():", parsed.choices[0].message.parsed); assert isinstance(parsed.choices[0].message.parsed, Person)

# a required call where the model would not have called, and a named function where it would have called the other
for stream in (False, True):
    text, made, finish = reply(stream, messages=ask("Say hello to Ada."), tools=[tools[0], clock], tool_choice="required")
    print(f"required (stream {stream}):", finish, made, repr(text))
    assert finish == "tool_calls" and made and not text
    for name, arguments in made: jsonschema.validate(json.loads(arguments), parameters[name])
    text, made, finish = reply(stream, messages=ask("What's the weather in Paris?"), tools=[tools[0], clock],
                               tool_choice={"type": "function", "function": {"name": "get_time"}}, parallel_tool_calls=False)
    print(f"named (stream {stream}):", finish, made, repr(text))
    assert finish == "tool_calls" and len(made) == 1 and made[0][0] == "get_time" and not text
    jsonschema.validate(json.loads(made[0][1]), parameters["get_time"])

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
    text, made, finish = reply(stream, messages=ask(about_note), tool_choice="required", parallel_tool_calls=False,
                               tools=[{"type": "function", "function": {"name": "save_note", "description": "Save a note", "parameters": note}}])
    print(f"required, a text of several lines (stream {stream}):", finish, made, repr(text))
    assert finish == "tool_calls" and len(made) == 1 and made[0][0] == "save_note" and not text
    given = json.loads(made[0][1]); jsonschema.validate(given, note)
    assert isinstance(given["body"], str) and "\n" in given["body"].strip() and isinstance(given["meta"]["lines"], int) and isinstance(given["tags"], list), given

# what cannot be enforced is refused, and says which keyword; `strict: false` buys no leniency
fussy = {"type": "function", "function": {"name": "lookup", "parameters": patterned}}
for extra in ({"response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": patterned}}},
              {"response_format": {"type": "json_schema", "json_schema": {"name": "p", "strict": False, "schema": patterned}}, "stream": True},
              {"tools": [fussy], "tool_choice": "required"},
              {"tools": tools, "tool_choice": "required", "response_format": {"type": "json_object"}}):
    try:
        r = c.chat.completions.create(model=M, messages=ask("hi"), max_tokens=32, **extra)
        if extra.get("stream"): list(r)
        sys.exit(f"{extra} was not refused")
    except openai.BadRequestError as e:
        print("refused:", e.status_code, e.body)
        assert e.status_code == 400 and e.body["type"] == "invalid_request_error", e.body
        assert "pattern" in e.body["message"] or "cannot be combined" in e.body["message"], e.body
# the same tool with tool_choice auto is served as before: nothing reads its schema
r = c.chat.completions.create(model=M, messages=ask("Say hi."), tools=[fussy], max_tokens=200, extra_body={"enable_thinking": False})
print("auto with the same tool:", r.choices[0].finish_reason, repr(r.choices[0].message.content))

# -- a call streamed as it is written (spec 027) ------------------------------------------------------------------
save = {"type": "function", "function": {"name": "save_note", "description": "Save a note", "parameters": note}}
about_two = "What's the weather in Paris and in Tokyo, in celsius? Call the tool once for each city, both calls in this one turn."
about_long = ('Save a note titled "Rain" whose body is a poem of eight lines about rain, each line on its own line, '
              "tagged poem and weather, high priority. Call the tool.")


def whole(**request):
    r = c.chat.completions.create(**request)
    m = r.choices[0]
    return (m.message.content or "", [(t.function.name, t.function.arguments) for t in m.message.tool_calls or []], m.finish_reason,
            (r.usage.prompt_tokens, r.usage.completion_tokens))


def as_written(**request):
    """A streamed reply and when its pieces came: the reply as `whole` gives it, then for each call the number
    of argument deltas, and the seconds from the request to its name, to its last delta, and to the stream's end."""
    began = time.perf_counter()
    text, made, finish, usage, ids = "", {}, None, None, set()
    for chunk in c.chat.completions.create(stream=True, stream_options={"include_usage": True}, **request):
        now = time.perf_counter() - began
        if chunk.usage: usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
        for ch in chunk.choices:
            text += ch.delta.content or ""
            for tc in ch.delta.tool_calls or []:
                e = made.setdefault(tc.index, {"name": "", "arguments": "", "deltas": 0, "named": None, "last": None})
                if tc.id:
                    assert tc.index == len(made) - 1 and tc.id not in ids and tc.function.name and not tc.function.arguments, tc
                    ids.add(tc.id); e["name"], e["named"] = tc.function.name, now
                elif tc.function and tc.function.arguments:
                    assert tc.index == len(made) - 1, "a delta for a call that is not the last begun"
                    e["arguments"] += tc.function.arguments; e["deltas"] += 1; e["last"] = now
            finish = ch.finish_reason or finish
    ended = time.perf_counter() - began
    calls = [made[i] for i in sorted(made)]
    return (text, [(e["name"], e["arguments"]) for e in calls], finish, usage), [(e["deltas"], e["named"], e["last"]) for e in calls], ended


cases = [("a free call, a long string", dict(messages=ask(about_long), tools=[save]), 1),
         ("a free call, thinking off", dict(messages=ask(about_long), tools=[save], extra_body={"enable_thinking": False}), 1),
         ("two calls in a reply", dict(messages=ask(about_two), tools=tools), 2),
         ("two asked, one allowed", dict(messages=ask(about_two), tools=tools, parallel_tool_calls=False), 1),
         ("a required call", dict(messages=ask(about_long), tools=[save], tool_choice="required", parallel_tool_calls=False), 1),
         ("a named call", dict(messages=ask("What's the weather in Paris?"), tools=[tools[0], clock],
                               tool_choice={"type": "function", "function": {"name": "get_time"}}, parallel_tool_calls=False), 1)]
for name, request, count in cases:
    request = dict(model=M, max_tokens=900, temperature=0.6, seed=5, **request)
    sent, (got, timing, ended) = whole(**request), as_written(**request)
    print(f"streamed, {name}: {got[2]}, {len(got[1])} call(s), content {got[0]!r}")
    for (call, arguments), (deltas, named, last) in zip(got[1], timing):
        print(f"  {call} in {deltas} deltas, named at {named:.2f} s, last delta {last:.2f} s, end {ended:.2f} s: {arguments[:120]}")
    assert got == sent, f"the stream is not the reply sent whole:\n{got}\n{sent}"
    assert got[2] == "tool_calls" and len(got[1]) == count, got
    for call, arguments in got[1]: jsonschema.validate(json.loads(arguments), note if call == "save_note" else parameters[call])
    # the name is out before the arguments have ended, and a long string comes in many deltas
    assert all(named < last for _, named, last in timing), timing
    if "save_note" == got[1][0][0]: assert timing[0][0] >= 8, f"a long string came in {timing[0][0]} deltas"
print("OK")
