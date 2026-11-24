"""The `openai` package's Responses client against `zspark serve`: a reply whole and streamed, a streamed function
call answered through `previous_response_id`, a conversation continued by it, GET and DELETE, then structured output
(`text.format` whole and streamed, a required and a named function call, a Pydantic model's schema, refusals),
then calls as they stream (spec 027): a free, a required and a named call, two calls in a reply, each streamed
Response the one sent whole, and a call's item open and its arguments under way before the reply ends.

Not a fixture and never runtime: a check of the server by the `openai` package, run by hand. With
`zspark serve DIR --port 8011 --name qwen3.8-flash-next` listening (`ZSPARK_URL` names another address):

    uv run --with openai --with jsonschema --with pydantic python tools/reference/responses_exchange.py

It prints what it received and ends with OK, or fails on the first thing that is not as it should be.
"""

import json, os, sys, time
from typing import Optional
import jsonschema, openai
from openai import OpenAI
from pydantic import BaseModel

base = os.environ.get("ZSPARK_URL", "http://127.0.0.1:8011")
c = OpenAI(base_url=base + "/v1", api_key="none")
M = "qwen3.8-flash-next"
tools = [{"type": "function", "name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
          "units": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}]


def streamed(**request):
    """The stream's event types in order, the text and argument deltas joined, and the Response its last event carries."""
    kinds, text, reasoning, arguments, final, numbers = [], "", "", {}, None, []
    for event in c.responses.create(stream=True, **request):
        kinds.append(event.type)
        numbers.append(event.sequence_number)
        if event.type == "response.output_text.delta": text += event.delta
        elif event.type == "response.reasoning_text.delta": reasoning += event.delta
        elif event.type == "response.function_call_arguments.delta":
            arguments[event.item_id] = arguments.get(event.item_id, "") + event.delta
        elif event.type in ("response.completed", "response.incomplete", "response.failed"): final = event.response
    assert numbers == list(range(len(numbers))), numbers
    assert kinds[:2] == ["response.created", "response.in_progress"], kinds[:2]
    return kinds, text, reasoning, arguments, final


# a reply whole, with instructions
r = c.responses.create(model=M, instructions="Answer in one short sentence.", input="What is the capital of France?",
                       max_output_tokens=400, extra_body={"seed": 1})
print("whole:", r.status, repr(r.output_text), [item.type for item in r.output]); print("usage:", r.usage)
assert r.status == "completed" and "Paris" in r.output_text
assert r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens

# the same streamed: the deltas are the Response's text
kinds, text, reasoning, _, final = streamed(model=M, instructions="Answer in one short sentence.",
                                            input="What is the capital of France?", max_output_tokens=400, extra_body={"seed": 1})
print("streamed:", final.status, repr(text), "| reasoning:", repr(reasoning[:80]))
print("events:", [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]])
assert kinds[-1] == "response.completed" and text == final.output_text and "Paris" in text
assert text == r.output_text, "the streamed reply is not the whole one"

# a streamed function call
kinds, text, reasoning, arguments, first = streamed(
    model=M, input="What's the weather in Paris for the next 3 days, in celsius? Use the tool.", tools=tools,
    max_output_tokens=512, temperature=0.6, extra_body={"seed": 1})
calls = [item for item in first.output if item.type == "function_call"]
print("reasoning:", reasoning[:200].replace("\n", " "))
print("events:", [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]])
print("calls:", [(call.call_id, call.name, call.arguments) for call in calls]); print("usage:", first.usage)
assert first.status == "completed" and len(calls) == 1 and calls[0].name == "get_weather", "no function call"
assert "response.function_call_arguments.done" in kinds
assert arguments[calls[0].id] == calls[0].arguments, "the argument deltas are not the call's arguments"
args = json.loads(calls[0].arguments); assert args["city"].lower().startswith("paris") and args.get("days") == 3, args

# the package's own accumulation of the same stream (`responses.stream`) ends with the same call; not stored
with c.responses.stream(model=M, input="What's the weather in Paris for the next 3 days, in celsius? Use the tool.", tools=tools,
                        max_output_tokens=512, temperature=0.6, store=False, extra_body={"seed": 1}) as helper:
    for event in helper: pass
    built = helper.get_final_response()
print("helper:", built.status, [(item.type, getattr(item, "arguments", None)) for item in built.output])
assert [item.type for item in built.output] == [item.type for item in first.output] and built.output[-1].arguments == calls[0].arguments
try:
    c.responses.retrieve(built.id); sys.exit("a response sent with store: false was stored")
except openai.NotFoundError: pass

# its result, answered through previous_response_id: the call is in the store, not in this request
second = c.responses.create(model=M, previous_response_id=first.id, tools=tools, max_output_tokens=512, temperature=0.6,
                            extra_body={"seed": 1}, input=[{"type": "function_call_output", "call_id": calls[0].call_id,
                            "output": json.dumps({"forecast": [{"day": 1, "high": 21, "sky": "sunny"},
                            {"day": 2, "high": 18, "sky": "rain"}, {"day": 3, "high": 19, "sky": "cloudy"}]})}])
print("second:", second.status, "|", second.output_text); print("usage:", second.usage)
assert second.status == "completed" and "21" in second.output_text and second.previous_response_id == first.id

# a conversation continued: the name is only in the stored turn
one = c.responses.create(model=M, input="My name is Ada and my favourite number is 41. Reply with OK.", max_output_tokens=400)
two = c.responses.create(model=M, previous_response_id=one.id, input="What is my name, and my favourite number plus one?",
                         max_output_tokens=600)
print("continued:", repr(one.output_text), "->", repr(two.output_text)); print("usage:", two.usage)
assert "Ada" in two.output_text and "42" in two.output_text
assert two.usage.input_tokens > one.usage.input_tokens

# GET and DELETE
got = c.responses.retrieve(one.id)
assert got.id == one.id and got.output_text == one.output_text
c.responses.delete(one.id)
try:
    c.responses.retrieve(one.id); sys.exit("a deleted response was found")
except openai.NotFoundError as e: print("deleted:", e.status_code, e.body)
try:
    c.responses.create(model=M, previous_response_id=one.id, input="again"); sys.exit("a deleted response was continued")
except openai.APIStatusError as e: print("gone:", e.status_code, e.body)

# a length cut is `incomplete`
cut = c.responses.create(model=M, input="Count from 1 to 200, one number a line.", max_output_tokens=16)
print("cut:", cut.status, cut.incomplete_details)
assert cut.status == "incomplete" and cut.incomplete_details.reason == "max_output_tokens"


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
clock = {"type": "function", "name": "get_time", "description": "The time in a timezone", "strict": True,
         "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}, "hours24": {"type": "boolean"}},
                        "required": ["timezone", "hours24"], "additionalProperties": False}}
parameters = {t["name"]: t["parameters"] for t in (tools[0], clock)}


def reply(stream, **request):
    """A Response's text, its function calls as (name, arguments) and its status, whole or from a stream."""
    request = dict(model=M, max_output_tokens=600, temperature=0.6, extra_body={"seed": 3}, **request)
    if stream:
        kinds, text, _, arguments, final = streamed(**request)
        assert text == final.output_text, "the deltas are not the Response's text"
        for item in final.output:
            if item.type == "function_call": assert arguments[item.id] == item.arguments
    else: final = c.responses.create(**request)
    return final.output_text, [(i.name, i.arguments) for i in final.output if i.type == "function_call"], final.status


# text.format: a schema, whole and streamed; `strict` either way
for stream in (False, True):
    text, made, status = reply(stream, input=about_city, text={"format": {"type": "json_schema", "name": "city", "strict": not stream, "schema": city}})
    print(f"json_schema (stream {stream}):", status, text)
    jsonschema.validate(json.loads(text), city); assert status == "completed" and not made
text, _, status = reply(False, input="Give me any small JSON object about tea.", text={"format": {"type": "json_object"}})
print("json_object:", status, text); assert isinstance(json.loads(text), dict)

# a Pydantic model's own schema, as it is and through the package's `parse`
for stream in (False, True):
    text, _, status = reply(stream, input=about_person, text={"format": {"type": "json_schema", "name": "person", "schema": Person.model_json_schema()}})
    print(f"pydantic (stream {stream}):", status, text); print("  ->", Person.model_validate_json(text))
parsed = c.responses.parse(model=M, input=about_person, text_format=Person, max_output_tokens=600, temperature=0.6, extra_body={"seed": 4})
print("parse():", parsed.output_parsed); assert isinstance(parsed.output_parsed, Person)

# a required call where the model would not have called, a named function, and the turn after a required call
for stream in (False, True):
    text, made, status = reply(stream, input="Say hello to Ada.", tools=[tools[0], clock], tool_choice="required")
    print(f"required (stream {stream}):", status, made, repr(text))
    assert status == "completed" and made and not text
    for name, arguments in made: jsonschema.validate(json.loads(arguments), parameters[name])
    text, made, status = reply(stream, input="What's the weather in Paris?", tools=[tools[0], clock],
                               tool_choice={"type": "function", "name": "get_time"}, parallel_tool_calls=False)
    print(f"named (stream {stream}):", status, made, repr(text))
    assert len(made) == 1 and made[0][0] == "get_time" and not text
    jsonschema.validate(json.loads(made[0][1]), parameters["get_time"])
held = c.responses.create(model=M, input="What time is it in Tokyo?", tools=[clock], tool_choice="required", parallel_tool_calls=False, max_output_tokens=600)
call = [i for i in held.output if i.type == "function_call"][0]
again = c.responses.create(model=M, previous_response_id=held.id, tools=[clock], tool_choice="required", parallel_tool_calls=False, max_output_tokens=600,
                           input=[{"type": "function_call_output", "call_id": call.call_id, "output": "09:41"}])
calls = [(i.name, i.arguments) for i in again.output if i.type == "function_call"]
print("required again, after the result:", calls)
assert len(calls) == 1; jsonschema.validate(json.loads(calls[0][1]), parameters["get_time"])

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
    text, made, status = reply(stream, input=about_note, tool_choice="required", parallel_tool_calls=False,
                               tools=[{"type": "function", "name": "save_note", "description": "Save a note", "parameters": note}])
    print(f"required, a text of several lines (stream {stream}):", status, made, repr(text))
    assert status == "completed" and len(made) == 1 and made[0][0] == "save_note" and not text
    given = json.loads(made[0][1]); jsonschema.validate(given, note)
    assert isinstance(given["body"], str) and "\n" in given["body"].strip() and isinstance(given["meta"]["lines"], int) and isinstance(given["tags"], list), given

# refusals, in the Responses error shape: what cannot be enforced names its keyword
fussy = {"type": "function", "name": "lookup", "parameters": patterned}
for extra in ({"text": {"format": {"type": "json_schema", "name": "p", "schema": patterned}}},
              {"text": {"format": {"type": "json_schema", "name": "p", "strict": False, "schema": patterned}}, "stream": True},
              {"tools": [fussy], "tool_choice": "required"},
              {"tools": tools, "tool_choice": "required", "text": {"format": {"type": "json_object"}}},
              {"text": {"format": {"type": "json_schema", "name": "p"}}}):
    try:
        r = c.responses.create(model=M, input="hi", max_output_tokens=32, **extra)
        if extra.get("stream"): list(r)
        sys.exit(f"{extra} was not refused")
    except openai.BadRequestError as e:
        print("refused:", e.status_code, e.body)
        assert e.body["type"] == "invalid_request_error" and e.body["message"], e.body
        assert any(word in e.body["message"] for word in ("pattern", "cannot be combined", "needs a schema")), e.body

# -- a call streamed as it is written (spec 027) ------------------------------------------------------------------
save = {"type": "function", "name": "save_note", "description": "Save a note", "parameters": note}
about_two = "What's the weather in Paris and in Tokyo, in celsius? Call the tool once for each city, both calls in this one turn."
about_long = ('Save a note titled "Rain" whose body is a poem of eight lines about rain, each line on its own line, '
              "tagged poem and weather, high priority. Call the tool.")


def shape(response):
    """What a client reads of a Response: its text, its calls, its status and its tokens."""
    return (response.output_text, [(i.name, i.arguments) for i in response.output if i.type == "function_call"], response.status,
            (response.usage.input_tokens + 0, response.usage.output_tokens))


def as_written(**request):
    """A streamed Response as `shape` gives it, built from its events alone, and when they came: for each call the
    number of argument deltas and the seconds to its item, to its last delta and to the stream's end."""
    began = time.perf_counter()
    text, items, order, final = "", {}, [], None
    for event in c.responses.create(stream=True, **request):
        now = time.perf_counter() - began
        if event.type == "response.output_text.delta": text += event.delta
        elif event.type == "response.output_item.added" and event.item.type == "function_call":
            assert event.item.arguments == "" and event.item.name, event.item
            items[event.item.id] = {"name": event.item.name, "arguments": "", "deltas": 0, "named": now, "last": None, "done": None}
            order.append(event.item.id)
        elif event.type == "response.function_call_arguments.delta":
            assert event.item_id == order[-1], "a delta for a call that is not the last begun"
            e = items[event.item_id]; e["arguments"] += event.delta; e["deltas"] += 1; e["last"] = now
        elif event.type == "response.function_call_arguments.done": items[event.item_id]["done"] = event.arguments
        elif event.type in ("response.completed", "response.incomplete", "response.failed"): final = event.response
    ended = time.perf_counter() - began
    calls = [items[i] for i in order]
    for e in calls: assert e["done"] == e["arguments"], "the deltas are not what `done` carries"
    built = (text, [(e["name"], e["arguments"]) for e in calls], final.status, (final.usage.input_tokens, final.usage.output_tokens))
    assert built == shape(final), "the events are not the Response that ends them"
    return built, [(e["deltas"], e["named"], e["last"]) for e in calls], ended


cases = [("a free call, a long string", dict(input=about_long, tools=[save]), 1),
         ("two calls in a reply", dict(input=about_two, tools=tools), 2),
         ("two asked, one allowed", dict(input=about_two, tools=tools, parallel_tool_calls=False), 1),
         ("a required call", dict(input=about_long, tools=[save], tool_choice="required", parallel_tool_calls=False), 1),
         ("a named call", dict(input="What's the weather in Paris?", tools=[tools[0], clock],
                               tool_choice={"type": "function", "name": "get_time"}, parallel_tool_calls=False), 1)]
for name, request, count in cases:
    request = dict(model=M, max_output_tokens=900, temperature=0.6, store=False, extra_body={"seed": 5}, **request)
    sent, (got, timing, ended) = shape(c.responses.create(**request)), as_written(**request)
    print(f"streamed, {name}: {got[2]}, {len(got[1])} call(s), text {got[0]!r}")
    for (call, arguments), (deltas, named, last) in zip(got[1], timing):
        print(f"  {call} in {deltas} deltas, item at {named:.2f} s, last delta {last:.2f} s, end {ended:.2f} s: {arguments[:120]}")
    # the prompt's tokens are compared whole: a second request holds the first's prefix, which is only how they are counted
    assert got == sent, f"the stream is not the Response sent whole:\n{got}\n{sent}"
    assert got[2] == "completed" and len(got[1]) == count, got
    for call, arguments in got[1]: jsonschema.validate(json.loads(arguments), note if call == "save_note" else parameters[call])
    assert all(named < last for _, named, last in timing), timing
    if "save_note" == got[1][0][0]: assert timing[0][0] >= 8, f"a long string came in {timing[0][0]} deltas"
print("OK")
