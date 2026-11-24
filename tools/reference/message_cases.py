"""Expected message normalisation and effort coercion for `zig/src/text/messages.zig`, from TensorFold's own code.

Each case runs `tensorfold.server.messages.normalize_messages` then `_normalize_tool_call_arguments`, as
`ChatTemplate.render` does before the template sees the messages, and `request_options.effort_levels` and
`coerce_effort`. Nothing here needs a GPU or the container; TensorFold's sources are enough:

    PYTHONPATH=~/repo/TensorFold/src python3 tools/reference/message_cases.py zig/tests/fixtures/text/message_cases.json

A message case is `{name, late_system, allow_images, messages, expected}` or carries `error` (the
`RequestError`'s text) in place of `expected`. A `loads` case is `{text, dumps}`: `dumps` is
`json.dumps(json.loads(text), ensure_ascii=False)`, or null where `json.loads` raises `ValueError`. Lone
surrogates and integers outside 64 bits are left out: `zig/src/text/pyjson.zig` says what it does with them.
"""
from __future__ import annotations

import json
import sys

from tensorfold.server.errors import RequestError
from tensorfold.server.messages import _normalize_tool_call_arguments, normalize_messages
from tensorfold.server.request_options import EFFORTS, coerce_effort, effort_levels

U = {"role": "user", "content": "hi"}


def call(arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}]}


MESSAGES = {
    "plain": [U],
    "one_system": [{"role": "system", "content": "s"}, U],
    "leading_merged": [{"role": "system", "content": "a"}, {"role": "developer", "content": "b", "name": "d"},
                       {"role": "system", "content": "c"}, U],
    "developer_first_keeps_its_keys": [{"name": "d", "role": "developer", "content": "b"}, U],
    "late_system": [{"role": "system", "content": "s"}, U, {"role": "assistant", "content": "a"},
                    {"role": "system", "content": "later"}, {"role": "developer", "content": "dev"}, U],
    "only_system": [{"role": "system", "content": "s"}],
    "system_after_user_only": [U, {"role": "system", "content": "later"}],
    "text_parts_joined": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
    "empty_parts": [{"role": "user", "content": []}],
    "system_parts_merged": [{"role": "system", "content": [{"type": "text", "text": "a"}]},
                            {"role": "system", "content": None}, U],
    "null_content": [{"role": "assistant", "content": None}, U],
    "missing_content": [{"role": "assistant"}, U],
    "extra_keys_kept": [{"role": "user", "content": None, "name": "n", "x": [1, {"y": 2}]}],
    "falsy_media_allowed": [{"role": "user", "content": "hi", "image": None, "images": [], "audio": ""}],
    "tool_result": [U, call({"a": 1}), {"role": "tool", "tool_call_id": "c1", "content": "42"}],
    "arguments_string": [U, call("{\"a\": 1, \"b\": [true, null, 1.5, \"x\"]}")],
    "arguments_string_spaced": [U, call("  {\"a\": {\"b\": 2}} \n")],
    "arguments_invalid_json": [U, call("{not json")],
    "arguments_string_not_object": [U, call("[1, 2]")],
    "arguments_string_number": [U, call("7")],
    "arguments_empty_string": [U, call("")],
    "arguments_null": [U, call(None)],
    "arguments_list": [U, call([1])],
    "arguments_number": [U, call(3)],
    "arguments_mapping_untouched": [U, call({})],
    "arguments_absent": [U, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]}],
    "tool_calls_empty": [U, {"role": "assistant", "content": "x", "tool_calls": []}],
    "tool_calls_null": [U, {"role": "assistant", "content": "x", "tool_calls": None}],
    "call_not_object": [U, {"role": "assistant", "content": "", "tool_calls": ["x", {"function": "y"}]}],
    "two_calls_one_touched": [U, {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": {"a": 1}}}, {"function": {"arguments": "{\"b\": 2}", "name": "g"}}]}],
    "unicode": [{"role": "system", "content": "你好"}, {"role": "system", "content": "é"},
                {"role": "user", "content": [{"type": "text", "text": "\U0001f600"}]}],
    "image_parts": [{"role": "user", "content": [{"type": "text", "text": "look"},
                                                  {"type": "image_url", "image_url": {"url": "data:x"}}]}],
    "image_only": [{"role": "user", "content": [{"type": "image", "image": "x"}]}],
    "video_part": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "x"}}, {"type": "video"}]}],
    "tool_image": [U, call({}), {"role": "tool", "tool_call_id": "c1",
                                 "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    "image_then_late_system": [{"role": "system", "content": "s"},
                               {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
                               {"role": "system", "content": [{"type": "text", "text": "later"}]}],
    "image_in_first_message_then_system": [{"role": "user", "content": [{"type": "image"}]},
                                           {"role": "system", "content": "s"}],
    "image_part_with_media_key": [{"role": "user", "content": [{"type": "text", "text": "a", "image": "x"},
                                                               {"type": "image_url"}]}],
    # refused
    "assistant_image": [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    "system_image": [{"role": "system", "content": [{"type": "image"}]}, U],
    "image_with_audio_part": [{"role": "user", "content": [{"type": "image"}, {"type": "input_audio"}]}],
    "image_with_bare_part": [{"role": "user", "content": [{"type": "image"}, "x"]}],
    "image_with_bad_text": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": None}]}],
    "image_with_untyped_part": [{"role": "user", "content": [{"type": "image"}, {"text": "x"}]}],
    "image_message_media_key": [{"role": "user", "content": [{"type": "image"}], "images": ["x"]}],
    "empty_list": [],
    "not_a_list": {"role": "user"},
    "message_not_object": ["hi"],
    "bad_role": [{"role": "function", "content": "x"}],
    "no_role": [{"content": "x"}],
    "role_not_string": [{"role": 3, "content": "x"}],
    "media_key": [{"role": "user", "content": "x", "image_url": "http://x"}],
    "image_part": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    "part_not_object": [{"role": "user", "content": ["x"]}],
    "part_with_media": [{"role": "user", "content": [{"type": "text", "text": "a", "image": "x"}]}],
    "part_text_not_string": [{"role": "user", "content": [{"type": "text", "text": 3}]}],
    "part_text_missing": [{"role": "user", "content": [{"type": "text"}]}],
    "content_number": [{"role": "user", "content": 3}],
    "content_object": [{"role": "user", "content": {"a": 1}}],
    "content_bool": [{"role": "user", "content": False}],
}

TEMPLATES = {
    "none": "{{ messages }}",
    "qwen_like": "{% if reasoning_effort == 'low' %}a{% elif reasoning_effort == \"medium\" %}b{% elif reasoning_effort == 'xhigh' %}c{% endif %}",
    "glm_like": "{{ 'low' }}{{ 'high' }}{{ 'max' }}",
    "mixed_quotes": "{{ 'minimal\" }}",
    "unquoted": "low medium 'lower' 'highest' \"max \"",
    "all": "'minimal' 'low' 'medium' 'high' 'xhigh' 'max' 'none'",
}


LOADS = [
    "{}", "[]", " \t\r\n{\"a\" : 1 , \"b\":[ ] }\n", "null", "true", "false", "0", "-0", "7", "-12", "9223372036854775807",
    "0.5", "-0.0", "1e5", "1E5", "1e+5", "1.5e-3", "1e22", "1e16", "123456789.125", "1e999", "-1e999", "1e-999",
    "0.1", "2.5e-5", "1e-7", "NaN", "Infinity", "-Infinity", "[NaN, Infinity, -Infinity]", "{\"a\": NaN}",
    "\"\"", "\"a\\nb\\t\\\"\\\\\\/\\b\\f\\r\"", "\"\\u00e9\\u4f60\"", "\"\\ud83d\\ude00\"", "\"\u4f60\u597d \U0001f600\"",
    "\"\\u0000\\u001f\\u007f\"", "\"\\uABcd\"", "\"<>&'\"",
    "{\"a\": 1, \"b\": 2, \"a\": 3}", "{\"a\": {\"a\": [1, {\"b\": null}]}}", "{\"\": 0}", "[[[[[[1]]]]]]",
    "{\"z\": 1, \"a\": 2}", "[1,2,3]", "[1 ,2 , 3 ]", "{\"a\":\"b\"}",
    # refused
    "", " ", "nul", "True", "None", "nan", "inf", "-NaN", "+1", "+Infinity", "Infinit", "-", "01", "-01", "1.", ".5",
    "1e", "1e+", "1.e5", "0x10", "1_000", "--1", "1 2", "{} {}", "[1,]", "[,1]", "[1 2]", "[", "]", "{", "}",
    "{\"a\"}", "{\"a\":}", "{\"a\": 1,}", "{a: 1}", "{'a': 1}", "{1: 2}", "{\"a\" 1}", "'a'", "\"a", "\"a\nb\"",
    "\"a\tb\"", "\"\\x41\"", "\"\\u12\"", "\"\\u12g4\"", "\"\\", "\"\\u+123\"", "\ufeff{}", "{}x", "// c\n{}",
    "[1] ,", "\u00a0{}", "\x0b{}",
]


def main() -> None:
    cases = []
    for name, messages in MESSAGES.items():
        for late in ("user", "system"):
            for images in (False, True):
                case = {"name": name, "late_system": late, "allow_images": images, "messages": messages}
                try:
                    case["expected"] = _normalize_tool_call_arguments(normalize_messages(
                        json.loads(json.dumps(messages)), late_system=late, allow_images=images))
                except RequestError as e:
                    case["error"] = str(e)
                cases.append(case)
    loads = []
    for text in LOADS:
        try:
            loads.append({"text": text, "dumps": json.dumps(json.loads(text), ensure_ascii=False)})
        except ValueError:
            loads.append({"text": text, "dumps": None})
    levels = [{"name": name, "source": source, "levels": sorted(effort_levels(source))}
              for name, source in TEMPLATES.items()]
    ladders = [[], ["low", "medium", "xhigh"], ["low", "high", "max"], ["medium"], ["minimal", "max"],
               ["low", "high"], ["minimal", "low", "medium", "high", "xhigh", "max"]]
    efforts = [{"effort": effort, "levels": ladder, "heard": coerce_effort(effort, frozenset(ladder))}
               for ladder in ladders for effort in EFFORTS]
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump({"messages": cases, "loads": loads, "levels": levels, "efforts": efforts}, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
