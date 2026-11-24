"""Expected chat-template renderings for `zig/src/text/template.zig`, from jinja2 as TensorFold configures it.

Every expectation goes through `tensorfold.cuda.chat_template.ChatTemplate`, so the environment is TensorFold's
own: `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])`, its
`tojson` (plain `json.dumps` with `ensure_ascii=False`, which does not escape `<`, `>`, `&` or `'`), and its
`raise_exception` and `strftime_now` globals. Nothing here needs a GPU.

Two files are written:

- `template_cases.json`: contexts rendered through a checkpoint's own `chat_template.jinja`. A `direct` case
  hands the context to the template as it is; a `server` case goes through `ChatTemplate.render` (message
  normalisation, tool-call arguments parsed into mappings, the tokenizer's special tokens) and records the
  context the template was finally given.
- `template_unit_cases.json`: small templates, one construct each, rendered by the same environment. A template
  that jinja2 refuses is recorded with its error.

A case is `{name, template | source, context, rendered}` or carries `error: {type, message}` in place of
`rendered`. `type` is the exception's class name: `TemplateError` is the template's own `raise_exception`.

    docker run --rm --entrypoint python --user "$(id -u):$(id -g)" -e PYTHONPATH=/tf/src \
      -v ~/.cache/zspark/tensorfold-609ca41:/tf:ro -v ~/.cache/huggingface:$HOME/.cache/huggingface:ro \
      -v "$PWD/tools/reference":/ref:ro -v "$PWD/tests/fixtures/text":/out tensorfold-glm53:v0.6.0 \
      /ref/template_cases.py --out-dir /out \
      --model qwen3.8-flash-next=$QWEN_SNAPSHOT --model glm-5.3-flash=$GLM_SNAPSHOT

TensorFold 0.6.5's archive (`tensorfold-609ca41`; its tree has no `cuda/chat_template.py` since 1.0), on the
FP4-Local checkpoints' snapshots (spec 121 WS-00: the Flash Next template is the retired NVFP4's to the byte, GLM's
is not the retired pack's).

The checkpoints' templates are not copied into the repository (the Flash Next checkpoint is under the NVIDIA Open
Model License and the Qwen Community License, not Apache-2.0); the fixture records each one's SHA-256, and the
Zig test reads the template from the checkpoint directory named by `ZSPARK_FLASH_NEXT_FP4_LOCAL` or `ZSPARK_GLM_FP4_LOCAL`.
The unit cases carry their own templates and always run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import jinja2

import tensorfold
from tensorfold.cuda.chat_template import ChatTemplate


def write(path: Path, head: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """One case a line, so a regenerated fixture shows in a diff as the cases that changed."""

    lines = ",\n".join(json.dumps(case, ensure_ascii=False) for case in cases)
    path.write_text(json.dumps(head, ensure_ascii=False, indent=1)[:-2] + ',\n "cases": [\n' + lines + "\n ]\n}\n")


def outcome(render) -> dict[str, Any]:
    try:
        return {"rendered": render()}
    except Exception as exc:  # noqa: BLE001 - the error is the expectation
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}


class Recorder:
    """Stands where `ChatTemplate.template` stands and keeps the keyword arguments of the last render."""

    def __init__(self, template):
        self.template, self.kwargs = template, None

    def render(self, **kwargs):
        self.kwargs = copy.deepcopy(kwargs)
        return self.template.render(**kwargs)


# ---------------------------------------------------------------------------------------------------------------
# Contexts for a checkpoint's template.

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city's name"}},
            "required": ["city"],
        },
    },
}

SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Recherche \"web\" <b>rapide</b> & sûre: 天气, погода, 🌦️ — it's 100% \\ safe\n\ttab \u0001 \u007f   end",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "threshold": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
                "tiny": {"type": "number", "default": 1e-05, "examples": [1e-07, 0.0001, 1e22, 1e16, 1e15,
                                                                         123456.789, -2.5, 1.5e300, 5e-324, 100.0]},
                "filters": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string", "enum": ["a", "b"]},
                            {"type": "object", "properties": {"k": {"type": ["string", "null"]}},
                             "additionalProperties": False},
                        ]
                    },
                },
                "strict": {"type": "boolean", "default": True},
                "nothing": {"default": None, "const": None},
                "empty": {"type": "object", "properties": {}, "required": []},
                "big": {"type": "integer", "default": 9007199254740993, "minimum": -9223372036854775808},
            },
            "required": ["query"],
        },
        "strict": True,
    },
}

FLAT_TOOL = {"name": "ping", "description": "No wrapper.", "parameters": {"type": "object", "properties": {}}}


def call(name: str, arguments: Any, ident: str | None = None) -> dict[str, Any]:
    out = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if ident is not None:
        out["id"] = ident
    return out


def user(content: Any) -> dict[str, Any]:
    return {"role": "user", "content": content}


def direct_cases() -> list[tuple[str, dict[str, Any]]]:
    hi = [user("Hello!")]
    system = [{"role": "system", "content": "You are terse."}, user("Hello!")]
    turns = system + [{"role": "assistant", "content": "Hi."}, user("List three primes."),
                      {"role": "assistant", "content": "2, 3, 5."}, user("And the next?")]
    reasoning = [user("2+2?"),
                 {"role": "assistant", "content": "4", "reasoning_content": "\n  Two and two make four.  \n"},
                 user("3+3?"),
                 {"role": "assistant", "content": "6", "reasoning_content": "Six."},
                 user("4+4?")]
    one_call = [user("Weather in Paris?"),
                {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Paris"}, "c1")]},
                {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 18}"},
                {"role": "assistant", "content": "18 degrees."},
                user("Thanks")]
    rich_args = {"query": "天气 <now> & 'later' \"q\"", "limit": 5, "threshold": 0.25, "deep": {"a": [1, 2.0, None, True],
                 "b": {"c": "é"}}, "list": ["x", 1, False], "flag": True, "none": None, "text": "  spaced\n", "empty": ""}
    several_calls = [user("Do both."),
                     {"role": "assistant", "content": "On it.", "reasoning_content": "Two tools.",
                      "tool_calls": [call("get_weather", {"city": "Paris"}, "c1"), call("search", rich_args, "c2"),
                                     call("ping", {}, "c3")]},
                     {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                     {"role": "tool", "tool_call_id": "c2", "content": "  3 results \n"},
                     {"role": "tool", "tool_call_id": "c3", "content": ""}]
    cases: list[tuple[str, dict[str, Any]]] = [
        ("user_only", {"messages": hi, "add_generation_prompt": True}),
        ("user_only_no_generation_prompt", {"messages": hi, "add_generation_prompt": False}),
        ("user_only_generation_prompt_unset", {"messages": hi}),
        ("system_user", {"messages": system, "add_generation_prompt": True}),
        ("multi_turn", {"messages": turns, "add_generation_prompt": True}),
        ("multi_turn_ends_with_assistant", {"messages": turns[:-1], "add_generation_prompt": False}),
        ("thinking_true", {"messages": system, "add_generation_prompt": True, "enable_thinking": True}),
        ("thinking_false", {"messages": system, "add_generation_prompt": True, "enable_thinking": False}),
        ("thinking_false_no_system", {"messages": hi, "add_generation_prompt": True, "enable_thinking": False}),
        ("thinking_false_no_generation_prompt", {"messages": hi, "add_generation_prompt": False,
                                                 "enable_thinking": False}),
        ("thinking_null", {"messages": hi, "add_generation_prompt": True, "enable_thinking": None}),
        ("thinking_one", {"messages": hi, "add_generation_prompt": True, "enable_thinking": 1}),
        ("thinking_zero", {"messages": hi, "add_generation_prompt": True, "enable_thinking": 0}),
        ("special_tokens", {"messages": hi, "add_generation_prompt": True, "bos_token": None,
                            "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "unk_token": None}),
    ]
    for effort in ("low", "medium", "high", "xhigh", "max", "turbo", "", None, 3, ["low"]):
        cases.append((f"effort_{json.dumps(effort)}", {"messages": system, "add_generation_prompt": True,
                                                       "reasoning_effort": effort}))
    cases += [
        ("effort_low_thinking_false", {"messages": system, "add_generation_prompt": True, "enable_thinking": False,
                                       "reasoning_effort": "low"}),
        ("effort_low_no_system", {"messages": hi, "add_generation_prompt": True, "reasoning_effort": "low"}),
        ("effort_medium_no_system", {"messages": hi, "add_generation_prompt": True, "reasoning_effort": "medium"}),
        ("effort_medium_tools", {"messages": hi, "tools": [WEATHER], "add_generation_prompt": True,
                                 "reasoning_effort": "medium"}),
        ("reasoning_content", {"messages": reasoning, "add_generation_prompt": True}),
        ("reasoning_content_preserve_false", {"messages": reasoning, "add_generation_prompt": True,
                                              "preserve_thinking": False}),
        ("reasoning_content_preserve_true", {"messages": reasoning, "add_generation_prompt": True,
                                             "preserve_thinking": True}),
        ("reasoning_content_clear_false", {"messages": reasoning, "add_generation_prompt": True,
                                           "clear_thinking": False}),
        ("reasoning_content_not_a_string", {"messages": [user("q"), {"role": "assistant", "content": "a",
                                                                    "reasoning_content": ["x"]}, user("r")],
                                            "add_generation_prompt": True}),
        ("think_tags_in_content", {"messages": [user("q"), {"role": "assistant",
                                                             "content": "<think>\nhmm\n</think>\n\nanswer  "},
                                                user("r")], "add_generation_prompt": True}),
        ("one_tool", {"messages": hi, "tools": [WEATHER], "add_generation_prompt": True}),
        ("one_tool_system", {"messages": system, "tools": [WEATHER], "add_generation_prompt": True}),
        ("one_tool_thinking_false", {"messages": system, "tools": [WEATHER], "add_generation_prompt": True,
                                     "enable_thinking": False}),
        ("several_tools", {"messages": system, "tools": [WEATHER, SEARCH, FLAT_TOOL], "add_generation_prompt": True}),
        ("tools_null", {"messages": hi, "tools": None, "add_generation_prompt": True}),
        ("tools_empty", {"messages": hi, "tools": [], "add_generation_prompt": True}),
        ("tools_mapping", {"messages": hi, "tools": WEATHER, "add_generation_prompt": True}),
        ("tools_string", {"messages": hi, "tools": "get_weather", "add_generation_prompt": True}),
        ("tools_system_whitespace_only", {"messages": [{"role": "system", "content": " \n "}, user("Hi")],
                                          "tools": [WEATHER], "add_generation_prompt": True}),
        ("tool_call_one", {"messages": one_call, "tools": [WEATHER], "add_generation_prompt": True}),
        ("tool_call_awaiting_reply", {"messages": one_call[:3], "tools": [WEATHER], "add_generation_prompt": True}),
        ("tool_call_no_generation_prompt", {"messages": one_call[:2], "tools": [WEATHER],
                                            "add_generation_prompt": False}),
        ("tool_calls_several", {"messages": several_calls, "tools": [WEATHER, SEARCH, FLAT_TOOL],
                                "add_generation_prompt": True}),
        ("tool_calls_several_preserve_false", {"messages": several_calls + [{"role": "assistant", "content": "Done."},
                                                                             user("More.")],
                                               "tools": [WEATHER, SEARCH], "add_generation_prompt": True,
                                               "preserve_thinking": False}),
        ("tool_results_without_ids", {"messages": [user("Go"),
                                                   {"role": "assistant", "content": None,
                                                    "tool_calls": [call("a", {"x": 1}), call("b", {"y": "z"})]},
                                                   {"role": "tool", "content": "first"},
                                                   {"role": "tool", "content": "second"},
                                                   user("Next")],
                                      "add_generation_prompt": True}),
        ("tool_results_out_of_order", {"messages": [user("Go"),
                                                    {"role": "assistant", "content": "",
                                                     "tool_calls": [call("a", {}, "1"), call("b", {}, "2")]},
                                                    {"role": "tool", "tool_call_id": "2", "content": "second"},
                                                    {"role": "tool", "tool_call_id": "1", "content": "first"}],
                                       "add_generation_prompt": True}),
        ("tool_call_arguments_json_string", {"messages": [user("Go"),
                                                          {"role": "assistant", "content": "",
                                                           "tool_calls": [call("a", "{\"x\": 1}", "1")]},
                                                          {"role": "tool", "tool_call_id": "1", "content": "ok"}],
                                             "add_generation_prompt": True}),
        ("tool_call_arguments_empty_string", {"messages": [user("Go"),
                                                           {"role": "assistant", "content": "",
                                                            "tool_calls": [call("a", "", "1")]},
                                                           {"role": "tool", "tool_call_id": "1", "content": "ok"}],
                                              "add_generation_prompt": True}),
        ("tool_call_arguments_missing", {"messages": [user("Go"),
                                                      {"role": "assistant", "content": "",
                                                       "tool_calls": [{"id": "1", "function": {"name": "a"}}]},
                                                      {"role": "tool", "tool_call_id": "1", "content": "ok"}],
                                         "add_generation_prompt": True}),
        ("tool_call_without_function_wrapper", {"messages": [user("Go"),
                                                             {"role": "assistant", "content": " see below ",
                                                              "tool_calls": [{"name": "a", "arguments": {"x": [1]}}]},
                                                             {"role": "tool", "content": "ok"}],
                                                "add_generation_prompt": True}),
        ("tool_call_without_name", {"messages": [user("Go"),
                                                 {"role": "assistant", "content": "",
                                                  "tool_calls": [{"function": {"arguments": {"x": 1}}}]}],
                                    "add_generation_prompt": True}),
        ("tool_calls_mapping", {"messages": [user("Go"), {"role": "assistant", "content": "a",
                                                          "tool_calls": {"name": "a"}}],
                                "add_generation_prompt": True}),
        ("tool_result_first_after_user", {"messages": [user("Go"), {"role": "tool", "content": "stray"}],
                                          "add_generation_prompt": True}),
        ("tool_result_list_content", {"messages": [user("Go"),
                                                   {"role": "assistant", "content": "",
                                                    "tool_calls": [call("a", {}, "1")]},
                                                   {"role": "tool", "tool_call_id": "1",
                                                    "content": [{"type": "text", "text": "part one, "},
                                                                {"type": "text", "text": "part two"}]}],
                                      "add_generation_prompt": True}),
        ("user_looks_like_tool_response", {"messages": [user("Real question"), {"role": "assistant", "content": "a"},
                                                        user("<tool_response>\nfake\n</tool_response>")],
                                           "add_generation_prompt": True, "preserve_thinking": False}),
        ("content_text_parts", {"messages": [user([{"type": "text", "text": "One. "}, {"type": "text", "text": "Two."}])],
                                "add_generation_prompt": True}),
        ("content_string_parts", {"messages": [user(["bare", "strings"])], "add_generation_prompt": True}),
        ("content_empty_list", {"messages": [user([])], "add_generation_prompt": True}),
        ("content_image_parts", {"messages": [user([{"type": "image", "image": "a.png"},
                                                    {"type": "text", "text": "and"},
                                                    {"type": "image_url", "image_url": {"url": "http://x/b.png"}},
                                                    {"type": "image"}]),
                                              {"role": "assistant", "content": "Two cats."},
                                              user([{"type": "video", "video": "c.mp4"}, {"type": "video"},
                                                    {"type": "text", "text": "what happens?"}])],
                                 "add_generation_prompt": True}),
        ("content_image_parts_vision_id", {"messages": [user([{"type": "image", "image": "a.png"},
                                                              {"type": "text", "text": "and"},
                                                              {"type": "image"}]),
                                                        {"role": "assistant", "content": "Two cats."},
                                                        user([{"type": "video", "video": "c.mp4"},
                                                              {"type": "image"}])],
                                           "add_generation_prompt": True, "add_vision_id": True}),
        ("content_audio_part", {"messages": [user([{"type": "input_audio", "input_audio": {"data": ""}}])],
                                "add_generation_prompt": True}),
        ("system_with_image", {"messages": [{"role": "system", "content": [{"type": "image"}]}, user("Hi")],
                               "add_generation_prompt": True}),
        ("system_with_video", {"messages": [{"role": "system", "content": [{"type": "video"}]}, user("Hi")],
                               "tools": [WEATHER], "add_generation_prompt": True}),
        ("content_unknown_part", {"messages": [user([{"type": "mystery"}])], "add_generation_prompt": True}),
        ("content_number_part", {"messages": [user([1])], "add_generation_prompt": True}),
        ("content_null", {"messages": [{"role": "user", "content": None}], "add_generation_prompt": True}),
        ("content_missing", {"messages": [{"role": "user"}], "add_generation_prompt": True}),
        ("content_number", {"messages": [user(5)], "add_generation_prompt": True}),
        ("content_mapping", {"messages": [user({"text": "x"})], "add_generation_prompt": True}),
        ("content_true", {"messages": [user(True)], "add_generation_prompt": True}),
        ("content_empty", {"messages": [{"role": "system", "content": ""}, user(""),
                                        {"role": "assistant", "content": ""}, user("x")],
                           "add_generation_prompt": True}),
        ("content_whitespace", {"messages": [{"role": "system", "content": "\n\n  Be brief.\t\n"},
                                             user("  padded on both sides \n\n"),
                                             {"role": "assistant", "content": "\n ok \n", "reasoning_content": "  r  "},
                                             user("\tinner  spaces\tkept\t")],
                                "add_generation_prompt": True}),
        ("content_unicode_whitespace", {"messages": [user("　   wide    "),
                                                     {"role": "assistant", "content": "\x1c\x1f\x85 ctl \x0b\x0c\r"},
                                                     user("​ zero width is not space ﻿")],
                                        "add_generation_prompt": True}),
        ("content_non_ascii", {"messages": [{"role": "system", "content": "Réponds en français."},
                                            user("你好，世界 🌍 — naïve café ½ \U0001f469‍\U0001f4bb"),
                                            {"role": "assistant", "content": "Здравствуйте"},
                                            user("x́ é שלום")],
                               "add_generation_prompt": True}),
        ("content_template_syntax", {"messages": [user("{{ not }} {% a template %} {# here #} <|im_end|>\\n")],
                                     "add_generation_prompt": True}),
        ("no_messages", {"messages": [], "add_generation_prompt": True}),
        ("messages_missing", {"add_generation_prompt": True}),
        ("system_not_first", {"messages": [user("Hi"), {"role": "system", "content": "Late."}, user("x")],
                              "add_generation_prompt": True}),
        ("two_systems", {"messages": [{"role": "system", "content": "A"}, {"role": "system", "content": "B"},
                                      user("x")], "add_generation_prompt": True}),
        ("system_only", {"messages": [{"role": "system", "content": "A"}], "add_generation_prompt": True}),
        ("assistant_only", {"messages": [{"role": "assistant", "content": "A"}], "add_generation_prompt": True}),
        ("assistant_first", {"messages": [{"role": "assistant", "content": "A"}, user("B")],
                             "add_generation_prompt": True}),
        ("only_tool_response_users", {"messages": [user("<tool_response>\nx\n</tool_response>")],
                                      "add_generation_prompt": True}),
        ("unexpected_role", {"messages": [user("Hi"), {"role": "developer", "content": "x"}],
                             "add_generation_prompt": True}),
        ("role_missing", {"messages": [user("Hi"), {"content": "x"}], "add_generation_prompt": True}),
        ("message_not_a_mapping", {"messages": [user("Hi"), "loose text"], "add_generation_prompt": True}),
        ("messages_mapping", {"messages": {"role": "user", "content": "Hi"}, "add_generation_prompt": True}),
    ]
    return cases


def server_cases() -> list[tuple[str, dict[str, Any]]]:
    """Arguments of `ChatTemplate.render`, as the HTTP server calls it."""

    return [
        ("user_only", {"messages": [user("Hello!")], "tools": None, "enable_thinking": True}),
        ("thinking_off", {"messages": [user("Hello!")], "tools": None, "enable_thinking": False}),
        ("effort_low", {"messages": [user("Hello!")], "tools": None, "enable_thinking": True,
                        "extra": {"reasoning_effort": "low"}}),
        ("template_kwargs", {"messages": [user("Hello!"), {"role": "assistant", "content": "Hi",
                                                           "reasoning_content": "greet"}, user("Again")],
                             "tools": None, "enable_thinking": True,
                             "extra": {"preserve_thinking": False, "clear_thinking": False, "custom": {"a": [1, 2]}}}),
        ("developer_and_late_system", {"messages": [{"role": "developer", "content": "Rule one."},
                                                    {"role": "system", "content": "Rule two."},
                                                    user("Hello!"), {"role": "assistant", "content": "Hi"},
                                                    {"role": "system", "content": "A late instruction."},
                                                    user([{"type": "text", "text": "Go "}, {"type": "text", "text": "on"}])],
                                       "tools": None, "enable_thinking": True}),
        ("tools_and_string_arguments", {"messages": [user("Weather in Paris?"),
                                                     {"role": "assistant", "content": None,
                                                      "tool_calls": [call("get_weather", "{\"city\": \"Paris\", \"n\": 1.0}", "c1"),
                                                                     call("search", "not json", "c2"),
                                                                     call("search", "[1, 2]", "c3")]},
                                                     {"role": "tool", "tool_call_id": "c1", "content": "18"},
                                                     {"role": "tool", "tool_call_id": "c2", "content": "none"},
                                                     {"role": "tool", "tool_call_id": "c3", "content": "none"}],
                                        "tools": [WEATHER, SEARCH], "enable_thinking": False}),
        ("empty_tools", {"messages": [user("Hello!")], "tools": [], "enable_thinking": True}),
    ]


# ---------------------------------------------------------------------------------------------------------------
# Small templates, one construct each.

NESTED = {"a": {"b": [10, 20, {"c": "deep"}]}, "s": "héllo wörld", "n": 3, "f": 2.5, "t": True, "z": None,
          "l": [1, 2, 3, 4, 5], "e": [], "d": {"k1": "v1", "k2": 2}, "items": {"items": "shadow"}}


def unit_cases() -> list[tuple[str, str, dict[str, Any]]]:
    u: list[tuple[str, str, dict[str, Any]]] = []

    def add(name: str, source: str, context: dict[str, Any] | None = None) -> None:
        u.append((name, source, context if context is not None else {}))

    # Whitespace control.
    add("ws_plain_text", "a\n  b\n\nc\n")
    add("ws_trailing_newline_once", "a\n\n")
    add("ws_crlf", "a\r\nb\r{% if true %}\r\nc\r\n{% endif %}\r\nd")
    add("ws_trim_blocks", "a\n{% if true %}\nb\n{% endif %}\nc")
    add("ws_trim_blocks_only_one_newline", "{% if true %}\n\nb{% endif %}\n\n\nc")
    add("ws_trim_blocks_not_after_variable", "{{ 'a' }}\n{{ 'b' }}\n  {{ 'c' }}  \nd")
    add("ws_lstrip_blocks", "a\n    {% if true %}\n    b\n  \t{% endif %}\nc")
    add("ws_lstrip_not_mid_line", "a  {% if true %}  b  {% endif %}  c")
    add("ws_lstrip_second_tag_on_line", "  {% if true %}  {% if true %}x{% endif %}  {% endif %}  y")
    add("ws_lstrip_first_line", "   {% if true %}x{% endif %}")
    add("ws_lstrip_after_trimmed_newline", "{% if true %}\n   {% if true %}x{% endif %}\n{% endif %}")
    add("ws_lstrip_after_minus_end", "{% if true -%}\n   {% if true %}x{% endif %}\n{% endif %}")
    add("ws_lstrip_after_variable", "{{ 'a' }}   {% if true %}x{% endif %}")
    add("ws_lstrip_after_variable_newline", "{{ 'a' }}\n   {% if true %}x{% endif %}")
    add("ws_lstrip_not_for_variable", "a\n   {{ 'b' }}\n")
    add("ws_plus_disables_lstrip", "a\n   {%+ if true %}x{% endif %}")
    add("ws_plus_disables_trim", "{% if true +%}\nx{% endif +%}\ny")
    add("ws_minus_left", "a \n\t {%- if true %} b {%- endif %} c")
    add("ws_minus_right", "a {% if true -%} \n\n b {% endif -%} \n c")
    add("ws_minus_variable", "a \n {{- 'b' -}} \n c \n {{- 'd' }} \n e")
    add("ws_minus_strips_unicode_space", "a 　{%- if true -%}  b{% endif %}")
    add("ws_comment", "a {# gone #} b\n  {# line #}\nc {#- tight -#} d")
    add("ws_comment_with_tags_inside", "a{# {% if %} {{ x }} #}b")
    add("ws_tag_spans_lines", "{% if\n  true\n%}x{%\nendif\n%}{{\n 'y'\n}}")
    add("ws_end_inside_string", "{{ '%}' ~ '}}' ~ \"{{\" }}{% set a = '%}' %}{{ a }}")
    add("ws_braces_in_dict", "{{ {'a': {'b': 1}}|tojson }}{{ {'a':{}}|tojson }}")
    add("ws_minus_then_expression", "{{ 3 -1 }}{{ 3 - 1 }}{{- 3 -}}  x")

    add("ws_lone_braces_in_text", "a { b } c {d} { { e } } }} %} #} {")
    add("ws_json_text_between_tags", '{"k": "{{ v }}", "l": [{% for x in l %}{{ x }}{{ ", " if not loop.last }}{% endfor %}]}', {"v": 1, "l": [1, 2]})
    add("ws_sign_glued_to_expression", "{{-1}}|{{- -1}}|{% set x = -1 %}{{x}}|{{ 1 --1 }}|{{+1}}")
    add("ws_tabs_and_blank_lines", "a\n\t{% if true %}\n\t\tb\n\n\t{% endif %}\n\n\tc\n")
    add("ws_comment_at_line_start", "a\n   {#- x #}\nb\n   {# y -#}\n   c{#+ z #}\nd")
    add("ws_trailing_commas", "{{ [1, 2,]|tojson }}{{ {'a': 1,}|tojson }}{{ (1, 2,)|tojson }}{% macro m(a, b=2) %}{{ a }}{{ b }}{% endmacro %}{{ m(1,) }}{{ m(1, b=3,) }}{{ 'x'|default('y',) }}")
    add("syntax_trailing_comma_in_macro_signature", "{% macro m(a, b=2,) %}{% endmacro %}")

    # Literals and output.
    add("lit_strings", r"""{{ 'single' }}{{ "double" }}{{ 'a\nb\tc\\d\'e"f' }}{{ "a\"b'c" }}{{ 'x' 'y' "z" }}""")
    add("lit_string_escapes", r"""{{ '\x41é中' }}{{ 'é中🌍' }}""")
    add("lit_numbers", "{{ 0 }} {{ 42 }} {{ -7 }} {{ 1.5 }} {{ 1.0 }} {{ 1e3 }} {{ 2.5e-7 }} {{ 1e16 }} {{ 1e15 }} {{ 0.0001 }} {{ 0.00001 }}")
    add("lit_constants", "{{ true }} {{ false }} {{ none }} {{ True }} {{ False }} {{ None }}")
    add("out_values", "{{ n }}|{{ f }}|{{ t }}|{{ z }}|{{ s }}|{{ missing }}|{{ a.b[0] }}", NESTED)
    add("out_floats", "{% for x in xs %}{{ x }} {% endfor %}",
        {"xs": [0.1, 1.0, 100.0, 1e22, 1e16, 1e15, 123456789012345680.0, 1e-05, 0.0001, 1.5e300, 5e-324, -0.5,
                3.141592653589793, 2.5e-07, 1e100, 9007199254740993, -1]})
    add("out_containers_as_python_repr", "{{ l }} {{ d }} {{ a }} {{ (1, 'a') }} {{ (1,) }} {{ () }} {{ [] }} {{ {} }} {{ [1.0, true, none, 'x'] }} {{ v }} {{ 'e: ' ~ l ~ d }}",
        dict(NESTED, v=["plain", "it's", 'say "hi"', "both ' and \"", "back\\slash", "line\nfeed\ttab\r", "ctl \x01\x1f\x7f", ""]))
    add("out_container_items_filter_pairs", "{% for pair in d|items %}{{ pair }}{% endfor %} {% for pair in d.items() %}{{ pair }}{% endfor %}", NESTED)

    # Precedence.
    add("prec_arithmetic", "{{ 1 + 2 * 3 }} {{ (1 + 2) * 3 }} {{ 7 - 2 - 1 }} {{ 2 * 3 % 4 }} {{ 7 // 2 * 2 }} {{ -2 + 5 }} {{ 10 - -3 }}")
    add("prec_division", "{{ 7 / 2 }} {{ 6 / 3 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }} {{ 7 % -3 }} {{ 7.5 // 2 }} {{ 7.5 % 2 }}")
    add("prec_concat_binds_tighter_than_plus", "{{ 'a' ~ 1 + 2 }}")
    add("prec_concat", "{{ 'a' ~ 1 ~ 2.5 ~ true ~ none ~ missing ~ 'z' }} {{ 1 ~ 2 * 3 }}")
    add("prec_filter_binds_tighter", "{{ l|length - 1 }} {{ 10 - l|length }} {{ -1|string ~ 'x' }} {{ 'a' ~ ' b '|trim ~ 'c' }}", NESTED)
    add("prec_not_and_or", "{{ not false and false }} {{ not (false and false) }} {{ true or false and false }} {{ not true or true }}")
    add("prec_compare", "{{ 1 + 1 == 2 }} {{ 1 < 2 == true }} {{ 1 < 2 < 3 }} {{ 3 > 2 > 2 }} {{ 1 == 1 != 2 }} {{ not 1 == 2 }}")
    add("prec_in", "{{ 'a' in 'cat' }} {{ 'a' not in 'cat' }} {{ not 'a' in 'dog' }} {{ 2 in l and 9 not in l }} {{ 1 + 1 in l }}", NESTED)
    add("prec_test", "{{ n is defined and missing is not defined }} {{ not n is string }} {{ n is number == true }} {{ s is string ~ '!' }}", NESTED)
    add("prec_conditional", "{{ 'a' if t else 'b' }} {{ 'a' if z else 'b' }} {{ 'a' ~ 'b' if false else 'c' ~ 'd' }} {{ ('x' if false) ~ '|' }} {{ 1 if false else 2 if false else 3 }}", NESTED)
    add("prec_conditional_with_filters", "{{ s|string if s is string else s|tojson|safe }} {{ n|string if n is string else n|tojson|safe }}", NESTED)
    add("prec_and_or_values", "{{ 0 or 'x' }} {{ 'a' and 'b' }} {{ '' and 'b' }}|{{ none or 0 }} {{ missing or 'd' }} {{ l and l[0] }}", NESTED)

    add("prec_test_then_keyword", "{{ n is defined and s is string or z is none }} {{ 'a' if n is defined else 'b' }} {{ n is not none and z is none }} {{ 1.0 in [1] }} {{ 1 not in [2] not in [true] }}", NESTED)

    # Comparison and truthiness.
    add("cmp_equality", "{{ 1 == 1.0 }} {{ true == 1 }} {{ false == 0 }} {{ '1' == 1 }} {{ none == none }} {{ none == false }} {{ [1, 2] == [1, 2] }} {{ (1, 2) == [1, 2] }} {{ (1, 2) == (1, 2) }} {{ {'a': 1} == {'a': 1.0} }} {{ 'a' != 'b' }} {{ missing == other }} {{ missing == none }} {{ missing == '' }}")
    add("cmp_order", "{{ 'a' < 'b' }} {{ 'B' < 'a' }} {{ 'é' > 'z' }} {{ 2 >= 2 }} {{ 1.5 <= 1 }} {{ true > 0 }} {{ 'abc' < 'abd' }} {{ 'ab' < 'abc' }}")
    add("cmp_order_mixed_types", "{{ 'a' < 1 }}")
    add("truthiness", "{% for v in [0, 1, 0.0, 0.5, '', ' ', [], [0], {}, {'a': 0}, none, true, false, missing] %}{{ 'T' if v else 'F' }}{% endfor %}")
    add("in_mapping_and_string", "{{ 'k1' in d }} {{ 'v1' in d }} {{ 1 in d }} {{ 'ö' in s }} {{ '' in s }} {{ 'x' in missing }} {{ none in l }}", NESTED)
    add("in_number_raises", "{{ 'a' in n }}", NESTED)
    add("in_string_needs_string", "{{ 1 in s }}", NESTED)

    # Undefined.
    add("undef_output_and_tests", "[{{ missing }}] {{ missing is defined }} {{ missing is undefined }} {{ missing is none }} {{ missing is string }} {{ missing is iterable }} {{ missing is mapping }}")
    add("undef_attribute_of_mapping", "[{{ d.nope }}] [{{ d['nope'] }}] {{ d.nope is defined }} {{ d.k1 is defined }}", NESTED)
    add("undef_attribute_of_undefined_raises", "{{ missing.attr }}")
    add("undef_item_of_undefined_raises", "{{ missing[0] }}")
    add("undef_attribute_chain_raises", "{{ d.nope.deeper }}", NESTED)
    add("undef_attribute_test_on_missing_parent_raises", "{{ missing.attr is defined }}")
    add("undef_plus_string_raises", "{{ 'a' + missing }}")
    add("undef_string_plus_raises", "{{ missing + 'a' }}")
    add("undef_loop_and_length", "{% for x in missing %}x{% endfor %}[{{ missing|length }}]")
    add("undef_filters", "[{{ missing|default('d') }}] [{{ missing|trim }}] [{{ missing|string }}] [{% for k, v in missing|items %}x{% endfor %}] [{{ missing|default }}] [{{ z|default('d') }}] [{{ z|default('d', true) }}] [{{ ''|d('e', true) }}] [{{ 0|default(5) }}]", NESTED)
    add("undef_tojson_raises", "{{ missing|tojson }}")
    add("undef_call_raises", "{{ missing() }}")
    add("undef_equality", "{{ missing == 1 }} {{ missing != 1 }} {{ d.nope == 'x' }} {{ d.nope != 'x' }}", NESTED)
    add("undef_compare_raises", "{{ missing < 1 }}")
    add("attr_of_scalars", "[{{ s.nope }}] [{{ n.nope }}] [{{ z.nope }}] [{{ l.nope }}] [{{ t.nope }}] [{{ s['nope'] }}] [{{ l['nope'] }}] [{{ n[0] }}] [{{ z[0] }}]", NESTED)
    add("attr_item_fallbacks", "{{ d['k1'] }} {{ d.k1 }} {{ a['b'][1] }} {{ a.b.1 }} {{ a.b.2.c }} {{ a.b[2]['c'] }} [{{ a.b.9 }}] [{{ a.b[9] }}] [{{ a.b[-1].c }}] [{{ a.b[-9] }}] [{{ d[0] }}] [{{ s[0] }}{{ s[1] }}{{ s[-1] }}] [{{ s[99] }}]", NESTED)

    # Slices and indexing.
    add("slice_lists", "{{ l[::-1]|tojson }} {{ l[1:3]|tojson }} {{ l[:2]|tojson }} {{ l[-2:]|tojson }} {{ l[::2]|tojson }} {{ l[4:1:-1]|tojson }} {{ l[10:]|tojson }} {{ l[:-10]|tojson }} {{ l[-10:2]|tojson }} {{ l[::-2]|tojson }} {{ l[1:n]|tojson }}", NESTED)
    add("slice_strings", "{{ s[:5] }}|{{ s[1:2] }}|{{ s[::-1] }}|{{ s[-5:] }}|{{ s[::3] }}", NESTED)
    add("slice_step_zero_raises", "{{ l[::0] }}", NESTED)

    # set, scoping and namespace.
    add("set_basic", "{% set x = 1 %}{% set y = x + 1 %}{% set x = 'new' %}{{ x }}{{ y }}")
    add("set_in_if_is_visible", "{% if true %}{% set x = 'in' %}{% endif %}{{ x }}")
    add("set_in_for_does_not_leak", "{% set x = 'out' %}{% set count = 0 %}{% for i in l %}{% set x = i %}{% set count = count + 1 %}[{{ x }}{{ count }}]{% endfor %}{{ x }}{{ count }}", NESTED)
    add("set_in_for_read_before_set", "{% set x = 'out' %}{% for i in l %}{{ x }}{% set x = i %}{{ x }} {% endfor %}", NESTED)
    add("set_in_for_fresh_each_iteration", "{% for i in l %}{{ seen is defined }}{% set seen = i %}{% endfor %}", NESTED)
    add("set_in_if_in_for", "{% set x = 'out' %}{% for i in l %}{% if i == 2 %}{% set x = 'two' %}{% endif %}{{ x }} {% endfor %}{{ x }}", NESTED)
    add("set_loop_variable", "{% for i in l %}{% if i > 2 %}{% set i = i * 10 %}{% endif %}{{ i }} {% endfor %}", NESTED)
    add("for_variable_does_not_leak", "{% set i = 'outer' %}{% for i in l %}{% endfor %}{{ i }}[{{ loop }}]", NESTED)
    add("namespace_counter", "{% set ns = namespace(count=0, last=none) %}{% for i in l %}{% set ns.count = ns.count + i %}{% set ns.last = i %}{% endfor %}{{ ns.count }} {{ ns.last }} [{{ ns.nope }}] {{ ns.nope is defined }}", NESTED)
    add("namespace_new_attribute", "{% set ns = namespace() %}{% set ns.a = 1 %}{{ ns.a }}{{ ns['a'] }}")
    add("namespace_in_nested_loops", "{% set ns = namespace(n=0) %}{% for i in l %}{% for j in l %}{% set ns.n = ns.n + 1 %}{% endfor %}{% endfor %}{{ ns.n }}", NESTED)
    add("set_attribute_on_mapping_raises", "{% set d.k1 = 2 %}", NESTED)
    add("set_attribute_on_undefined_raises", "{% set nope.k1 = 2 %}")

    # for and loop.
    add("loop_variables", "{% for x in l %}{{ loop.index }}{{ loop.index0 }}{{ loop.revindex }}{{ loop.revindex0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }} {% endfor %}", NESTED)
    add("loop_previtem_nextitem", "{% for x in l %}[{{ loop.previtem }}<{{ x }}>{{ loop.nextitem }}]{{ loop.previtem is defined }}{{ loop.nextitem is defined }}{% endfor %}", NESTED)
    add("loop_empty", "a{% for x in e %}never{% endfor %}b", NESTED)
    add("loop_over_mapping_keys", "{% for k in d %}{{ k }}={{ d[k] }};{% endfor %}", NESTED)
    add("loop_over_string", "{% for c in s %}{{ c }}.{% endfor %}", NESTED)
    add("loop_unpack_items_filter", "{% for k, v in d|items %}{{ k }}={{ v }}{{ ',' if not loop.last }}{% endfor %}", NESTED)
    add("loop_unpack_items_method", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}{% for k in d.keys() %}{{ k }}{% endfor %}{% for v in d.values() %}{{ v }}{% endfor %}", NESTED)
    add("loop_unpack_pairs", "{% for a, b in [[1, 2], [3, 4]] %}{{ a + b }} {% endfor %}{% for a, b, c in [(1, 2, 3)] %}{{ a ~ b ~ c }}{% endfor %}")
    add("loop_unpack_mismatch_raises", "{% for a, b in [[1, 2, 3]] %}{% endfor %}")
    add("loop_nested", "{% for x in [1, 2] %}{% for y in ['a', 'b'] %}{{ x }}{{ y }}{{ loop.index }} {% endfor %}{{ loop.index }}|{% endfor %}")
    add("loop_over_number_raises", "{% for x in n %}{% endfor %}", NESTED)
    add("loop_over_none_raises", "{% for x in z %}{% endfor %}", NESTED)
    add("loop_range", "{% for i in range(3) %}{{ i }}{% endfor %} {% for i in range(2, 5) %}{{ i }}{% endfor %} {% for i in range(5, 0, -2) %}{{ i }}{% endfor %} {% for i in range(3, 3) %}x{% endfor %}{{ range(4)|length }}")
    add("loop_break_continue", "{% for i in l %}{% if i == 2 %}{% continue %}{% endif %}{% if i == 4 %}{% break %}{% endif %}{{ i }}{% endfor %}|{% for i in l %}{% for j in l %}{% if j > i %}{% break %}{% endif %}{{ j }}{% endfor %},{% endfor %}", NESTED)
    add("loop_reversed_slice_index", "{% for m in l[::-1] %}{{ (l|length - 1) - loop.index0 }}:{{ m }} {% endfor %}", NESTED)

    add("range_is_not_a_list", "{{ range(3)|length }} {{ 2 in range(3) }} {{ 5 in range(3) }} {{ range(3) is sequence }}{{ range(3) is iterable }}{{ range(3) is mapping }} {{ range(2, 9)[1] }}{{ range(2, 9)[-1] }}[{{ range(3)[7] }}] {{ 'T' if range(0) else 'F' }}{{ 'T' if range(1) else 'F' }} {{ range(4)|join('') }}{% set r = range(2) %}{% for i in r %}{{ i }}{% endfor %}{% for i in r %}{{ i }}{% endfor %}")
    add("range_tojson_raises", "{{ range(3)|tojson }}")
    add("loop_inside_macro_called_from_loop", "{% macro m() %}[{{ loop is defined }}{% for q in [7, 8] %}{{ loop.index }}{{ q }}{% endfor %}]{% endmacro %}{% for i in [1, 2] %}{{ loop.index }}{{ m() }}{% endfor %}")
    add("loop_iterable_expressions", "{% for x in missing|default([1, 2]) %}{{ x }}{% endfor %}{% for x in (l if t else e) %}{{ x }}{% endfor %}{% for x in l[1:3] + [9] %}{{ x }}{% endfor %}", NESTED)

    # if.
    add("if_elif_else", "{% for v in [1, 2, 3, 4] %}{% if v == 1 %}one{% elif v == 2 %}two{% elif v == 3 %}three{% else %}many{% endif %} {% endfor %}")
    add("if_nested", "{% if t %}{% if z %}a{% else %}b{% if n > 2 %}c{% endif %}{% endif %}{% endif %}", NESTED)

    # Macros.
    add("macro_basic", "{% macro greet(name, greeting='Hello') %}{{ greeting }}, {{ name }}!{% endmacro %}{{ greet('A') }} {{ greet('B', 'Hi') }} {{ greet(greeting='Yo', name='C') }} {{ greet('D', greeting='Hey') }}")
    add("macro_missing_argument_is_undefined", "{% macro m(a, b) %}[{{ a }}|{{ b }}|{{ b is defined }}]{% endmacro %}{{ m(1) }}{{ m() }}")
    add("macro_too_many_arguments_raises", "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}")
    add("macro_unknown_keyword_raises", "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(b=2) }}")
    add("macro_result_is_a_string", "{% macro m(a) %}  {{ a }}  {% endmacro %}{{ m(5)|trim ~ '!' }}{{ m(5) is string }}{{ m(5)|length }}{{ m('x') + 'y' }}{% set r = m('q')|trim %}{{ r.startswith('q') }}")
    add("macro_sees_globals_not_caller_locals", "{% set g = 'global' %}{% macro m() %}{{ g }}/{{ local }}/{{ later }}/{{ ctx }}{% endmacro %}{% for i in [1] %}{% set local = 'loop' %}{{ m() }}{% endfor %}{% set later = 'set later' %} {{ m() }}", {"ctx": "context"})
    add("macro_set_is_local", "{% set x = 'outer' %}{% macro m() %}{% set x = 'inner' %}{{ x }}{% endmacro %}{{ m() }}{{ x }}")
    add("macro_namespace_mutation", "{% set ns = namespace(v=0) %}{% macro bump(by) %}{% set ns.v = ns.v + by %}{% endmacro %}{{ bump(2) }}{{ bump(3) }}{{ ns.v }}")
    add("macro_argument_shadows_global", "{% set content = 'global' %}{% macro m(content) %}{{ content }}{% endmacro %}{{ m('arg') }}{{ content }}")
    add("macro_calls_macro", "{% macro inner(x) %}<{{ x }}>{% endmacro %}{% macro outer(x) %}{{ inner(x) }}{{ later(x) }}{% endmacro %}{% macro later(x) %}({{ x }}){% endmacro %}{{ outer(1) }}")
    add("macro_recursion", "{% macro down(n) %}{{ n }}{% if n > 0 %}{{ down(n - 1) }}{% endif %}{% endmacro %}{{ down(3) }}")
    add("macro_defined_in_if", "{% if t %}{% macro m() %}yes{% endmacro %}{% endif %}{{ m() }}", NESTED)
    add("macro_not_defined_raises", "{% if z %}{% macro m() %}yes{% endmacro %}{% endif %}{{ m() }}", NESTED)
    add("macro_default_uses_earlier_argument", "{% macro m(a, b=a) %}{{ a }}{{ b }}{% endmacro %}{{ m(1) }}{{ m(1, 2) }}")
    add("macro_whitespace", "{%- macro m(x) %}\n    {%- if x %}\n        {{- x }}\n    {%- endif %}\n{%- endmacro %}\n[{{ m('v') }}][{{ m('') }}]")
    add("macro_break_in_loop_inside", "{% macro m() %}{% for i in [1, 2, 3] %}{% if i == 2 %}{% break %}{% endif %}{{ i }}{% endfor %}{% endmacro %}{% for j in [1, 2] %}{{ m() }}{% endfor %}")

    # Tests.
    add("tests_types", "{% for v in [1, 1.5, 's', [1], {'a': 1}, none, true, false, missing] %}{{ 'd' if v is defined }}{{ 'n' if v is none }}{{ 's' if v is string }}{{ 'm' if v is mapping }}{{ 'i' if v is iterable }}{{ 'q' if v is sequence }}{{ '#' if v is number }}{{ 'I' if v is integer }}{{ 'F' if v is float }}{{ 'b' if v is boolean }}{{ 'T' if v is true }}{{ 'f' if v is false }} {% endfor %}")
    add("tests_true_is_identity", "{{ 1 is true }} {{ t is true }} {{ 0 is false }} {{ z is false }} {{ t is not false }} {{ 'x' is not string }}", NESTED)
    add("tests_on_special_values", "{% set ns = namespace(a=1) %}{% macro m() %}{% endmacro %}{{ ns is mapping }}{{ ns is iterable }}{{ ns is defined }}{{ m is defined }}{{ m is string }}{% for i in [1] %}{{ loop is defined }}{% endfor %}")

    # Filters.
    add("filter_trim", "[{{ '  a b  '|trim }}][{{ '\n\t x \r\n'|trim }}][{{ ws|trim }}][{{ ''|trim }}][{{ '   '|trim }}][{{ 5|trim }}][{{ none|trim }}][{{ true|trim }}]",
        {"ws": "　 \x1c\x1d\x1e\x1f\x85       \x0b\x0c mid dle ​﻿ "})
    add("filter_tojson_scalars", "{{ 1|tojson }} {{ 1.0|tojson }} {{ 's'|tojson }} {{ true|tojson }} {{ none|tojson }} {{ 2.5e-7|tojson }} {{ 1e22|tojson }} {{ []|tojson }} {{ {}|tojson }}")
    add("filter_tojson_escaping", "{{ v|tojson }}", {"v": "<tag attr='x'> & \"quoted\" \\ / \n \r \t \b \f \x00 \x1f \x7f \u0080 é 中 🌍     </script>"})
    add("filter_tojson_ensure_ascii", "{{ v|tojson(ensure_ascii=True) }} {{ v|tojson(ensure_ascii=False) }} {{ v|tojson(ensure_ascii=true) }}", {"v": {"k é": ["中", "🌍", "\x7f\u0080￿", "<a&b>'"]}})
    add("filter_tojson_nested", "{{ a|tojson }} {{ d|tojson }} {{ [1, 'a', [none, true], {'k': 1.50}]|tojson }} {{ (1, 2)|tojson }}", NESTED)
    add("filter_tojson_key_order", "{{ v|tojson }}", {"v": {"zebra": 1, "apple": {"m": 1, "b": 2, "a": 3}, "Mango": [], "10": 0, "9": 0, "": 0}})
    add("filter_tojson_floats", "{{ v|tojson }}", {"v": [0.1, 1.0, 100.0, 1e22, 1e16, 1e15, 9999999999999998.0, 123456789012345680.0, 1e-05,
                                                        0.0001, 0.00012345, 1.5e300, 5e-324, -0.5, 3.141592653589793, 1.7976931348623157e308,
                                                        2.5e-07, 1e100, 0.0, 12345.678, 1e-10, 4.35, 0.30000000000000004,
                                                        9007199254740993, -9223372036854775808, 9223372036854775807]})
    add("filter_tojson_namespace_raises", "{% set ns = namespace(a=1) %}{{ ns|tojson }}")
    add("filter_length", "{{ l|length }} {{ s|length }} {{ d|length }} {{ ''|length }} {{ e|count }} {{ '🌍é'|length }}", NESTED)
    add("filter_length_of_number_raises", "{{ n|length }}", NESTED)
    add("filter_string", "[{{ n|string }}][{{ f|string }}][{{ t|string }}][{{ z|string }}][{{ s|string }}]{{ n|string is string }}{{ (n|string) + '0' }}", NESTED)
    add("filter_safe", "{{ '<b>'|safe }}{{ n|safe }}{{ ('<i>'|safe) is string }}{{ '<b>'|tojson|safe }}{% set v = d|tojson|safe %}{{ v }}{{ v|length }}", NESTED)
    add("filter_items", "{% for pair in d|items %}{{ pair[0] }}{{ pair[1] }}{{ pair|length }}{% endfor %}", NESTED)
    add("filter_items_is_a_generator", "{% set g = d|items %}{{ g is iterable }}{{ g is sequence }}{{ 'T' if g else 'F' }}{{ 'T' if e2|items else 'F' }}{% for k, v in g %}{{ k }}{% endfor %}|{% for k, v in g %}{{ k }}{% endfor %}|[{{ (d|items)[0] }}]", dict(NESTED, e2={}))
    add("filter_items_length_raises", "{{ d|items|length }}", NESTED)
    add("filter_items_tojson_raises", "{{ d|items|tojson }}", NESTED)
    add("filter_items_of_string_raises", "{% for k, v in s|items %}{% endfor %}", NESTED)
    add("filter_items_of_list_raises", "{% for k, v in l|items %}{% endfor %}", NESTED)
    add("filter_default", "{{ missing|default('x') }}{{ n|default('x') }}{{ d.nope|default(7) }}{{ ''|default('x') }}|{{ ''|default('x', true) }}{{ missing|default(none) is none }}", NESTED)
    add("filter_capitalize_replace", "{{ 'xHIGH'|capitalize }} {{ 'max'|capitalize }} {{ ''|capitalize }}|{{ 'image_url'|replace('_url', '')|replace('input_', '') }} {{ 'aaa'|replace('a', 'bb') }} {{ 'input_audio'|replace('_url', '')|replace('input_', '') }} {{ 5|replace(5, 6) }}")
    add("filter_join_lower_upper", "{{ l|join(', ') }} {{ l|join }} {{ ['a', 1, none]|join('-') }} {{ 'AbC'|lower }} {{ 'AbC'|upper }} {{ d|join('+') }}", NESTED)
    add("filter_chain_on_call_and_attribute", "{% macro m(x) %} {{ x }} {% endmacro %}[{{ m(d.k1)|trim|length }}][{{ d.k1|trim ~ d.k2|string|trim }}]", NESTED)

    # String and mapping methods.
    add("method_startswith_endswith", "{{ s.startswith('hé') }} {{ s.startswith('x') }} {{ s.endswith('ld') }} {{ s.endswith('') }} {{ s.startswith(('x', 'h')) }} {{ not(s.startswith('h') and s.endswith('d')) }}", NESTED)
    add("method_strip", "[{{ '  a  '.strip() }}][{{ '  a  '.lstrip() }}][{{ '  a  '.rstrip() }}][{{ 'xxaxx'.strip('x') }}][{{ ws.strip() }}][{{ 'abcba'.strip('ab') }}]", {"ws": "　\n a  \x1f"})
    add("method_split", "{{ 'a</think>b</think>c'.split('</think>')|tojson }} {{ 'x'.split('</think>')[0] }} {{ '<think>r</think>a'.split('</think>')[0].split('<think>')[-1] }} {{ ' a  b\tc\n'.split()|tojson }} {{ ''.split(',')|tojson }} {{ ''.split()|tojson }}")
    add("method_replace", "{{ 'a-b-c'.replace('-', '+') }} {{ s.replace('ö', 'oe') }}", NESTED)
    add("method_mapping", "{{ d.get('k1') }} {{ d.get('nope') }} {{ d.get('nope', 'dflt') }} {{ d.keys()|length }} {{ d.items()|length }}", NESTED)
    add("method_mapping_views", "{{ 'k1' in d.keys() }} {{ 'v1' in d.values() }} {{ 'T' if d.keys() else 'F' }}{{ 'T' if {}.keys() else 'F' }} [{{ d.keys()[0] }}] {{ d.values() is sequence }}{{ d.values() is iterable }}{{ d.keys() is mapping }} {{ d.keys()|join(',') }} {% set ks = d.keys() %}{% for k in ks %}{{ k }}{% endfor %}{% for k in ks %}{{ k }}{% endfor %}", NESTED)
    add("method_mapping_values_tojson_raises", "{{ d.values()|tojson }}", NESTED)
    add("method_on_undefined_raises", "{{ missing.startswith('a') }}")
    add("method_on_wrong_type_raises", "{{ n.startswith('a') }}", NESTED)
    add("method_items_on_key_named_items", "{% for k, v in items.items() %}{{ k }}={{ v }}{% endfor %}", NESTED)

    # Arithmetic on other types, and errors.
    add("add_strings_and_lists", "{{ 'a' + 'b' }} {{ ([1] + [2, 3])|tojson }} {{ (1,) + (2,) }} {{ 1 + 2.5 }} {{ true + 1 }} {{ 2 * 3.0 }} {{ 5 - 2.5 }}")
    add("add_string_and_number_raises", "{{ 'a' + 1 }}")
    add("divide_by_zero_raises", "{{ 1 / 0 }}")
    add("raise_exception", "before{{ raise_exception('Something ' ~ n ~ ' went wrong: é') }}after", NESTED)
    add("raise_exception_in_macro", "{% macro m(x) %}{% if x %}{{ raise_exception('from the macro') }}{% endif %}ok{% endmacro %}{{ m(false) }}{{ m(true) }}")
    add("raise_exception_in_loop", "{% for i in l %}{% if i == 3 %}{{ raise_exception('at ' ~ loop.index0) }}{% endif %}{% endfor %}", NESTED)
    add("dict_and_list_literals", "{% set v = {'a': [1, 2, (3, 4)], 'b': {'c': none}, n: s} %}{{ v|tojson }} {{ v.a[2][1] }} {{ [n, s, [t]]|tojson }} {{ ()|tojson }} {{ (1,)|tojson }} {{ ('xhigh', 'medium', 'low')|length }}", {"n": "key", "s": "val", "t": True})

    # Templates jinja2 itself refuses.
    add("syntax_unclosed_if", "{% if true %}x")
    add("syntax_unknown_tag", "{% frobnicate %}")
    add("syntax_unknown_filter", "{{ 1|frobnicate }}")
    add("syntax_unknown_test", "{{ 1 is frobnicated }}")
    add("syntax_unclosed_variable", "{{ 1 + }}")
    add("syntax_unclosed_tag", "a {% if true")
    add("syntax_break_outside_loop", "{% break %}")
    add("syntax_stray_endif", "x{% endif %}")
    add("syntax_unterminated_string", "{{ 'abc }}")
    return u


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", action="append", default=[], metavar="NAME=DIR",
                        help="a checkpoint directory with chat_template.jinja; NAME is what the cases call it")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    provenance = {"generator": "tools/reference/template_cases.py", "tensorfold": tensorfold.__version__,
                  "jinja2": jinja2.__version__,
                  "environment": "ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, "
                                 "extensions=[loopcontrols]); tojson = json.dumps(ensure_ascii=False)"}

    templates, cases = [], []
    for spec in args.model:
        name, _, directory = spec.partition("=")
        model_dir = Path(directory)
        source = (model_dir / "chat_template.jinja").read_bytes()
        chat = ChatTemplate(model_dir)
        templates.append({"name": name, "sha256": hashlib.sha256(source).hexdigest(), "bytes": len(source),
                          "specials": chat.specials, "late_system": chat.late_system})
        for case, context in direct_cases():
            cases.append({"name": f"{name}/direct/{case}", "template": name, "context": context,
                          **outcome(lambda: chat.template.render(**context))})
        recorder = Recorder(chat.template)
        chat.template = recorder
        for case, call_args in server_cases():
            recorder.kwargs = None
            result = outcome(lambda: chat.render(call_args["messages"], tools=call_args["tools"],
                                                 enable_thinking=call_args["enable_thinking"],
                                                 extra=call_args.get("extra")))
            if recorder.kwargs is None:
                raise SystemExit(f"{name}/server/{case}: the request was refused before the template: {result}")
            cases.append({"name": f"{name}/server/{case}", "template": name, "context": recorder.kwargs, **result})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if templates:
        write(args.out_dir / "template_cases.json", {"provenance": provenance, "templates": templates}, cases)
        failed = sum("error" in c for c in cases)
        print(f"template_cases.json: {len(cases)} cases over {len(templates)} templates, {failed} expect an error")

    units = []
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "tokenizer_config.json").write_text("{}")
        for name, source, context in unit_cases():
            (root / "chat_template.jinja").write_text(source, newline="")

            def render():
                return ChatTemplate(root).template.render(**context)

            units.append({"name": name, "source": source, "context": context, **outcome(render)})
    names = [c["name"] for c in units]
    assert len(names) == len(set(names)), "unit case names must be unique"
    write(args.out_dir / "template_unit_cases.json", {"provenance": provenance}, units)
    failed = sum("error" in c for c in units)
    print(f"template_unit_cases.json: {len(units)} cases, {failed} expect an error")


if __name__ == "__main__":
    main()
