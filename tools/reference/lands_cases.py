#!/usr/bin/env python3
"""The cases spec 085 records its chains on: a fitting set and a held-out set of prose and of
code prompts, no prompt in both. Not runtime; standard library only, no GPU (`zspark tokens` reads the tokenizer).

  lands_cases.py ZSPARK DIR OUT.cases [TOKENS]   a line a case for `zspark generate DIR 0 cases=OUT.cases`:           
                                                 `TOKENS chains name=KIND-SET-N ID..`, TOKENS 256 unless given

  lands_cases.py held OUT_DIR                    the held-out prompts as `client_bench.py streams --prompts` reads
                                                 them: OUT_DIR/prose-held.json and code-held.json, the cases' order

KIND is `prose` or `code`, SET `fit` or `held`. The held-out set begins with the bench's own prompts
(`client_bench.py streams`: eight paragraphs, four functions; spec 078's `bench.cases`), so the replay's counts
on them stay those of a load; the others were written for this spec, and four of each prose set are a summary of
a cut of `client_bench_corpus.txt` (the repository's own specs: technical prose, not general). Every prompt is
one user turn without thinking, as the bench sends it.
"""
import os
import subprocess
import sys

from client_bench import _CODE_PROMPTS, _STREAM_PROMPTS

PROSE = {
    "fit": [
        "Write one paragraph about a fishing boat leaving before dawn.",
        "Write one paragraph about a baker's first hour of work.",
        "Write one paragraph about an old bridge over a slow river.",
        "Write one paragraph about a school playground after the bell.",
        "Write one paragraph about a tailor's shop on a narrow street.",
        "Write one paragraph about the last train of the night.",
        "Write one paragraph about a beekeeper in early summer.",
        "Write one paragraph about a chess game in a public park.",
        "Write one paragraph about a glassblower at the furnace.",
        "Write one paragraph about the first snow in a small town.",
        "Write one paragraph about an astronomer's long night at the telescope.",
        "Write one paragraph about a ferry crossing in heavy fog.",
        "Explain in two paragraphs how a refrigerator keeps food cold.",
        "Explain in two paragraphs why the sky is blue and sunsets are red.",
        "Explain in two paragraphs how vaccines train the immune system.",
        "Explain in two paragraphs how a sailing boat can travel against the wind.",
        "Explain in two paragraphs what inflation is and why it matters to a household.",
        "Explain in two paragraphs how bread rises.",
        "Explain in two paragraphs how the tides follow the moon.",
        "Explain in two paragraphs how a public key lets strangers exchange a secret.",
        "Describe the causes of the French Revolution in a short essay.",
        "Describe how the Roman aqueducts were built and why they mattered.",
        "Describe the life cycle of a star like the Sun.",
        "Describe what a good code review looks like, for a new engineer.",
        "Give advice to someone moving to a new city where they know nobody.",
        "Give a short account of how the telegraph changed the news.",
        "Compare travelling by train with travelling by plane, in two paragraphs.",
        "Compare living in a large city with living in a village, in two paragraphs.",
        "Write a short letter thanking a teacher who is retiring.",
        "Write a short speech for the opening of a neighbourhood library.",
        "Write a product description for a hand-made wooden desk lamp.",
        "Write a short review of an imaginary restaurant by the sea.",
    ],
    "held": list(_STREAM_PROMPTS) + [
        "Write one paragraph about a potter's wheel turning.",
        "Write one paragraph about a lighthouse seen from a ship.",
        "Write one paragraph about a violin maker choosing wood.",
        "Write one paragraph about a street musician in the rain.",
        "Write one paragraph about a mountain hut at the end of a long walk.",
        "Write one paragraph about an orchard at harvest.",
        "Explain in two paragraphs how a bicycle stays upright.",
        "Explain in two paragraphs how a thermostat controls a room's temperature.",
        "Explain in two paragraphs why leaves change colour in autumn.",
        "Explain in two paragraphs how a bank creates money when it lends.",
        "Describe the invention of the printing press and what followed from it.",
        "Describe how a river delta forms.",
        "Give advice to a student preparing for a difficult examination.",
        "Compare reading a novel with watching its film, in two paragraphs.",
        "Write a short letter welcoming a new neighbour.",
        "Write a short review of an imaginary museum of clocks.",
    ],
}
CODE = {
    "fit": [
        "Write a Python function that returns the longest common prefix of a list of strings.",
        "Write a Python function that checks whether a string of brackets is balanced.",
        "Write a Python function that groups a list of words by their sorted letters.",
        "Write a Python function that computes the moving average of a list with a given window.",
        "Write a Python function that flattens a nested list of any depth.",
        "Write a Python function that parses a duration such as 1h30m into seconds.",
        "Write a Python class for a stack with push, pop and a constant-time minimum.",
        "Write a Python function that returns the k most frequent elements of a list.",
        "Write a Go function that returns whether an integer is prime.",
        "Write a Go function that removes duplicates from a sorted slice in place.",
        "Write a Go function that computes the greatest common divisor of two integers.",
        "Write a Go function that counts the words of a string into a map.",
        "Write a Rust function that returns the maximum of a slice of i32 as an Option.",
        "Write a Rust function that checks whether a string is a palindrome, ignoring case.",
        "Write a Rust function that sums the even numbers of a slice of u64.",
        "Write a Rust function that parses a comma-separated line of integers into a Vec.",
        "Write a C function that reverses a string in place.",
        "Write a C function that returns the length of a singly linked list.",
        "Write a C function that swaps two integers through pointers.",
        "Write a C function that finds a value in a sorted array by binary search.",
        "Write a JavaScript function that debounces another function by a number of milliseconds.",
        "Write a JavaScript function that deep-clones a plain object.",
        "Write a JavaScript function that removes the falsy values of an array.",
        "Write a JavaScript function that formats a number of bytes as a readable string.",
        "Write a SQL query that returns the three customers with the highest total order value, from tables customers and orders.",
        "Write a SQL query that counts the orders of each month of 2024 from a table orders with a column created_at.",
        "Write a Java method that returns whether two strings are anagrams.",
        "Write a Java method that computes the factorial of n as a BigInteger.",
        "Write a Bash script that prints the ten largest files under a directory.",
        "Write a Zig function that returns the sum of a slice of u32.",
        "Write a TypeScript function that returns the unique values of an array, keeping their order.",
        "Write a C++ function that rotates a vector of int left by k places.",
    ],
    "held": list(_CODE_PROMPTS) + [
        "Write a Python function that returns the nth row of Pascal's triangle.",
        "Write a Python function that converts a Roman numeral to an integer.",
        "Write a Python function that finds the two numbers of a list that add up to a target.",
        "Write a Python function that removes the vowels of a string.",
        "Write a Python class for a queue built from two stacks.",
        "Write a Go function that merges two maps of string to int, summing the values of a shared key.",
        "Write a Go function that returns the median of a slice of int.",
        "Write a Rust function that reverses the words of a sentence.",
        "Write a Rust function that returns the indices of a slice's local maxima.",
        "Write a C function that copies a string into a buffer of a given size safely.",
        "Write a C function that computes the integer square root of an unsigned 64-bit integer.",
        "Write a JavaScript function that returns a random integer between two bounds, inclusive.",
        "Write a JavaScript function that capitalizes the first letter of each word of a string.",
        "Write a SQL query that returns the employees who earn more than their department's average, from a table employees.",
        "Write a Java method that reverses the words of a sentence.",
        "Write a Bash script that counts the lines of every .txt file of a directory.",
        "Write a TypeScript function that chunks an array into pieces of a given size.",
        "Write a C++ function that returns whether a vector of int is sorted.",
        "Write a Zig function that counts the bytes of a slice equal to a given byte.",
        "Write a Python function that validates an IPv4 address given as a string.",
    ],
}
ONLY = " Output only the code."
OPEN, CLOSE = [248045, 846, 198], [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]


def corpus_cuts():
    """Eight cuts of the bench's corpus by characters, 900 each, ended at a space; newlines made spaces."""
    here = os.path.dirname(os.path.abspath(__file__))
    text = " ".join(open(os.path.join(here, "client_bench_corpus.txt"), encoding="utf-8").read().split())
    cuts = []
    for i in range(8):
        cut = text[2000 + i * 18000:][:900]
        cuts.append("Summarize these notes in one paragraph: " + cut[:cut.rindex(" ")])
    return cuts


def ids(zspark, directory, text):
    out = subprocess.run([zspark, "tokens", directory, text], capture_output=True, text=True, check=True).stdout
    found = [int(line.split("\t")[0]) for line in out.split("\n") if "\t" in line and line.split("\t")[0].isdigit()]
    count = int(out.strip().split("\n")[-1].split()[0])
    if len(found) != count:
        raise SystemExit(f"{text!r}: {len(found)} ids read where {count} are said")
    return found


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "held":
        import json
        cuts = corpus_cuts()
        for kind, prompts in (("prose", PROSE["held"] + cuts[4:]), ("code", [c if c in _CODE_PROMPTS else c + ONLY for c in CODE["held"]])):
            with open(os.path.join(sys.argv[2], f"{kind}-held.json"), "w", encoding="utf-8") as f:
                json.dump(prompts, f, indent=1)
        return
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    zspark, directory, out = sys.argv[1:4]
    tokens = sys.argv[4] if len(sys.argv) > 4 else "256"
    cuts = corpus_cuts()
    sets = {("prose", "fit"): PROSE["fit"] + cuts[:4], ("prose", "held"): PROSE["held"] + cuts[4:],
            ("code", "fit"): [c + ONLY for c in CODE["fit"]],
            ("code", "held"): [c if c in _CODE_PROMPTS else c + ONLY for c in CODE["held"]]}
    seen = set()
    with open(out, "w") as f:
        for (kind, part), prompts in sets.items():
            for n, prompt in enumerate(prompts):
                if prompt in seen:
                    raise SystemExit(f"twice: {prompt!r}")
                seen.add(prompt)
                line = OPEN + ids(zspark, directory, prompt) + CLOSE
                f.write(f"{tokens} chains name={kind}-{part}-{n} " + " ".join(map(str, line)) + "\n")
    print(", ".join(f"{k}-{p} {len(v)}" for (k, p), v in sets.items()))


if __name__ == "__main__":
    main()
