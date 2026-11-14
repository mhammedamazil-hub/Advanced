"""Synthetic, legal training-knowledge generator.

The spec asks for training knowledge covering "every single skill".  We cannot
download The Stack/CodeSearchNet from this sandbox, so we honestly *generate*
a deterministic curriculum corpus across a broad skill taxonomy (see
``TAXONOMY``).  Each entry renders (a) an instruction/example pair used to
train the compact model and (b) an executable task (prompt + reference
solution + tests) used for practice during sleep and for evaluation.

Nothing here claims external datasets were used: every sample is synthetic and
provenance-tagged ``synthetic-curriculum``.  ``external_sources.py`` documents
the real datasets to fetch on a connected machine with license filtering.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

PROVENANCE = "synthetic-curriculum"


@dataclass
class Task:
    task_id: str
    domain: str
    prompt: str                # natural language description incl. signature
    reference: str             # reference solution code
    tests: str                 # python asserts to verify a candidate solution
    difficulty: int = 1

    def instruction_pair(self) -> str:
        return (f"### Task: {self.prompt}\n"
                f"### Solution:\n{self.reference.strip()}\n### End\n")


# --------------------------------------------------------------------------
# Task template families.  Each family is a function that returns Task(s) with
# parameters randomized by a seeded RNG.
# --------------------------------------------------------------------------

def _function_header(reference: str) -> str:
    m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*\([^)]*\))", reference)
    return m.group(1) if m else ""


def _arith_tasks(rng: random.Random) -> list[Task]:
    tasks = []
    a, b = rng.randint(2, 20), rng.randint(2, 20)
    tasks.append(Task(
        task_id=f"py-add-{a}-{b}", domain="python-basics",
        prompt="Write a function add(a, b) that returns the sum of a and b.",
        reference="def add(a, b):\n    return a + b\n",
        tests=f"assert add(2, 3) == 5\nassert add({a}, {b}) == {a + b}\nassert add(-1, 1) == 0",
        difficulty=1))
    tasks.append(Task(
        task_id="py-clamp", domain="python-basics",
        prompt="Write a function clamp(x, lo, hi) that limits x to [lo, hi].",
        reference="def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
        tests="assert clamp(5, 0, 10) == 5\nassert clamp(-3, 0, 10) == 0\n"
              "assert clamp(99, 0, 10) == 10",
        difficulty=1))
    tasks.append(Task(
        task_id="py-celsius-to-fahrenheit", domain="python-basics",
        prompt="Write a function celsius_to_fahrenheit(c) converting °C to °F.",
        reference="def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32\n",
        tests="assert celsius_to_fahrenheit(0) == 32\n"
              "assert celsius_to_fahrenheit(100) == 212",
        difficulty=1))
    tasks.append(Task(
        task_id="py-max-of-three", domain="python-basics",
        prompt="Write a function max_of_three(a, b, c) returning the largest.",
        reference="def max_of_three(a, b, c):\n    return max(a, b, c)\n",
        tests="assert max_of_three(1, 2, 3) == 3\nassert max_of_three(9, 2, 5) == 9\n"
              "assert max_of_three(-1, -2, -3) == -1",
        difficulty=1))
    return tasks


def _string_tasks(rng: random.Random) -> list[Task]:
    word = rng.choice(["python", "memory", "brain", "neural", "coder"])
    return [
        Task(task_id="py-reverse-string", domain="python-strings",
             prompt="Write a function reverse_string(s) that returns s reversed.",
             reference="def reverse_string(s):\n    return s[::-1]\n",
             tests=f"assert reverse_string('{word}') == '{word[::-1]}'\n"
                   "assert reverse_string('') == ''",
             difficulty=1),
        Task(task_id="py-count-vowels", domain="python-strings",
             prompt="Write a function count_vowels(s) counting a,e,i,o,u (case-insensitive).",
             reference=("def count_vowels(s):\n"
                        "    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"),
             tests="assert count_vowels('hello') == 2\nassert count_vowels('SKY') == 0\n"
                   "assert count_vowels('Programming') == 3",
             difficulty=1),
        Task(task_id="py-is-palindrome", domain="python-strings",
             prompt="Write a function is_palindrome(s) checking if s reads the same both ways.",
             reference=("def is_palindrome(s):\n"
                        "    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
                        "    return cleaned == cleaned[::-1]\n"),
             tests="assert is_palindrome('racecar')\nassert not is_palindrome('brain')\n"
                   "assert is_palindrome('A man, a plan, a canal: Panama')",
             difficulty=2),
    ]


def _algo_tasks(rng: random.Random) -> list[Task]:
    n = rng.randint(6, 12)
    fibs = [0, 1]
    for _ in range(n):
        fibs.append(fibs[-1] + fibs[-2])
    return [
        Task(task_id="algo-factorial", domain="algorithms",
             prompt="Write a function factorial(n) returning n! for n >= 0.",
             reference=("def factorial(n):\n"
                        "    result = 1\n"
                        "    for i in range(2, n + 1):\n"
                        "        result *= i\n"
                        "    return result\n"),
             tests="assert factorial(0) == 1\nassert factorial(5) == 120\n"
                   "assert factorial(7) == 5040",
             difficulty=1),
        Task(task_id=f"algo-fibonacci-{n}", domain="algorithms",
             prompt="Write a function fibonacci(n) returning the n-th Fibonacci number (fib(0)=0, fib(1)=1).",
             reference=("def fibonacci(n):\n"
                        "    a, b = 0, 1\n"
                        "    for _ in range(n):\n"
                        "        a, b = b, a + b\n"
                        "    return a\n"),
             tests=f"assert fibonacci(0) == 0\nassert fibonacci(1) == 1\n"
                   f"assert fibonacci({n}) == {fibs[n]}",
             difficulty=2),
        Task(task_id="algo-is-prime", domain="algorithms",
             prompt="Write a function is_prime(n) returning True iff n is prime.",
             reference=("def is_prime(n):\n"
                        "    if n < 2:\n"
                        "        return False\n"
                        "    for i in range(2, int(n ** 0.5) + 1):\n"
                        "        if n % i == 0:\n"
                        "            return False\n"
                        "    return True\n"),
             tests="assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\n"
                   "assert not is_prime(15)\nassert is_prime(97)",
             difficulty=2),
        Task(task_id="algo-gcd", domain="algorithms",
             prompt="Write a function gcd(a, b) using the Euclidean algorithm.",
             reference=("def gcd(a, b):\n"
                        "    while b:\n"
                        "        a, b = b, a % b\n"
                        "    return abs(a)\n"),
             tests="assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1\nassert gcd(0, 4) == 4",
             difficulty=2),
        Task(task_id="algo-binary-search", domain="algorithms",
             prompt=("Write a function binary_search(arr, target) returning the index "
                     "of target in sorted arr, or -1."),
             reference=("def binary_search(arr, target):\n"
                        "    lo, hi = 0, len(arr) - 1\n"
                        "    while lo <= hi:\n"
                        "        mid = (lo + hi) // 2\n"
                        "        if arr[mid] == target:\n"
                        "            return mid\n"
                        "        if arr[mid] < target:\n"
                        "            lo = mid + 1\n"
                        "        else:\n"
                        "            hi = mid - 1\n"
                        "    return -1\n"),
             tests=("assert binary_search([1, 3, 5, 7], 5) == 2\n"
                    "assert binary_search([1, 3, 5, 7], 4) == -1\n"
                    "assert binary_search([], 1) == -1\n"
                    "assert binary_search([2], 2) == 0"),
             difficulty=2),
        Task(task_id="algo-bubble-sort", domain="algorithms",
             prompt="Write a function bubble_sort(arr) returning a sorted copy of arr.",
             reference=("def bubble_sort(arr):\n"
                        "    arr = list(arr)\n"
                        "    n = len(arr)\n"
                        "    for i in range(n):\n"
                        "        for j in range(0, n - i - 1):\n"
                        "            if arr[j] > arr[j + 1]:\n"
                        "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
                        "    return arr\n"),
             tests=("assert bubble_sort([3, 1, 2]) == [1, 2, 3]\n"
                    "assert bubble_sort([]) == []\n"
                    "assert bubble_sort([5, -1, 0]) == [-1, 0, 5]"),
             difficulty=2),
    ]


def _data_structure_tasks(rng: random.Random) -> list[Task]:
    return [
        Task(task_id="ds-sum-list", domain="data-structures",
             prompt="Write a function sum_list(xs) summing a list of numbers.",
             reference="def sum_list(xs):\n    return sum(xs)\n",
             tests="assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0",
             difficulty=1),
        Task(task_id="ds-unique-elements", domain="data-structures",
             prompt="Write a function unique_elements(xs) returning unique values, order preserved.",
             reference=("def unique_elements(xs):\n"
                        "    seen = []\n"
                        "    for x in xs:\n"
                        "        if x not in seen:\n"
                        "            seen.append(x)\n"
                        "    return seen\n"),
             tests="assert unique_elements([1, 2, 2, 3, 1]) == [1, 2, 3]\n"
                   "assert unique_elements([]) == []",
             difficulty=1),
        Task(task_id="ds-square-evens", domain="data-structures",
             prompt="Write a function square_evens(xs) returning squares of the even numbers.",
             reference=("def square_evens(xs):\n"
                        "    return [x * x for x in xs if x % 2 == 0]\n"),
             tests="assert square_evens([1, 2, 3, 4]) == [4, 16]\n"
                   "assert square_evens([1, 3]) == []",
             difficulty=1),
        Task(task_id="ds-merge-sorted", domain="data-structures",
             prompt="Write a function merge_sorted(a, b) merging two sorted lists into one sorted list.",
             reference=("def merge_sorted(a, b):\n"
                        "    i = j = 0\n"
                        "    out = []\n"
                        "    while i < len(a) and j < len(b):\n"
                        "        if a[i] <= b[j]:\n"
                        "            out.append(a[i]); i += 1\n"
                        "        else:\n"
                        "            out.append(b[j]); j += 1\n"
                        "    return out + a[i:] + b[j:]\n"),
             tests="assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]\n"
                   "assert merge_sorted([], [1]) == [1]\n"
                   "assert merge_sorted([1, 2], []) == [1, 2]",
             difficulty=2),
        Task(task_id="ds-stack-with-min", domain="data-structures",
             prompt=("Write a class MinStack with push(x), pop(), and get_min() "
                     "all O(1) — keep an auxiliary stack of running minima."),
             reference=("class MinStack:\n"
                        "    def __init__(self):\n"
                        "        self._items = []\n"
                        "        self._mins = []\n"
                        "    def push(self, x):\n"
                        "        self._items.append(x)\n"
                        "        self._mins.append(x if not self._mins else min(x, self._mins[-1]))\n"
                        "    def pop(self):\n"
                        "        self._mins.pop()\n"
                        "        return self._items.pop()\n"
                        "    def get_min(self):\n"
                        "        return self._mins[-1]\n"),
             tests=("s = MinStack()\ns.push(3)\ns.push(1)\ns.push(2)\n"
                    "assert s.get_min() == 1\ns.pop()\nassert s.get_min() == 1\n"
                    "s.pop()\nassert s.get_min() == 3"),
             difficulty=3),
    ]


def _fizzbuzz_task(rng: random.Random) -> list[Task]:
    return [
        Task(task_id="py-fizzbuzz", domain="python-basics",
             prompt=("Write a function fizzbuzz(n) returning a list of strings 1..n "
                     "with 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
                     "'FizzBuzz' for both."),
             reference=("def fizzbuzz(n):\n"
                        "    out = []\n"
                        "    for i in range(1, n + 1):\n"
                        "        if i % 15 == 0:\n"
                        "            out.append('FizzBuzz')\n"
                        "        elif i % 3 == 0:\n"
                        "            out.append('Fizz')\n"
                        "        elif i % 5 == 0:\n"
                        "            out.append('Buzz')\n"
                        "        else:\n"
                        "            out.append(str(i))\n"
                        "    return out\n"),
             tests=("r = fizzbuzz(5)\n"
                    "assert r[:5] == ['1', '2', 'Fizz', '4', 'Buzz']\n"
                    "assert fizzbuzz(15)[14] == 'FizzBuzz'"),
             difficulty=1),
        Task(task_id="py-comprehension-filter", domain="python-basics",
             prompt="Write a function positive_odds(xs) returning the positive odd numbers from xs.",
             reference="def positive_odds(xs):\n    return [x for x in xs if x > 0 and x % 2 == 1]\n",
             tests="assert positive_odds([-3, 0, 1, 2, 5]) == [1, 5]\n"
                   "assert positive_odds([]) == []",
             difficulty=1),
    ]


# SQL / JS / git / debugging live here as instruction pairs (not all executable
# in the tiny Python sandbox), so the model still sees those skills in training.
_KNOWLEDGE_SNIPPETS: list[tuple[str, str, str]] = [
    ("sql", "Write an SQL query to find the 5 most recent orders with customer names.",
     "SELECT c.name, o.created_at\nFROM orders o\nJOIN customers c ON c.id = o.customer_id\n"
     "ORDER BY o.created_at DESC\nLIMIT 5;"),
    ("sql", "Explain and fix a slow query that scans a large users table by email.",
     "Add an index: CREATE INDEX idx_users_email ON users(email); — then verify with EXPLAIN."),
    ("javascript", "Write a JavaScript function debounce(fn, ms).",
     "function debounce(fn, ms) {\n  let t;\n  return (...args) => {\n"
     "    clearTimeout(t);\n    t = setTimeout(() => fn(...args), ms);\n  };\n}"),
    ("javascript", "How do you deep-copy a JSON-safe object in JavaScript?",
     "const copy = structuredClone(obj);  // or JSON.parse(JSON.stringify(obj)) for old runtimes"),
    ("git", "How do you undo the last local commit but keep the changes staged?",
     "git reset --soft HEAD~1  # commit undone, changes staged for a new commit"),
    ("git", "How do you update your feature branch with the latest main without merge commits?",
     "git fetch origin && git rebase origin/main"),
    ("debugging", "A Python program throws 'IndexError: list index out of range'. How do you debug it?",
     "Read the traceback to find the exact line, print len(xs) and the index used, "
     "then guard the loop bounds or fix the off-by-one."),
    ("debugging", "A script works locally but fails in CI with 'ModuleNotFoundError'. Why?",
     "The dependency is installed locally but not in CI. Add it to requirements.txt "
     "and install it in the CI setup step."),
    ("debugging", "A function returns None unexpectedly. Common cause?",
     "A code path lacks a return statement — check every branch returns a value."),
    ("testing", "Write a pytest test for add(a, b).",
     "def test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0"),
    ("regex", "Write a regex that matches an ISO date YYYY-MM-DD.",
     r"^\d{4}-\d{2}-\d{2}$"),
    ("web", "What HTTP status code should a successful POST that creates a resource return?",
     "201 Created, ideally with a Location header pointing at the new resource."),
    ("security", "Why should SQL queries use parameterized statements?",
     "To prevent SQL injection: parameters are never concatenated into the query string, "
     "so user input cannot change query structure."),
    ("rust", "What does the Rust borrow checker guarantee?",
     "Memory safety without a garbage collector: data has a single owner, borrows are "
     "either many immutable or one mutable reference, and no dangling references exist."),
    ("math", "What is the time complexity of binary search on n sorted items and why?",
     "O(log n): each comparison halves the remaining search interval."),
]


TAXONOMY: dict[str, str] = {
    "python-basics": "Python functions, control flow, comprehensions",
    "python-strings": "String processing and parsing",
    "algorithms": "Classic algorithms & complexity",
    "data-structures": "Lists, stacks, dictionaries, patterns",
    "sql": "Querying and optimizing databases",
    "javascript": "Core JavaScript and async utilities",
    "git": "Version control workflows",
    "debugging": "Systematic debugging strategies",
    "testing": "Unit testing and verification",
    "regex": "Regular expressions",
    "web": "HTTP/API fundamentals",
    "security": "Security basics",
    "rust": "Systems programming concepts",
    "math": "Complexity analysis & discrete math",
}


def executable_tasks(seed: int = 0) -> list[Task]:
    """All sandbox-executable practice tasks (python), deterministic."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    for gen in (_arith_tasks, _string_tasks, _algo_tasks,
                _data_structure_tasks, _fizzbuzz_task):
        tasks.extend(gen(rng))
    return tasks


def build_corpus(samples_per_task: int = 2, seed: int = 0) -> list[str]:
    """Build the synthetic instruction corpus used to train the compact model.

    Combines executable tasks (rendered as instruction/solution pairs) and
    knowledge snippets covering the full TAXONOMY.  Repeated formats teach the
    model the '### Task / ### Solution' contract reliably on a tiny budget.
    """
    rng = random.Random(seed)
    corpus: list[str] = []
    for task in executable_tasks(seed):
        for _ in range(samples_per_task):
            corpus.append(task.instruction_pair())
    for domain, q, a in _KNOWLEDGE_SNIPPETS:
        corpus.append(f"### Task: {q}\n### Solution:\n{a}\n### End\n")
        variants = rng.random() < 0.5
        if variants:
            corpus.append(f"### Task: Explain briefly: {q.lower()}\n### Solution:\n{a}\n### End\n")
    return corpus


def practice_tasks_for(domain: str, k: int = 2, seed: int = 0) -> list[Task]:
    """Targeted practice tasks for a weak domain (sleep stage 6)."""
    pool = [t for t in executable_tasks(seed) if t.domain == domain]
    rng = random.Random(seed + 7)
    rng.shuffle(pool)
    return pool[:k]


def skill_taxonomy() -> dict[str, str]:
    return dict(TAXONOMY)


def function_names(code: str) -> list[str]:
    return re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", code)
