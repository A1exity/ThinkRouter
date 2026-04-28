from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    task_type: str
    query: str
    expected_answer: str
    split: str = "train"
    metadata: dict[str, str] = field(default_factory=dict)


GSM8K_SEED_SAMPLES: list[BenchmarkSample] = [
    BenchmarkSample("gsm8k_seed_001", "gsm8k", "A store had 12 apples and sold 5. How many apples are left?", "7", "train"),
    BenchmarkSample("gsm8k_seed_002", "gsm8k", "Tom has 3 bags with 4 marbles in each bag. How many marbles does he have?", "12", "train"),
    BenchmarkSample("gsm8k_seed_003", "gsm8k", "Lena read 8 pages on Monday and 9 pages on Tuesday. How many pages did she read?", "17", "train"),
    BenchmarkSample("gsm8k_seed_004", "gsm8k", "A taxi ride costs 6 dollars plus 2 dollars per mile. What is the cost for 5 miles?", "16", "train"),
    BenchmarkSample("gsm8k_seed_005", "gsm8k", "There are 24 cookies shared equally by 6 children. How many cookies does each child get?", "4", "train"),
    BenchmarkSample("gsm8k_seed_006", "gsm8k", "A book costs 15 dollars. Mia buys 3 books. How much does she spend?", "45", "train"),
    BenchmarkSample("gsm8k_seed_007", "gsm8k", "Noah had 30 stickers and gave 11 away. How many stickers remain?", "19", "train"),
    BenchmarkSample("gsm8k_seed_008", "gsm8k", "A class has 18 girls and 14 boys. How many students are in the class?", "32", "train"),
    BenchmarkSample("gsm8k_seed_009", "gsm8k", "Sam saves 7 dollars each week for 6 weeks. How much does Sam save?", "42", "train"),
    BenchmarkSample("gsm8k_seed_010", "gsm8k", "A farmer has 5 rows of carrots with 9 carrots in each row. How many carrots are there?", "45", "train"),
    BenchmarkSample("gsm8k_seed_011", "gsm8k", "If 40 pencils are packed into boxes of 8, how many boxes are needed?", "5", "train"),
    BenchmarkSample("gsm8k_seed_012", "gsm8k", "A movie starts at 3 PM and lasts 2 hours. What hour does it end?", "5", "train"),
    BenchmarkSample("gsm8k_seed_013", "gsm8k", "Nina has 50 cents and buys candy for 35 cents. How many cents are left?", "15", "dev"),
    BenchmarkSample("gsm8k_seed_014", "gsm8k", "Four friends each bring 6 balloons. How many balloons do they bring together?", "24", "dev"),
    BenchmarkSample("gsm8k_seed_015", "gsm8k", "A rope is 20 meters long. It is cut into 4 equal pieces. How long is each piece?", "5", "dev"),
    BenchmarkSample("gsm8k_seed_016", "gsm8k", "Ella scored 10 points in the first game and twice as many in the second. What is her total?", "30", "dev"),
    BenchmarkSample("gsm8k_seed_017", "gsm8k", "There are 9 birds on a tree. 4 more birds arrive, then 3 fly away. How many birds remain?", "10", "test"),
    BenchmarkSample("gsm8k_seed_018", "gsm8k", "A pack has 12 cards. Ben buys 2 packs and gives away 5 cards. How many cards does he keep?", "19", "test"),
    BenchmarkSample("gsm8k_seed_019", "gsm8k", "A recipe uses 3 cups of flour per cake. How many cups are needed for 7 cakes?", "21", "test"),
    BenchmarkSample("gsm8k_seed_020", "gsm8k", "Jill ran 2 miles each day for 5 days and then 3 miles on Saturday. How many miles total?", "13", "test"),
]

MATH_SEED_SAMPLES: list[BenchmarkSample] = [
    BenchmarkSample("math_seed_001", "math", "Compute 3^2 + 4^2.", "25", "train"),
    BenchmarkSample("math_seed_002", "math", "Solve for x: 2x + 7 = 19.", "6", "train"),
    BenchmarkSample("math_seed_003", "math", "What is the larger root of x^2 - 5x + 6 = 0?", "3", "train"),
    BenchmarkSample("math_seed_004", "math", "A triangle has angles 40 and 65 degrees. What is the third angle?", "75", "train"),
    BenchmarkSample("math_seed_005", "math", "If f(x)=2x^2 and x=3, what is f(x)?", "18", "train"),
    BenchmarkSample("math_seed_006", "math", "What is 15 percent of 80?", "12", "train"),
    BenchmarkSample("math_seed_007", "math", "Simplify 4/8 as a decimal.", "0.5", "dev"),
    BenchmarkSample("math_seed_008", "math", "Compute the mean of 4, 8, and 12.", "8", "dev"),
    BenchmarkSample("math_seed_009", "math", "What is the perimeter of a square with side length 9?", "36", "test"),
    BenchmarkSample("math_seed_010", "math", "Evaluate 2^5 - 10.", "22", "test"),
]

HUMANEVAL_SEED_SAMPLES: list[BenchmarkSample] = [
    BenchmarkSample(
        "humaneval_seed_001",
        "humaneval",
        "Write a Python function add(a, b) that returns the sum of a and b.",
        "pass",
        "train",
        metadata={
            "entry_point": "add",
            "canonical_solution": "def add(a, b):\n    return a + b\n",
            "test_code": "assert add(1, 2) == 3\nassert add(-1, 4) == 3\n",
        },
    ),
    BenchmarkSample(
        "humaneval_seed_002",
        "humaneval",
        "Write a Python function is_even(n) that returns True if n is even.",
        "pass",
        "train",
        metadata={
            "entry_point": "is_even",
            "canonical_solution": "def is_even(n):\n    return n % 2 == 0\n",
            "test_code": "assert is_even(2) is True\nassert is_even(3) is False\n",
        },
    ),
    BenchmarkSample(
        "humaneval_seed_003",
        "humaneval",
        "Write a Python function first_item(xs) that returns the first item of a list.",
        "pass",
        "train",
        metadata={
            "entry_point": "first_item",
            "canonical_solution": "def first_item(xs):\n    return xs[0]\n",
            "test_code": "assert first_item([3, 4, 5]) == 3\nassert first_item(['a', 'b']) == 'a'\n",
        },
    ),
    BenchmarkSample(
        "humaneval_seed_004",
        "humaneval",
        "Write a Python function square(n) that returns n multiplied by itself.",
        "pass",
        "train",
        metadata={
            "entry_point": "square",
            "canonical_solution": "def square(n):\n    return n * n\n",
            "test_code": "assert square(5) == 25\nassert square(-3) == 9\n",
        },
    ),
    BenchmarkSample(
        "humaneval_seed_005",
        "humaneval",
        "Write a Python function max_of_two(a, b) that returns the larger value.",
        "pass",
        "dev",
        metadata={
            "entry_point": "max_of_two",
            "canonical_solution": "def max_of_two(a, b):\n    return a if a >= b else b\n",
            "test_code": "assert max_of_two(3, 5) == 5\nassert max_of_two(8, 1) == 8\n",
        },
    ),
    BenchmarkSample(
        "humaneval_seed_006",
        "humaneval",
        "Write a Python function reverse_string(s) that returns s reversed.",
        "pass",
        "dev",
        metadata={
            "entry_point": "reverse_string",
            "canonical_solution": "def reverse_string(s):\n    return s[::-1]\n",
            "test_code": "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\n",
        },
    ),
    BenchmarkSample(
        "humaneval_seed_007",
        "humaneval",
        "Write a Python function count_vowels(s) that counts lowercase vowels.",
        "pass",
        "test",
        metadata={
            "entry_point": "count_vowels",
            "canonical_solution": "def count_vowels(s):\n    return sum(ch in 'aeiou' for ch in s)\n",
            "test_code": "assert count_vowels('banana') == 3\nassert count_vowels('rhythm') == 0\n",
        },
    ),
    BenchmarkSample(
        "humaneval_seed_008",
        "humaneval",
        "Write a Python function factorial(n) for nonnegative n.",
        "pass",
        "test",
        metadata={
            "entry_point": "factorial",
            "canonical_solution": "def factorial(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result\n",
            "test_code": "assert factorial(0) == 1\nassert factorial(5) == 120\n",
        },
    ),
]

FROZEN_SEED_SAMPLES: list[BenchmarkSample] = GSM8K_SEED_SAMPLES + MATH_SEED_SAMPLES + HUMANEVAL_SEED_SAMPLES
VALID_SPLITS = {"train", "dev", "test", "all"}
VALID_TASKS = {"gsm8k", "math", "humaneval", "all"}


def load_frozen_samples(task_type: str = "all", split: str = "all", limit: int | None = None) -> list[BenchmarkSample]:
    if task_type not in VALID_TASKS:
        raise ValueError(f"Unsupported task_type: {task_type}. Expected one of {sorted(VALID_TASKS)}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}. Expected one of {sorted(VALID_SPLITS)}")
    samples = FROZEN_SEED_SAMPLES
    if task_type != "all":
        samples = [sample for sample in samples if sample.task_type == task_type]
    if split != "all":
        samples = [sample for sample in samples if sample.split == split]
    if limit is not None:
        samples = samples[:limit]
    return samples


def summarize_frozen_samples() -> dict[tuple[str, str], int]:
    summary: dict[tuple[str, str], int] = {}
    for sample in FROZEN_SEED_SAMPLES:
        key = (sample.task_type, sample.split)
        summary[key] = summary.get(key, 0) + 1
    return summary
