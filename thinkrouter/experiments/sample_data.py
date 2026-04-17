from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    task_type: str
    query: str
    expected_answer: str


DAY1_GSM8K_SAMPLES: list[BenchmarkSample] = [
    BenchmarkSample("gsm8k_day1_001", "gsm8k", "A store had 12 apples and sold 5. How many apples are left?", "7"),
    BenchmarkSample("gsm8k_day1_002", "gsm8k", "Tom has 3 bags with 4 marbles in each bag. How many marbles does he have?", "12"),
    BenchmarkSample("gsm8k_day1_003", "gsm8k", "Lena read 8 pages on Monday and 9 pages on Tuesday. How many pages did she read?", "17"),
    BenchmarkSample("gsm8k_day1_004", "gsm8k", "A taxi ride costs 6 dollars plus 2 dollars per mile. What is the cost for 5 miles?", "16"),
    BenchmarkSample("gsm8k_day1_005", "gsm8k", "There are 24 cookies shared equally by 6 children. How many cookies does each child get?", "4"),
    BenchmarkSample("gsm8k_day1_006", "gsm8k", "A book costs 15 dollars. Mia buys 3 books. How much does she spend?", "45"),
    BenchmarkSample("gsm8k_day1_007", "gsm8k", "Noah had 30 stickers and gave 11 away. How many stickers remain?", "19"),
    BenchmarkSample("gsm8k_day1_008", "gsm8k", "A class has 18 girls and 14 boys. How many students are in the class?", "32"),
    BenchmarkSample("gsm8k_day1_009", "gsm8k", "Sam saves 7 dollars each week for 6 weeks. How much does Sam save?", "42"),
    BenchmarkSample("gsm8k_day1_010", "gsm8k", "A farmer has 5 rows of carrots with 9 carrots in each row. How many carrots are there?", "45"),
    BenchmarkSample("gsm8k_day1_011", "gsm8k", "If 40 pencils are packed into boxes of 8, how many boxes are needed?", "5"),
    BenchmarkSample("gsm8k_day1_012", "gsm8k", "A movie starts at 3 PM and lasts 2 hours. What hour does it end?", "5"),
    BenchmarkSample("gsm8k_day1_013", "gsm8k", "Nina has 50 cents and buys candy for 35 cents. How many cents are left?", "15"),
    BenchmarkSample("gsm8k_day1_014", "gsm8k", "Four friends each bring 6 balloons. How many balloons do they bring together?", "24"),
    BenchmarkSample("gsm8k_day1_015", "gsm8k", "A rope is 20 meters long. It is cut into 4 equal pieces. How long is each piece?", "5"),
    BenchmarkSample("gsm8k_day1_016", "gsm8k", "Ella scored 10 points in the first game and twice as many in the second. What is her total?", "30"),
    BenchmarkSample("gsm8k_day1_017", "gsm8k", "There are 9 birds on a tree. 4 more birds arrive, then 3 fly away. How many birds remain?", "10"),
    BenchmarkSample("gsm8k_day1_018", "gsm8k", "A pack has 12 cards. Ben buys 2 packs and gives away 5 cards. How many cards does he keep?", "19"),
    BenchmarkSample("gsm8k_day1_019", "gsm8k", "A recipe uses 3 cups of flour per cake. How many cups are needed for 7 cakes?", "21"),
    BenchmarkSample("gsm8k_day1_020", "gsm8k", "Jill ran 2 miles each day for 5 days and then 3 miles on Saturday. How many miles total?", "13"),
]


def load_day1_samples(limit: int = 20) -> list[BenchmarkSample]:
    return DAY1_GSM8K_SAMPLES[:limit]
