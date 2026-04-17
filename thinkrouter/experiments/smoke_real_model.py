from __future__ import annotations

import argparse
import sys

from thinkrouter.app.evaluators import get_evaluator
from thinkrouter.experiments.real_model import check_openai_compatible_config, run_openai_compatible_smoke

DEFAULT_QUERY = "A store had 12 apples and sold 5. How many apples are left?"
DEFAULT_EXPECTED = "7"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or run a real OpenAI-compatible model smoke test.")
    parser.add_argument("--model", default=None, help="Non-mock model id. Defaults to THINKROUTER_STRONG_MODEL.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Prompt to send when --run is set.")
    parser.add_argument("--expected-answer", default=DEFAULT_EXPECTED, help="Expected answer for evaluator reporting.")
    parser.add_argument("--task", default="gsm8k", choices=["gsm8k", "math", "humaneval", "custom"], help="Task type for evaluation.")
    parser.add_argument("--budget", type=int, default=0, help="Thinking budget level to send.")
    parser.add_argument("--run", action="store_true", help="Actually call the configured endpoint. Without this flag, only validate config.")
    args = parser.parse_args()

    check = check_openai_compatible_config(args.model)
    print(f"backend: {check.backend}")
    print(f"model: {check.model_id or '<missing>'}")
    print(f"base_url: {check.base_url or '<missing>'}")
    if check.missing:
        print("missing:")
        for item in check.missing:
            print(f"- {item}")
        sys.exit(2)
    print("config: ok")

    if not args.run:
        print("dry_run: ok. Pass --run to call the model endpoint.")
        return

    response, estimated_cost = run_openai_compatible_smoke(
        model_id=check.model_id,
        query=args.query,
        task_type=args.task,
        expected_answer=args.expected_answer,
        budget=args.budget,
    )
    evaluation = get_evaluator(args.task).evaluate(response.output_text, args.expected_answer)
    print(f"latency_s: {response.latency_s:.3f}")
    print(f"total_tokens: {response.total_tokens}")
    print(f"estimated_cost_usd: {estimated_cost:.8f}")
    print(f"extracted_answer: {evaluation.extracted_answer}")
    print(f"is_correct: {evaluation.is_correct}")
    print("output:")
    print(response.output_text)


if __name__ == "__main__":
    main()