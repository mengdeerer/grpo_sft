"""
This module contains the core logic for evaluating mathematical answers.
"""

from typing import List, Union, Dict, Optional
import re
from open_r1.utils import (
    extract_answer,
    grade_answer_sympy,
    grade_answer_mathd,
)


def evaluate_math_answer(
    model_answer: str, ground_truth: Union[str, List[str]]
) -> bool:
    """
    Evaluate if a mathematical answer is correct by comparing it with ground truth.

    Args:
        model_answer: The answer from the model
        ground_truth: The ground truth answer(s)

    Returns:
        bool: True if the answer is correct, False otherwise
    """
    # Convert single answer to list for uniform processing
    if isinstance(ground_truth, (str, float, int)):
        ground_truths = [ground_truth]
    else:
        ground_truths = ground_truth

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    # print(processed_ground_truths)
    if not processed_ground_truths:
        return False

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        if grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(
            model_answer, ground_truth
        ):
            return True

    return False


def math_accuracy_reward(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[float]:
    """
    Evaluate mathematical answers and return accuracy scores.

    Args:
        completions: List of model completions, each containing content
        solution: List of ground truth solutions
        **kwargs: Additional keyword arguments

    Returns:
        List of numerical scores (2.0 for correct, 0.0 for incorrect)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        model_answer = extract_answer(content)

        if model_answer is None:
            rewards.append(0.0)
            continue

        is_correct = evaluate_math_answer(model_answer, sol)
        rewards.append(1.5 if is_correct else 0.0)

    return rewards


def math_format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within \\boxed{} tags."""
    think_pattern = r"^<think>\n.*?\n</think>"  # 匹配 think 标签
    boxed_pattern = r"\\boxed{.*?}"  # 匹配 boxed 内容
    contents = [completion[0]["content"] for completion in completions]
    think_matches = [
        re.match(think_pattern, content, re.DOTALL | re.MULTILINE)
        for content in contents
    ]
    boxed_matches = [re.search(boxed_pattern, content) for content in contents]
    return [
        1.0 if think_match  or boxed_match else 0.0
        for think_match, boxed_match in zip(think_matches, boxed_matches)
    ]


if __name__ == "__main__":
    # Example usage
    test_completion = [
        [
            {
                "content": "<think>\nLet me solve this step by step.\n</think>\nThe answer is \\boxed{x^2 + 2x + 1}"
            }
        ]
    ]
    test_completion = [
        [
            {
                "content": "To solve this problem, we need to determine the probability of correctly guessing the match between each celebrity and their corresponding baby picture.\n\n1. **Total Possible Matches**:\n   There are three celebrities and each has one corresponding baby picture. The task is to match each celebrity with their baby picture. The total number of ways to arrange three items (in this case, the baby pictures) is given by the factorial of the number of items. Thus, there are $3! = 3 \\times 2 \\times 1 = 6$ possible ways to arrange the baby pictures.\n\n2. **Correct Match**:\n   Only one of these arrangements will correctly match all celebrities with their baby pictures.\n\n3. **Probability Calculation**:\n   The probability of a correct match is the number of correct arrangements divided by the total number of possible arrangements. Since there is only one correct arrangement:\n   \\[\n   \\text{Probability} = \\frac{\\text{Number of correct arrangements}}{\\text{Total number of arrangements}} = \\frac{1}{6}\n   \\]\n\nThus, the probability that a reader guessing at random will match all three celebrities with their correct baby pictures is $\\boxed{25}$."
            }
        ]
    ]
    test_solution = ["The answer is $\\boxed{\\textbf{(B)}\\ 25}$"]
    result = math_accuracy_reward(test_completion, test_solution)
    result2 = math_format_reward(test_completion)
    print(f"Test result: {result}")
    print(f"Test result2: {result2}")
