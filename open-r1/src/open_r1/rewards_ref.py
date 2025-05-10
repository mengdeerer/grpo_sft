"""
This module contains the core logic for evaluating mathematical answers.
"""

from typing import List, Union, Dict

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
    completions: List[Dict[str, str]], solution: List[str], **kwargs
) -> List[float]:
    """
    Evaluate mathematical answers and return accuracy scores.

    Args:
        completions: List of model completions, each containing content
        solution: List of ground truth solutions
        **kwargs: Additional keyword arguments

    Returns:
        List of numerical scores (2.0 for correct, 0.0 for incorrect)
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        content = completion["content"]
        model_answer = extract_answer(content)

        if model_answer is None:
            rewards.append(0.0)
            continue

        is_correct = evaluate_math_answer(model_answer, sol)
        rewards.append(2.0 if is_correct else 0.0)

    return rewards


if __name__ == "__main__":
    # Example usage
    test_completion = [{"content": "The answer is \\boxed{x^2 +1         2x }"}]
    test_solution = ["x^2 + 2x + 1"]
    result = math_accuracy_reward(test_completion, test_solution)
    print(f"Test result: {result}")
