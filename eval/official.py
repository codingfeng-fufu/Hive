import ast


def mmlongbench(question: dict, prediction: str) -> float:
    from mmlongbench_eval_score import eval_score

    reference = question["answers"][0]
    answer_type = question["answer_format"]
    if answer_type == "List":
        # Parse list literals as data before invoking the upstream evaluator.
        parsed = []
        for value in (reference, prediction):
            if isinstance(value, str) and value.lstrip().startswith("["):
                try:
                    value = ast.literal_eval(value.strip())
                except (ValueError, SyntaxError):
                    return 0.0
            parsed.append(value if isinstance(value, list) else [value])
        reference, prediction = parsed
        if not reference or not prediction:
            return float(reference == prediction)
    elif answer_type == "Float":
        try:
            float(str(reference).strip().lstrip("$").rstrip("%"))
        except ValueError:
            return 0.0
    elif any(
        isinstance(value, str) and value.lstrip().startswith("[")
        for value in (reference, prediction)
    ) and answer_type not in {"Str", "None", "Int"}:
        return 0.0
    return float(eval_score(reference, prediction, answer_type))
