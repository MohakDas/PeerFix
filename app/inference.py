import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import pipeline
from triage_rules import get_triage

actionability_clf = pipeline(
    "text-classification",
    model="training/actionability_model/final",
    tokenizer="training/actionability_model/final"
)

comment_type_clf = pipeline(
    "text-classification",
    model="training/comment_type_model/final",
    tokenizer="training/comment_type_model/final"
)

def predict_review_comment(text: str):
    actionability_result = actionability_clf(text)[0]

    actionability = actionability_result["label"]
    actionability_score = round(float(actionability_result["score"]), 4)

    if actionability == "non_actionable":
        comment_type = "non_actionable_comment"
        comment_type_score = None

        triage = {
            "priority": "low",
            "response_mode": "acknowledge only",
            "guidance": "This comment is likely non-actionable. Acknowledge it politely unless extra clarification is needed."
        }
    else:
        comment_type_result = comment_type_clf(text)[0]
        comment_type = comment_type_result["label"]
        comment_type_score = round(float(comment_type_result["score"]), 4)

        triage = get_triage(actionability, comment_type)

    return {
        "review_comment": text,
        "actionability": actionability,
        "actionability_score": actionability_score,
        "comment_type": comment_type,
        "comment_type_score": comment_type_score,
        "priority": triage["priority"],
        "response_mode": triage["response_mode"],
        "guidance": triage["guidance"],
    }

if __name__ == "__main__":
    sample_text = "The paper reads well, but it is not really clear what the claimed contribution is."
    result = predict_review_comment(sample_text)
    print(result)