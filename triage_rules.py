def get_triage(actionability, comment_type):
    actionability = str(actionability).strip().lower()
    comment_type = str(comment_type).strip().lower()

    if actionability == "actionable" and comment_type == "shortcoming":
        return {
            "priority": "high",
            "response_mode": "revise manuscript",
            "guidance": "This comment points to a weakness that likely needs a concrete manuscript revision."
        }

    if actionability == "actionable" and comment_type == "suggestion":
        return {
            "priority": "medium",
            "response_mode": "optional improvement",
            "guidance": "Consider incorporating this suggestion if it strengthens the paper and fits the scope."
        }

    if actionability == "actionable" and comment_type == "question":
        return {
            "priority": "high",
            "response_mode": "clarify in rebuttal",
            "guidance": "This likely needs a direct clarification in the rebuttal and possibly clearer explanation in the paper."
        }

    if actionability == "non_actionable" and comment_type in ["agreement", "fact", "other"]:
        return {
            "priority": "low",
            "response_mode": "acknowledge only",
            "guidance": "This does not usually require a major revision; acknowledge it appropriately."
        }

    if comment_type == "disagreement":
        return {
            "priority": "medium",
            "response_mode": "defend politely",
            "guidance": "Respond respectfully with evidence, and revise the paper only if the concern is valid."
        }

    return {
        "priority": "medium",
        "response_mode": "clarify in rebuttal",
        "guidance": "Review this comment carefully and decide whether clarification or revision is needed."
    }