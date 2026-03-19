from typing import List, Tuple
from langchain_core.documents import Document


class AnswerGuardrail:
    """
    Enterprise Answer Guardrail

    Responsibilities:
    - Detect weak / ungrounded answers
    - Prevent hallucinated or empty responses
    - Return (final_answer, guardrail_triggered)
    """

    def __init__(self):
        self.failure_phrases = [
            "could not find",
            "not available",
            "no information",
            "not present in the documents"
        ]

        self.min_answer_length = 30  # configurable later

    def apply(self, answer: str, docs: List[Document]) -> Tuple[str, bool]:
        """
        Apply guardrail checks to answer.
        Returns:
            (final_answer, guardrail_triggered)
        """

        triggered = False

        # ---- Rule 1: Failure phrases ----
        if any(p in answer.lower() for p in self.failure_phrases):
            triggered = True

        # ---- Rule 2: Very short answer ----
        if len(answer.strip()) < self.min_answer_length:
            triggered = True

        # ---- If triggered → replace with safe answer ----
        if triggered:
            answer = (
                "The available documents do not contain sufficient information "
                "to answer this query reliably."
            )

        return answer, triggered