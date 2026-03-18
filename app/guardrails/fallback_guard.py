from typing import List
from langchain_core.documents import Document


class FallbackGuard:
    """
    Ultra-light grounding guardrail.

    Detects weak / ungrounded LLM answers and replaces them
    with a safe enterprise response.

    No extra retrieval or LLM call.
    Zero latency overhead.
    """

    FALLBACK_PHRASES = [
        "not found",
        "no information",
        "cannot determine",
        "not available",
        "not present",
        "do not have access"
    ]

    MIN_ANSWER_WORDS = 20

    SAFE_RESPONSE = (
        "The requested information is not clearly available in the current knowledge base. "
        "Please refine your query or contact the internal support team."
    )

    def is_weak_answer(self, answer: str) -> bool:
        """
        Heuristic grounding detection.
        """

        if not answer:
            return True

        answer_lower = answer.lower()

        # phrase based detection
        for phrase in self.FALLBACK_PHRASES:
            if phrase in answer_lower:
                return True

        # length based detection
        if len(answer.split()) < self.MIN_ANSWER_WORDS:
            return True

        return False

    def apply(self, answer: str, docs: List[Document]) -> str:
        """
        Replace weak answer with safe enterprise response.
        """

        if self.is_weak_answer(answer):
            return self.SAFE_RESPONSE

        return answer