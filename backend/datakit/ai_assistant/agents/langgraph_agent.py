"""
High level LangGraph Agent interface.
"""

class LangGraphAgent:

    def __init__(
        self,
        graph
    ):
        self.graph = graph

    def ask(
        self,
        question: str
    ):
        result = (
            self.graph
            .invoke(
                {
                    "question": question,
                    "selected_files": [],
                    "documents": [],
                    "dataset_context": "",
                    "prompt": "",
                    "answer": "",
                    "formatted_answer": "",
                    "structured": None,
                    "recommendation": None,
                    "success": False,
                    "format_success": False,
                    "error": "",
                    "retry_count": 0
                }
            )
        )

        # Use formatted_answer if available, otherwise fallback to answer
        answer = result.get("formatted_answer", "")
        if not answer:
            answer = result.get("answer", "")

        return {
            "answer": answer,
            "structured": result.get("structured", None),
            "recommendation": result.get("recommendation", None),
            "documents": result.get("documents", []),
            "selected_files": result.get("selected_files", []),
            "success": result.get("success", False)
        }