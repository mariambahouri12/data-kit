"""
High level LangGraph Agent interface.
"""


class LangGraphAgent:
    """
    High-level interface used to interact with the compiled LangGraph.
    """

    def __init__(self, graph):
        self.graph = graph

    def ask(self, question: str) -> dict:
        """
        Execute the LangGraph workflow for a user question.
        """

        initial_state = {
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
            "retry_count": 0,
        }

        result = self.graph.invoke(initial_state)

        # Prefer formatted answer when formatting succeeded.
        answer = result.get("formatted_answer", "")

        if not answer:
            answer = result.get("answer", "")

        return {
            "answer": answer,
            "structured": result.get("structured"),
            "recommendation": result.get("recommendation"),
            "documents": result.get("documents", []),
            "selected_files": result.get("selected_files", []),
            "success": result.get("success", False),
            "error": result.get("error", ""),
        }