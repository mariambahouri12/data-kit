# datakit/ai_assistant/rag/rag_pipeline.py

"""
Complete RAG pipeline for DataKit AI Assistant.
"""

from typing import Dict, Any, Optional


class RAGPipeline:
    """
    End-to-end RAG pipeline: retrieve → prompt → generate.
    """

    def __init__(
        self,
        retriever,
        prompt_manager,
        llm_client
    ):
        self.retriever = retriever
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client
        self.is_ready = False

    def initialize(self, documents: list = None) -> bool:
        """
        Initialize RAG pipeline with knowledge base.
        """
        try:
            if documents:
                self.retriever.initialize(documents)
            self.is_ready = True
            return True
        except Exception as e:
            print(f"Failed to initialize RAG pipeline: {e}")
            return False

    def ask(
        self,
        question: str,
        dataset_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user question.
        """
        try:
            # Retrieve relevant documents
            documents = self.retriever.retrieve(question)
            
            # Build prompt
            prompt = self.prompt_manager.build_prompt(
                user_question=question,
                dataset_context=dataset_context,
                retrieved_documents=documents
            )
            
            # Generate response
            response = self.llm_client.generate_response(prompt)
            
            return {
                "question": question,
                "context": documents,
                "answer": response,
                "success": True
            }
            
        except Exception as e:
            return {
                "question": question,
                "context": [],
                "answer": f"Erreur: {str(e)}",
                "success": False
            }

    def stream_ask(
        self,
        question: str,
        dataset_context: Optional[str] = None
    ):
        """
        Process user question with streaming response.
        """
        try:
            documents = self.retriever.retrieve(question)
            prompt = self.prompt_manager.build_prompt(
                user_question=question,
                dataset_context=dataset_context,
                retrieved_documents=documents
            )
            
            for chunk in self.llm_client.stream_response(prompt):
                yield chunk
                
        except Exception as e:
            yield f"Erreur: {str(e)}"