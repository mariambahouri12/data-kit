"""
Complete RAG pipeline for DataKit AI Assistant.
"""


class RAGPipeline:


    def __init__(
        self,
        retriever,
        prompt_manager,
        llm_client
    ):

        self.retriever = retriever

        self.prompt_manager = prompt_manager

        self.llm_client = llm_client



    def ask(
        self,
        question,
        dataset_context=None
    ):
        """
        Process user question.
        """


        documents = (
            self.retriever
            .retrieve(question)
        )


        prompt = (
            self.prompt_manager
            .build_prompt(
                user_question=question,
                dataset_context=dataset_context,
                retrieved_documents=documents
            )
        )


        response = (
            self.llm_client
            .generate_response(
                prompt
            )
        )


        return {

            "question":
                question,

            "context":
                documents,

            "answer":
                response
        }