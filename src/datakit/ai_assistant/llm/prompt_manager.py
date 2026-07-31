"""
Prompt construction module for DataKit AI Assistant.
"""


class PromptManager:


    def __init__(
        self,
        assistant_name="DataKit AI"
    ):

        self.assistant_name = assistant_name



    def build_prompt(
        self,
        user_question,
        dataset_context=None,
        retrieved_documents=None
    ):
        """
        Build final prompt sent to LLM.
        """


        prompt = f"""
You are {self.assistant_name},
an AI assistant specialized in Data Science.

Your role:
- Analyze dataset context
- Explain preprocessing choices
- Recommend ML practices
- Provide justified suggestions


"""


        # Dataset information

        if dataset_context:

            prompt += """
DATASET CONTEXT:
----------------
"""

            prompt += dataset_context



        # RAG documents

        if retrieved_documents:

            prompt += """

KNOWLEDGE BASE:
---------------
"""

            for doc in retrieved_documents:

                prompt += f"""
{doc}

"""



        # User question

        prompt += f"""

USER QUESTION:
--------------
{user_question}


Answer requirements:
- Explain your reasoning
- Mention advantages and limitations
- Give practical recommendation
"""


        return prompt