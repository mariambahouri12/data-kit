"""
Context manager for DataKit AI Assistant.

Combines all available information
before sending to the LLM.
"""



import json



class ContextManager:


    def __init__(
        self,
        dataset_builder,
        preprocessing_context
    ):

        self.dataset_builder = dataset_builder

        self.preprocessing_context = (
            preprocessing_context
        )

        self.dataset_context = None




    def update_dataset(
        self,
        dataframe,
        dataset_name="dataset"
    ):
        """
        Generate and store dataset context.
        """


        self.dataset_context = (
            self.dataset_builder
            .build(
                dataframe,
                dataset_name
            )
        )




    def get_full_context(
        self
    ):
        """
        Return complete AI context.
        """


        return {

            "dataset":
                self.dataset_context,


            "preprocessing":
                (
                    self.preprocessing_context
                    .get_context()
                )

        }




    def to_prompt_format(
        self
    ):
        """
        Convert context into readable text
        for PromptManager.
        """


        context = self.get_full_context()



        return json.dumps(
            context,
            indent=2,
            ensure_ascii=False
        )