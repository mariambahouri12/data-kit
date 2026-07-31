"""
Preprocessing context module.

Tracks preprocessing operations
performed during the workflow.
"""



class PreprocessingContext:


    def __init__(self):

        self.operations = []



    def add_operation(
        self,
        operation_name,
        columns,
        parameters=None
    ):
        """
        Register preprocessing action.
        """


        operation = {

            "operation":
                operation_name,


            "columns":
                columns,


            "parameters":
                parameters or {}

        }


        self.operations.append(
            operation
        )



    def get_context(self):
        """
        Return preprocessing history.
        """

        return {

            "operations":
                self.operations
        }



    def clear(self):

        self.operations = []