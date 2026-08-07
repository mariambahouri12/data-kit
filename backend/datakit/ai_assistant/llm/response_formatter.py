"""
Format LLM responses for DataKit UI.
"""

import json



class ResponseFormatter:



    def format_text(
        self,
        response
    ):
        """
        Basic cleaning.
        """

        return response.strip()



    def format_recommendation(
        self,
        response
    ):
        """
        Convert LLM answer into UI-friendly structure.
        """

        return {

            "answer": response,

            "type":
            "recommendation",

            "source":
            "DataKit AI"

        }



    def try_json_format(
        self,
        response
    ):
        """
        Try extracting JSON response.
        """

        try:

            return json.loads(response)

        except:

            return {
                "answer":response
            }