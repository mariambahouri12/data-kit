"""
Ollama client for DataKit AI Assistant.

Responsible for communication with local Ollama models.
"""

import ollama


class OllamaClient:
    """
    Client to interact with local Ollama LLM.
    """

    def __init__(
        self,
        model_name="mistral",
        host="http://localhost:11434"
    ):
        """
        Initialize Ollama client.

        Args:
            model_name:
                Name of local model installed in Ollama.
            
            host:
                Ollama server address.
        """

        self.model_name = model_name
        self.host = host


    def check_connection(self):
        """
        Verify that Ollama is running
        and model is available.
        """

        try:

            models = ollama.list()

            available_models = [
                model["name"]
                for model in models["models"]
            ]

            if self.model_name not in available_models:
                return {
                    "status": False,
                    "message":
                    f"Model {self.model_name} not found"
                }


            return {
                "status": True,
                "message":
                "Ollama connection successful"
            }


        except Exception as e:

            return {
                "status": False,
                "message": str(e)
            }



    def generate_response(
        self,
        prompt,
        temperature=0.2
    ):
        """
        Generate response from Ollama model.

        Args:
            prompt:
                Complete prompt sent to LLM
            
            temperature:
                Creativity level

        Returns:
            Generated text
        """

        try:

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ],
                options={
                    "temperature":temperature
                }
            )


            return response["message"]["content"]


        except Exception as e:

            return (
                "Error while generating response: "
                + str(e)
            )



    def stream_response(
        self,
        prompt
    ):
        """
        Stream LLM answer token by token.
        Useful for Streamlit chat interface.
        """

        try:

            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ],
                stream=True
            )


            for chunk in stream:

                yield chunk["message"]["content"]


        except Exception as e:

            yield f"Error: {str(e)}"