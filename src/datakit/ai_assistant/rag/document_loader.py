"""
Load selected documents for RAG.
"""

from pathlib import Path


class DocumentLoader:


    def __init__(
        self,
        knowledge_base_path
    ):

        self.base_path = Path(
            knowledge_base_path
        )



    def load(
        self,
        selected_files
    ):

        documents = []


        for file_path in self.base_path.rglob("*.md"):


            if file_path.name in selected_files:


                content = file_path.read_text(
                    encoding="utf-8"
                )


                documents.append(
                    {
                        "content": content,

                        "source":
                            file_path.name,

                        "category":
                            file_path.parent.name
                    }
                )


        return documents