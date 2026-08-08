"""
Format LLM responses for DataKit UI.
"""

import json
from typing import Optional, Dict, Any


class ResponseFormatter:

    def format_text(self, response: str) -> str:
        """
        Basic cleaning.
        """
        return response.strip()

    def format_recommendation(self, response: str) -> Dict[str, Any]:
        """
        Convert LLM answer into UI-friendly structure.
        """
        return {
            "answer": response,
            "type": "recommendation",
            "source": "DataKit AI"
        }

    def try_json_format(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Try extracting a JSON object from the response.

        FIX (bug critique) : retourne désormais None si le parsing échoue
        ou si le JSON parsé n'est pas un dict, au lieu de fabriquer un
        {"answer": response} de repli. L'ancien comportement faisait que
        *toute* réponse texte normale (non-JSON, la grande majorité des
        réponses LLM) finissait avec la clé "answer" dans `structured`,
        ce qui déclenchait à tort format_recommendation() dans
        formatter_node.py sur presque chaque réponse.
        """
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return None

        return parsed if isinstance(parsed, dict) else None