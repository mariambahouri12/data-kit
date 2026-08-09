from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Query:
    text: str
    embedding: Optional[np.ndarray] = None
    scope: Optional[str] = None
    dataset_fingerprint: Optional[str] = None