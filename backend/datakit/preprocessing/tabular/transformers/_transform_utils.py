


def _positivity_shift(min_val: float, base_shift: float, strict: bool) -> float:
    """Décalage additif pour garantir que toutes les valeurs deviennent
    strictement positives (strict=True) ou positives ou nulles (strict=False)."""
    if strict:
        return base_shift if min_val > 0 else abs(min_val) + base_shift
    return base_shift if min_val >= 0 else abs(min_val) + base_shift
