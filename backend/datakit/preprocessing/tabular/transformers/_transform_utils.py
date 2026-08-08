
def _positivity_shift(min_val: float, base_shift: float, strict: bool) -> float:
    """Additive shift to ensure that all values become
    strictly positive (strict=True) or non-negative (strict=False)."""
    if strict:
        return base_shift if min_val > 0 else abs(min_val) + base_shift
    return base_shift if min_val >= 0 else abs(min_val) + base_shift

