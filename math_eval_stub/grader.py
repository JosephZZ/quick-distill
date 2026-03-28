def math_equal(pred, target, timeout=False):
    """Simple stub for math_equal - compares normalized strings"""
    if pred is None or target is None:
        return False
    # Normalize: remove spaces, lower case
    pred_norm = str(pred).strip().replace(" ", "").lower()
    target_norm = str(target).strip().replace(" ", "").lower()
    return pred_norm == target_norm
