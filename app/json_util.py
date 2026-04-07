import math
from datetime import date, datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


def to_json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str):
                nk = k
            elif isinstance(k, (datetime, date)):
                nk = k.isoformat()
            elif hasattr(k, "isoformat") and callable(getattr(k, "isoformat")):
                try:
                    nk = k.isoformat()
                except Exception:
                    nk = str(k)
            else:
                nk = str(k)
            out[nk] = to_json_safe(v)
        return out
    if isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(np, "bool8") and isinstance(obj, np.bool8):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    try:
        if isinstance(obj, (float, np.floating)) and pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj
