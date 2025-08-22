import re
import ast

TRUE_SET = {"true", "yes", "y", "1"}
FALSE_SET = {"false", "no", "n", "0"}

def parse_literal(val: str):
    v = val.strip()
    if v.lower() in TRUE_SET: return True
    if v.lower() in FALSE_SET: return False
    try:
        if re.match(r'^-?\d+$', v): return int(v)
        if re.match(r'^-?\d*\.\d+(e-?\d+)?$', v, re.I): return float(v)
        # try python literal (list, dict, etc.)
        return ast.literal_eval(v)
    except Exception:
        return v  # fallback string

def set_by_path(config: dict, dotted_key: str, value):
    keys = dotted_key.split('.')
    d = config
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def override_config(cfg: dict, pairs):
    for dotted, val in pairs:
        set_by_path(cfg, dotted, parse_literal(val))
    return cfg
