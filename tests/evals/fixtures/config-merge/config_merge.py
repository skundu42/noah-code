def merge_config(base: dict, override: dict) -> dict:
    result = dict(base)
    result.update(override)
    return result
