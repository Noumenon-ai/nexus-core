from __future__ import annotations

import json
from collections.abc import Iterable



def bullet_list(items: Iterable[str]) -> str:
    values = [item.strip() for item in items if item and item.strip()]
    if not values:
        return ''
    return '\n'.join(f'- {item}' for item in values)



def compact_json(data: dict) -> str:
    return json.dumps(data, separators=(',', ':'), sort_keys=True)
