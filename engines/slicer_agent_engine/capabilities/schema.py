from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

_MISSING = object()
SchemaObject = Dict[str, Any]


def _with_common(
    schema: SchemaObject,
    *,
    description: Optional[str] = None,
    default: Any = _MISSING,
    enum: Optional[Sequence[Any]] = None,
    minimum: Any = _MISSING,
    maximum: Any = _MISSING,
    min_items: Any = _MISSING,
    max_items: Any = _MISSING,
) -> SchemaObject:
    if description:
        schema["description"] = str(description)
    if default is not _MISSING:
        schema["default"] = default
    if enum is not None:
        schema["enum"] = list(enum)
    if minimum is not _MISSING:
        schema["minimum"] = minimum
    if maximum is not _MISSING:
        schema["maximum"] = maximum
    if min_items is not _MISSING:
        schema["minItems"] = int(min_items)
    if max_items is not _MISSING:
        schema["maxItems"] = int(max_items)
    return schema


def string(*, description: Optional[str] = None, default: Any = _MISSING, enum: Optional[Sequence[str]] = None) -> SchemaObject:
    return _with_common({"type": "string"}, description=description, default=default, enum=enum)


def integer(*, description: Optional[str] = None, default: Any = _MISSING, minimum: Any = _MISSING, maximum: Any = _MISSING) -> SchemaObject:
    return _with_common({"type": "integer"}, description=description, default=default, minimum=minimum, maximum=maximum)


def number(*, description: Optional[str] = None, default: Any = _MISSING, minimum: Any = _MISSING, maximum: Any = _MISSING) -> SchemaObject:
    return _with_common({"type": "number"}, description=description, default=default, minimum=minimum, maximum=maximum)


def boolean(*, description: Optional[str] = None, default: Any = _MISSING) -> SchemaObject:
    return _with_common({"type": "boolean"}, description=description, default=default)


def array(
    items: SchemaObject,
    *,
    description: Optional[str] = None,
    default: Any = _MISSING,
    min_items: Any = _MISSING,
    max_items: Any = _MISSING,
) -> SchemaObject:
    return _with_common({"type": "array", "items": items}, description=description, default=default, min_items=min_items, max_items=max_items)


def obj(
    properties: Optional[Mapping[str, SchemaObject]] = None,
    *,
    required: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
) -> SchemaObject:
    schema: SchemaObject = {"type": "object", "properties": dict(properties or {})}
    req = list(required or [])
    if req:
        schema["required"] = req
    else:
        schema["required"] = []
    if description:
        schema["description"] = str(description)
    return schema


def point_1000(*, description: Optional[str] = None) -> SchemaObject:
    return array(
        integer(minimum=0, maximum=999),
        description=description,
        min_items=2,
        max_items=2,
    )


def bbox_1000(*, description: Optional[str] = None) -> SchemaObject:
    return array(
        integer(minimum=0, maximum=999),
        description=description,
        min_items=4,
        max_items=4,
    )


def num_vec(*, description: Optional[str] = None, min_items: Any = _MISSING, max_items: Any = _MISSING) -> SchemaObject:
    return array(number(), description=description, min_items=min_items, max_items=max_items)


def int_vec(*, description: Optional[str] = None, min_items: Any = _MISSING, max_items: Any = _MISSING) -> SchemaObject:
    return array(integer(), description=description, min_items=min_items, max_items=max_items)
