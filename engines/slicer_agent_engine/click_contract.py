from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple




def normalize_point_1000(point: Sequence[int | float]) -> list[int]:
    if len(point) != 2:
        raise ValueError("point_1000 must be [x,y] in integer 0..999 coordinates")
    vals: list[int] = []
    for v in point:
        try:
            iv = int(round(float(v)))
        except Exception as e:
            raise ValueError(f"point value {v!r} is not numeric: {e}") from e
        vals.append(max(0, min(999, iv)))
    return vals


def point_norm_from_1000(point: Sequence[int | float]) -> Tuple[float, float]:
    x, y = normalize_point_1000(point)
    return float(x) / 999.0, float(y) / 999.0


def bbox_corners_1000(bbox: Sequence[int | float]) -> dict[str, list[int]]:
    x1, y1, x2, y2 = normalize_bbox_1000(bbox)
    return {
        "top_left": [x1, y1],
        "top_right": [x2, y1],
        "bottom_right": [x2, y2],
        "bottom_left": [x1, y2],
        "center": list(bbox_center_1000([x1, y1, x2, y2])),
    }


def normalize_bbox_1000(bbox: Sequence[int | float]) -> list[int]:
    if len(bbox) != 4:
        raise ValueError("seed_bbox_1000 must be [x_min, y_min, x_max, y_max] in integer 0..999 coordinates")
    vals: list[int] = []
    for v in bbox:
        try:
            iv = int(round(float(v)))
        except Exception as e:
            raise ValueError(f"bbox value {v!r} is not numeric: {e}") from e
        vals.append(max(0, min(999, iv)))
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def bbox_center_1000(bbox: Sequence[int | float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = normalize_bbox_1000(bbox)
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def bbox_center_norm(bbox: Sequence[int | float]) -> Tuple[float, float]:
    cx, cy = bbox_center_1000(bbox)
    return float(cx) / 999.0, float(cy) / 999.0


def scale_bbox_1000_to_pixels(bbox: Sequence[int | float], *, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = normalize_bbox_1000(bbox)
    w = max(int(width), 1)
    h = max(int(height), 1)

    def sx(v: int) -> int:
        return int(round((float(v) / 999.0) * float(max(w - 1, 1))))

    def sy(v: int) -> int:
        return int(round((float(v) / 999.0) * float(max(h - 1, 1))))

    return sx(x1), sy(y1), sx(x2), sy(y2)


def focus_crop_box_for_png(
    in_path: str | Path,
    *,
    alpha_threshold: int = 8,
    brightness_threshold: int = 8,
    min_active_fraction: float = 0.10,
) -> Tuple[int, int, int, int]:
    from PIL import Image

    src = Path(in_path).expanduser().resolve()
    with Image.open(src) as im:
        img = im.convert("RGBA")
        width, height = img.size
        pixels = img.load()

        col_counts = [0] * width
        row_counts = [0] * height
        min_x, min_y = width, height
        max_x, max_y = -1, -1

        for y in range(height):
            active_in_row = 0
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a <= alpha_threshold:
                    continue
                if max(r, g, b) <= brightness_threshold:
                    continue
                active_in_row += 1
                col_counts[x] += 1
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
            row_counts[y] = active_in_row

        if max_x < min_x or max_y < min_y:
            return (0, 0, width, height)

        min_col_active = max(1, int(round(float(height) * float(min_active_fraction))))
        min_row_active = max(1, int(round(float(width) * float(min_active_fraction))))
        xs = [idx for idx, count in enumerate(col_counts) if count >= min_col_active]
        ys = [idx for idx, count in enumerate(row_counts) if count >= min_row_active]

        if xs and ys:
            left, right = xs[0], xs[-1] + 1
            top, bottom = ys[0], ys[-1] + 1
        else:
            left, top, right, bottom = min_x, min_y, max_x + 1, max_y + 1

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        return (left, top, right, bottom)


def crop_png_to_focus(
    in_path: str | Path,
    *,
    out_path: Optional[str | Path] = None,
    alpha_threshold: int = 8,
    brightness_threshold: int = 8,
    min_active_fraction: float = 0.10,
) -> tuple[Path, tuple[int, int, int, int], tuple[int, int]]:
    from PIL import Image

    src = Path(in_path).expanduser().resolve()
    dst = Path(out_path).expanduser().resolve() if out_path is not None else src
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        img = im.copy()
        full_size = img.size
        crop_box = focus_crop_box_for_png(
            src,
            alpha_threshold=alpha_threshold,
            brightness_threshold=brightness_threshold,
            min_active_fraction=min_active_fraction,
        )
        cropped = img.crop(crop_box)
        save_kwargs = {}
        if img.format:
            save_kwargs["format"] = img.format
        cropped.save(dst, **save_kwargs)

    return dst, crop_box, full_size


def point_1000_to_full_norm(
    point_1000: Sequence[int | float],
    *,
    crop_box: Sequence[int | float],
    full_width: int,
    full_height: int,
) -> Tuple[float, float]:
    x1000, y1000 = normalize_point_1000(point_1000)
    left, top, right, bottom = [int(round(float(v))) for v in crop_box]
    crop_width = max(int(right - left), 1)
    crop_height = max(int(bottom - top), 1)
    x_px = float(left) + (float(x1000) / 999.0) * float(max(crop_width - 1, 1))
    y_px = float(top) + (float(y1000) / 999.0) * float(max(crop_height - 1, 1))
    x_norm = x_px / float(max(int(full_width) - 1, 1))
    y_norm = y_px / float(max(int(full_height) - 1, 1))
    return max(0.0, min(1.0, x_norm)), max(0.0, min(1.0, y_norm))


def bbox_1000_to_full_norm(
    bbox_1000: Sequence[int | float],
    *,
    crop_box: Sequence[int | float],
    full_width: int,
    full_height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = normalize_bbox_1000(bbox_1000)
    fx1, fy1 = point_1000_to_full_norm([x1, y1], crop_box=crop_box, full_width=full_width, full_height=full_height)
    fx2, fy2 = point_1000_to_full_norm([x2, y2], crop_box=crop_box, full_width=full_width, full_height=full_height)
    return fx1, fy1, fx2, fy2


def annotate_png_with_corner_text(
    in_path: str | Path,
    *,
    lines: Sequence[str],
    out_path: str | Path | None = None,
    corner: str = "top_left",
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    src = Path(in_path).expanduser().resolve()
    dst = Path(out_path).expanduser().resolve() if out_path is not None else src
    dst.parent.mkdir(parents=True, exist_ok=True)

    normalized_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not normalized_lines:
        if dst != src:
            dst.write_bytes(src.read_bytes())
        return dst

    with Image.open(src) as im:
        img = im.convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        pad = max(4, min(img.width, img.height) // 96)
        line_gap = max(2, pad // 2)

        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in normalized_lines]
        text_width = max((box[2] - box[0]) for box in line_boxes)
        text_height = sum((box[3] - box[1]) for box in line_boxes) + line_gap * max(len(line_boxes) - 1, 0)
        box_width = text_width + pad * 2
        box_height = text_height + pad * 2

        corner_key = str(corner or "top_left").strip().lower()
        if corner_key in {"top_left", "tl"}:
            x0 = pad
            y0 = pad
        elif corner_key in {"top_right", "tr"}:
            x0 = max(pad, img.width - box_width - pad)
            y0 = pad
        elif corner_key in {"bottom_left", "bl"}:
            x0 = pad
            y0 = max(pad, img.height - box_height - pad)
        elif corner_key in {"bottom_right", "br"}:
            x0 = max(pad, img.width - box_width - pad)
            y0 = max(pad, img.height - box_height - pad)
        else:
            raise ValueError(f"Unsupported corner: {corner!r}")

        draw.rectangle([x0, y0, x0 + box_width, y0 + box_height], fill=(0, 0, 0, 180))
        y = y0 + pad
        for line, box in zip(normalized_lines, line_boxes):
            draw.text((x0 + pad, y), line, fill=(255, 255, 255, 255), font=font)
            y += (box[3] - box[1]) + line_gap

        img.convert("RGB").save(dst)
    return dst


def annotate_png_with_bbox(
    in_path: str | Path,
    *,
    bbox_1000: Sequence[int | float],
    out_path: str | Path,
    draw_crosshair: bool = True,
    label: str | None = None,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    src = Path(in_path).expanduser().resolve()
    dst = Path(out_path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        img = im.convert("RGBA")
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = scale_bbox_1000_to_pixels(bbox_1000, width=img.width, height=img.height)
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        line_w = max(2, img.width // 256)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80, 255), width=line_w)
        if draw_crosshair:
            arm = max(10, min(img.width, img.height) // 18)
            draw.line([(cx - arm, cy), (cx + arm, cy)], fill=(255, 255, 0, 255), width=line_w)
            draw.line([(cx, cy - arm), (cx, cy + arm)], fill=(255, 255, 0, 255), width=line_w)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 0, 255))
        if label:
            font = ImageFont.load_default()
            tx = max(0, min(x1, img.width - 1))
            ty = max(0, y1 - 18)
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            pad = 4
            draw.rectangle([tx, ty, tx + (r - l) + pad * 2, ty + (b - t) + pad * 2], fill=(0, 0, 0, 180))
            draw.text((tx + pad, ty + pad), label, fill=(255, 255, 255, 255), font=font)
        img.convert("RGB").save(dst)
    return dst


def annotate_png_with_point(
    in_path: str | Path,
    *,
    point_1000: Sequence[int | float],
    out_path: str | Path,
    label: str | None = None,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    src = Path(in_path).expanduser().resolve()
    dst = Path(out_path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    x1000, y1000 = normalize_point_1000(point_1000)
    with Image.open(src) as im:
        img = im.convert("RGBA")
        draw = ImageDraw.Draw(img)
        x = int(round((float(x1000) / 999.0) * float(max(img.width - 1, 1))))
        y = int(round((float(y1000) / 999.0) * float(max(img.height - 1, 1))))
        line_w = max(2, img.width // 256)
        arm = max(10, min(img.width, img.height) // 18)
        draw.line([(x - arm, y), (x + arm, y)], fill=(255, 255, 0, 255), width=line_w)
        draw.line([(x, y - arm), (x, y + arm)], fill=(255, 255, 0, 255), width=line_w)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 0, 255))
        if label:
            font = ImageFont.load_default()
            tx = max(0, min(x + 6, img.width - 1))
            ty = max(0, y - 18)
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            pad = 4
            draw.rectangle([tx, ty, tx + (r - l) + pad * 2, ty + (b - t) + pad * 2], fill=(0, 0, 0, 180))
            draw.text((tx + pad, ty + pad), label, fill=(255, 255, 255, 255), font=font)
        img.convert("RGB").save(dst)
    return dst
