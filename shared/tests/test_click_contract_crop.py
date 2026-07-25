import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from slicer_agent_engine.click_contract import (
    bbox_1000_to_full_norm,
    crop_png_to_focus,
    focus_crop_box_for_png,
    point_1000_to_full_norm,
)


class ClickContractCropTests(unittest.TestCase):
    def _make_test_png(self, path: Path) -> None:
        img = Image.new("RGBA", (100, 60), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 5, 79, 54], fill=(180, 180, 180, 255))
        # Small bright artifact that should be ignored by coverage-based cropping.
        draw.rectangle([2, 48, 6, 52], fill=(255, 255, 255, 255))
        img.save(path)

    def test_focus_crop_box_ignores_small_bright_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            self._make_test_png(src)
            crop_box = focus_crop_box_for_png(src)
            self.assertEqual(crop_box, (20, 5, 80, 55))

    def test_crop_png_to_focus_saves_cropped_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            dst = Path(tmp) / "cropped.png"
            self._make_test_png(src)
            out_path, crop_box, full_size = crop_png_to_focus(src, out_path=dst)
            self.assertEqual(out_path, dst.resolve())
            self.assertEqual(crop_box, (20, 5, 80, 55))
            self.assertEqual(full_size, (100, 60))
            with Image.open(dst) as img:
                self.assertEqual(img.size, (60, 50))

    def test_point_and_bbox_mapping_use_cropped_coordinate_space(self) -> None:
        crop_box = (20, 5, 80, 55)
        full_width, full_height = 100, 60
        x_norm, y_norm = point_1000_to_full_norm([500, 500], crop_box=crop_box, full_width=full_width, full_height=full_height)
        self.assertAlmostEqual(x_norm, (20.0 + (500.0 / 999.0) * 59.0) / 99.0, places=6)
        self.assertAlmostEqual(y_norm, (5.0 + (500.0 / 999.0) * 49.0) / 59.0, places=6)

        x1, y1, x2, y2 = bbox_1000_to_full_norm([0, 0, 999, 999], crop_box=crop_box, full_width=full_width, full_height=full_height)
        self.assertAlmostEqual(x1, 20.0 / 99.0, places=6)
        self.assertAlmostEqual(y1, 5.0 / 59.0, places=6)
        self.assertAlmostEqual(x2, 79.0 / 99.0, places=6)
        self.assertAlmostEqual(y2, 54.0 / 59.0, places=6)

if __name__ == "__main__":
    unittest.main()
