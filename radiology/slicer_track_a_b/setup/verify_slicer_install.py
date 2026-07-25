"""Validate the 3D Slicer modules and extensions used by MedFlowBench."""

import importlib
import json
import os
import sys
from pathlib import Path

import qt
import slicer


def _csv(name):
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _application_value(name):
    value = getattr(slicer.app, name, "")
    return value() if callable(value) else value


required_extensions = _csv("SLICER_EXTENSIONS_CSV")
optional_extensions = _csv("SLICER_OPTIONAL_EXTENSIONS_CSV")
extension_manager = slicer.app.extensionsManagerModel()

report = {
    "application_version": str(slicer.app.applicationVersion),
    "revision": str(slicer.app.repositoryRevision),
    "executable": str(_application_value("applicationFilePath")),
    "settings": {
        "user": str(slicer.app.slicerUserSettingsFilePath),
        "revision": str(slicer.app.slicerRevisionUserSettingsFilePath),
    },
    "required_extensions": {},
    "extension_sources": {},
    "optional_extensions": {},
    "core_modules": {},
    "feature_checks": {},
    "errors": [],
}

for name in required_extensions:
    installed = bool(extension_manager.isExtensionInstalled(name))
    built_in = False
    if name == "WebServer":
        try:
            importlib.import_module("WebServer")
            built_in = True
        except Exception:
            built_in = False
    available = installed or built_in
    report["required_extensions"][name] = available
    report["extension_sources"][name] = (
        "extension" if installed else "application" if built_in else "unavailable"
    )
    if not available:
        report["errors"].append("Required extension is not installed: " + name)

for name in optional_extensions:
    report["optional_extensions"][name] = bool(extension_manager.isExtensionInstalled(name))

for name in (
    "brainsfit",
    "brainsresample",
    "dicom",
    "markups",
    "segmentations",
    "segmenteditor",
    "transforms",
    "volumes",
):
    available = getattr(slicer.modules, name, None) is not None
    report["core_modules"][name] = available
    if not available:
        report["errors"].append("Required core module is unavailable: " + name)

opengl_context_available = False
try:
    opengl_context = qt.QOpenGLContext()
    opengl_context_available = bool(
        opengl_context.create() and opengl_context.isValid()
    )
except Exception:
    opengl_context_available = False
report["feature_checks"]["opengl_context"] = opengl_context_available
if not opengl_context_available:
    report["errors"].append(
        "Slicer could not create an OpenGL context; check DISPLAY, Xvfb GLX, "
        "and Mesa runtime libraries"
    )


def check_import(label, module_name, required):
    try:
        importlib.import_module(module_name)
        report["feature_checks"][label] = True
    except Exception as exc:
        report["feature_checks"][label] = False
        if required:
            report["errors"].append(f"{label} failed: {exc}")


check_import("webserver_python_module", "WebServer", "WebServer" in required_extensions)
check_import(
    "dicom_segmentation_plugin",
    "DICOMSegmentationPlugin",
    "QuantitativeReporting" in required_extensions,
)

pet_suv_cli_available = getattr(slicer.modules, "suvfactorcalculator", None) is not None
report["feature_checks"]["pet_suv_cli"] = pet_suv_cli_available
if "QuantitativeReporting" in required_extensions and not pet_suv_cli_available:
    report["errors"].append(
        "The PETDICOMExtension dependency is installed but its "
        "SUVFactorCalculator CLI is unavailable"
    )

elastix_available = getattr(slicer.modules, "elastix", None) is not None
report["feature_checks"]["elastix_module"] = elastix_available
if "SlicerElastix" in required_extensions and not elastix_available:
    report["errors"].append("SlicerElastix is installed but the Elastix module is unavailable")

local_threshold_available = False
try:
    segment_editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
    effect_names = [str(name) for name in segment_editor_widget.availableEffectNames()]
    report["feature_checks"]["segment_editor_effects"] = effect_names
    local_threshold_available = "Local Threshold" in effect_names
except Exception:
    local_threshold_available = False
report["feature_checks"]["local_threshold_effect"] = local_threshold_available
if "SegmentEditorExtraEffects" in required_extensions and not local_threshold_available:
    report["errors"].append(
        "SegmentEditorExtraEffects is installed but the Local Threshold effect is unavailable"
    )

monai_available = getattr(slicer.modules, "monailabel", None) is not None
report["feature_checks"]["monai_label_module"] = monai_available

output_path = os.environ.get("MEDFLOW_SLICER_VERIFY_OUTPUT", "").strip()
rendered = json.dumps(report, indent=2, sort_keys=True)
print(rendered)
if output_path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered + "\n", encoding="utf-8")

if report["errors"]:
    print("[slicer-verify] validation failed", file=sys.stderr)
    sys.exit(1)
print("[slicer-verify] validation passed")
sys.exit(0)
