from __future__ import annotations

import numpy as np

from radiology_track_c.segment_gt import prepare_dicom_segment_ground_truth_py
from radiology_track_c.artifacts import build_compressed_figures, build_table3_artifacts
from radiology_track_c.helpers import majority_vote, normalize_method_name, uniform_indices
from radiology_track_c.volumes import LoadedVolume
from slicer_agent_engine.benchmarking.check_segment import check_ras_point


def _make_volume(tmp_path, *, name: str = "case_T1c_bias.nii.gz") -> LoadedVolume:
    data = np.linspace(0.0, 1.0, 16 * 16 * 12, dtype=np.float32).reshape(16, 16, 12)
    display = (data * 255.0).round().astype(np.uint8)
    return LoadedVolume(
        path=tmp_path / name,
        name=name,
        stem="case_T1c_bias",
        data_ijk=data,
        affine_ijk_to_ras=np.eye(4, dtype=float),
        display_uint8=display,
    )


def test_normalize_method_aliases() -> None:
    assert normalize_method_name("figure") == "compressed_figures"
    assert normalize_method_name("table3") == "table3_api"
    assert normalize_method_name("m3d") == "m3d"
    assert normalize_method_name("radfm") == "radfm"


def test_uniform_indices_even_spacing() -> None:
    assert uniform_indices(10, 4) == [0, 3, 6, 9]
    assert uniform_indices(5, 10) == [0, 1, 2, 3, 4]
    assert uniform_indices(0, 3) == []


def test_majority_vote_prefers_first_winner() -> None:
    choice, counts = majority_vote([("t1c", "B"), ("flair", "A"), ("t2", "B"), ("t1", "A")])
    assert choice == "B"
    assert counts == {"A": 2, "B": 2}


def test_build_table3_artifacts_montage(tmp_path) -> None:
    volume = _make_volume(tmp_path)
    bundle = build_table3_artifacts(
        [volume],
        out_dir=tmp_path / "table3",
        case_id="CASE-001",
        planes=["axial"],
        sample_slices=5,
        delivery="montage",
    )
    assert bundle.manifest_path.exists()
    assert len(bundle.artifacts) == 1
    assert bundle.artifacts[0].path.exists()
    assert bundle.manifest[0]["n_tiles"] == 5


def test_build_compressed_figures_caps_total_tiles(tmp_path) -> None:
    volume = _make_volume(tmp_path)
    bundle = build_compressed_figures(
        [volume],
        out_dir=tmp_path / "compressed",
        case_id="CASE-002",
        planes=["axial"],
        slices_per_figure=4,
        max_figures=2,
    )
    total_tiles = sum(int(item["n_tiles"]) for item in bundle.manifest)
    assert total_tiles <= 8
    assert len(bundle.artifacts) <= 2
    for artifact in bundle.artifacts:
        assert artifact.path.exists()


def test_prepare_dicom_segment_ground_truth_uses_frame_geometry(tmp_path) -> None:
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, SegmentationStorage, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SegmentationStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(
        str(tmp_path / "tumor_seg.dcm"),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "SEG"
    ds.SeriesDescription = "Synthetic tumor segmentation"
    ds.Rows = 3
    ds.Columns = 4
    ds.NumberOfFrames = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SegmentationType = "FRACTIONAL"
    ds.SegmentationFractionalType = "PROBABILITY"
    ds.MaximumFractionalValue = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    segment = Dataset()
    segment.SegmentNumber = 1
    segment.SegmentLabel = "Heart"
    segment.SegmentAlgorithmType = "MANUAL"
    segment.SegmentAlgorithmName = "test"
    property_code = Dataset()
    property_code.CodeValue = "T-D0050"
    property_code.CodingSchemeDesignator = "SRT"
    property_code.CodeMeaning = "Tissue"
    segment.SegmentedPropertyTypeCodeSequence = Sequence([property_code])
    ds.SegmentSequence = Sequence([segment])

    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PlaneOrientationSequence = Sequence([orientation])
    measures = Dataset()
    measures.PixelSpacing = [1.0, 1.0]
    measures.SliceThickness = 2.0
    shared.PixelMeasuresSequence = Sequence([measures])
    segment_identification = Dataset()
    segment_identification.ReferencedSegmentNumber = 1
    shared.SegmentIdentificationSequence = Sequence([segment_identification])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])

    per_frame = []
    for z_position in (30.0, 32.0):
        frame = Dataset()
        position = Dataset()
        position.ImagePositionPatient = [10.0, 20.0, z_position]
        frame.PlanePositionSequence = Sequence([position])
        per_frame.append(frame)
    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)

    pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    pixels[:, 1, 2] = 1
    ds.PixelData = pixels.tobytes()
    segment_path = tmp_path / "tumor_seg.dcm"
    ds.save_as(segment_path, enforce_file_format=True)

    gt = prepare_dicom_segment_ground_truth_py(
        case_path=tmp_path,
        out_dir=tmp_path / "prepared",
        segment_series_paths=[segment_path],
    )

    assert gt.has_segment is True
    assert gt.segment_name == "public primary tumor segmentation"
    assert gt.voxel_count == 2
    assert gt.mask_shape_kji == (2, 3, 4)
    assert check_ras_point((-12.0, -21.0, 30.0), gt)["correct"] is True
