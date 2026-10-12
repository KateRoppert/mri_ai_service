"""Reading a session's DICOM identity off disk — shared, so two readers agree.

patient_registry.register_patient upserts by study_hash. If the local
registrar and the Kappa uploader computed that hash differently, the upload
would create a SECOND row instead of filling in the first, and the
longitudinal view would show one session twice. So the walk is shared and
pinned by test.

The two consumers do NOT select the same way, and that difference is kept:
the hash skips a metadata file whose PatientID or StudyInstanceUID is empty
and keeps looking, while the registry fields come from the first readable
file as they are. Unifying the selection would have changed the hash — the
dedup key for every upload — for the sake of tidiness.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import session_identity as si


def _session(root: Path, sub="sub-001", ses="ses-001", modalities=("t1",)):
    """One session's preprocessed files, laid out as stage 05 leaves them."""
    files = []
    for m in modalities:
        d = root / "preprocessed" / sub / ses / "anat"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{sub}_{ses}_{m}.nii.gz"
        f.write_bytes(b"x")
        files.append(f)
    return files


def _meta(root: Path, ident, sub="sub-001", ses="ses-001", modality="t1",
          name="0001.json"):
    d = root / "metadata" / sub / ses / "anat" / modality
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({"identification": ident}), encoding="utf-8")


def test_hash_is_sha256_of_patient_and_study(tmp_path):
    files = _session(tmp_path)
    _meta(tmp_path, {"PatientID": "P1", "StudyInstanceUID": "1.2.3"})

    import hashlib
    expected = hashlib.sha256(b"P1:1.2.3").hexdigest()[:16]

    assert si.study_hash(tmp_path, files) == expected


def test_the_uploader_and_the_registrar_agree(tmp_path):
    """The one that matters: register_patient upserts by this value, so a
    disagreement silently doubles every session in the registry."""
    from kappa_uploader import KappaUploader
    import yaml

    files = _session(tmp_path)
    _meta(tmp_path, {"PatientID": "P1", "StudyInstanceUID": "1.2.3"})

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"preprocessing": {"steps": []}}), encoding="utf-8")
    up = KappaUploader(
        run_id="r1", output_path=str(tmp_path), token="t", user_id=1,
        user_type_id=1, lesion_type="multiple_sclerosis",
        preprocessing_config_path=str(cfg),
    )

    assert up._compute_study_hash({"preprocessed": files}) == si.study_hash(
        tmp_path, files)


def test_the_hash_skips_a_file_missing_its_identifiers(tmp_path):
    """Preserved from the original walk: an empty PatientID does not end the
    search, it moves on to the next file."""
    files = _session(tmp_path, modalities=("t1", "t2"))
    _meta(tmp_path, {"PatientID": "", "StudyInstanceUID": ""}, modality="t1")
    _meta(tmp_path, {"PatientID": "P1", "StudyInstanceUID": "1.2.3"}, modality="t2")

    import hashlib
    assert si.study_hash(tmp_path, files) == hashlib.sha256(
        b"P1:1.2.3").hexdigest()[:16]


def test_registry_fields_take_the_first_readable_file(tmp_path):
    """Also preserved: these are taken as they are, empty or not."""
    files = _session(tmp_path, modalities=("t1", "t2"))
    _meta(tmp_path, {"PatientID": "", "StudyDate": "20260101"}, modality="t1")
    _meta(tmp_path, {"PatientID": "P1", "StudyDate": "20270101"}, modality="t2")

    fields = si.registry_fields(tmp_path, files)

    assert fields["PatientID"] == ""
    assert fields["StudyDate"] == "20260101"


def test_a_patient_name_split_into_characters_is_joined(tmp_path):
    """Some exports write PatientName as a list of characters."""
    files = _session(tmp_path)
    _meta(tmp_path, {"PatientID": "P1", "PatientName": ["И", "в", "а", "н"]})

    assert si.registry_fields(tmp_path, files)["PatientName"] == "Иван"


def test_no_metadata_directory_is_not_an_error(tmp_path):
    """metadata/ is written by stage 01 and can be absent on an old run."""
    files = _session(tmp_path)

    assert si.study_hash(tmp_path, files) is None
    assert si.registry_fields(tmp_path, files) == {}


def test_unreadable_json_does_not_stop_the_walk(tmp_path):
    files = _session(tmp_path, modalities=("t1", "t2"))
    d = tmp_path / "metadata" / "sub-001" / "ses-001" / "anat" / "t1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    _meta(tmp_path, {"PatientID": "P1", "StudyInstanceUID": "1.2.3"}, modality="t2")

    assert si.study_hash(tmp_path, files) is not None


def test_sessions_are_discovered_from_preprocessed_output(tmp_path):
    _session(tmp_path, sub="sub-001", ses="ses-001", modalities=("t1", "t2"))
    _session(tmp_path, sub="sub-002", ses="ses-003", modalities=("t1",))

    found = si.discover_preprocessed_sessions(tmp_path)

    assert set(found) == {"sub-001_ses-001", "sub-002_ses-003"}
    assert len(found["sub-001_ses-001"]) == 2


def test_no_preprocessed_output_yields_nothing(tmp_path):
    assert si.discover_preprocessed_sessions(tmp_path) == {}
