import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import pipeline_manager
from pipeline_manager import PipelineManager


def _write_mapping(output_dir: Path, patients: dict):
    bids_dir = output_dir / "bids_organized"
    bids_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"patients": patients, "output_dir": str(bids_dir), "created_at": "x", "updated_at": "x"}
    (bids_dir / "dataset_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")


class TestGetIncompletePatients:
    def test_returns_incomplete_and_discarded_but_not_untouched_complete_sessions(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "complete",
                        "series": {"t1": {}, "t1c": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230115",
                        "status": "incomplete",
                        "series": {"t1c": {}},
                        "excluded_series": [
                            {
                                "original_path": "/raw/weird", "series_description": "xyz",
                                "slice_count": 20, "detected_modality": None, "reason": "unrecognized",
                            }
                        ],
                    },
                },
            },
            "sub-002": {
                "original_id": "P2",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "discarded",
                        "series": {"t1c": {}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))

        # sub-001/ses-001 is untouched, always-complete, no alternatives — excluded.
        # sub-001/ses-002 is incomplete — included.
        # sub-002/ses-001 is discarded — now included (this is the behavior change).
        assert len(result) == 2
        by_key = {(r["patient_id"], r["session_id"]): r for r in result}
        assert ("sub-001", "ses-002") in by_key
        assert by_key[("sub-001", "ses-002")]["status"] == "incomplete"
        assert ("sub-002", "ses-001") in by_key
        assert by_key[("sub-002", "ses-001")]["status"] == "discarded"
        assert ("sub-001", "ses-001") not in by_key

    def test_includes_complete_sessions_that_still_have_excluded_series(self, tmp_path):
        """A complete session with leftover excluded_series (e.g. a dedup loser
        that lost to the winner but is still a plausible alternative) belongs
        in the review queue too — the doctor may want to reconsider which
        series won, not just fill a gap. A complete session with NOTHING left
        to reconsider (excluded_series == []) stays excluded."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "complete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [
                            {
                                "original_path": "/raw/dup_t1", "series_description": "sT1W_3D_TFE",
                                "slice_count": 200, "detected_modality": "t1", "reason": "lost_deduplication",
                            }
                        ],
                    },
                    "ses-002": {
                        "original_date": "20230201",
                        "status": "complete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))

        assert len(result) == 1
        assert result[0]["session_id"] == "ses-001"
        assert result[0]["status"] == "complete"
        assert len(result[0]["excluded_series"]) == 1

    def test_missing_mapping_file_returns_empty_list(self, tmp_path):
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))
        assert result == []

    def test_missing_field_is_lesion_type_aware(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},  # MS-complete, glio-incomplete (no t1c)
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()

        glio_result = pm.get_incomplete_patients(str(tmp_path), lesion_type="glioblastoma")
        assert glio_result[0]["missing"] == ["t1c"]

        ms_result = pm.get_incomplete_patients(str(tmp_path), lesion_type="multiple_sclerosis")
        assert ms_result[0]["missing"] == []


import shutil


class TestRelabelSeries:
    def _make_dicom_series(self, series_dir: Path, n_files=2):
        series_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (series_dir / f"IM-{i:04d}.dcm").write_bytes(b"fake dicom bytes")
        return series_dir

    def test_relabel_completes_session_and_moves_to_main_tree(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-001" / "ses-001"
        raw_series = self._make_dicom_series(tmp_path / "raw" / "weird_series")

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [
                            {
                                "original_path": str(raw_series),
                                "series_description": "xyz | xyz",
                                "slice_count": 2,
                                "detected_modality": None,
                                "reason": "unrecognized",
                            }
                        ],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        pm = PipelineManager()
        result = pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-001",
            session_id="ses-001",
            original_path=str(raw_series),
            modality="t1c",
        )

        assert result["status"] == "complete"
        # session moved out of _incomplete/
        assert not incomplete_dir.exists()
        assert (bids_dir / "sub-001" / "ses-001" / "anat" / "t1c").exists()
        copied_files = list((bids_dir / "sub-001" / "ses-001" / "anat" / "t1c").glob("*.dcm"))
        assert len(copied_files) == 2

        # dataset_mapping.json updated on disk
        mapping = json.loads((bids_dir / "dataset_mapping.json").read_text())
        session = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        assert session["status"] == "complete"
        assert "t1c" in session["series"]
        assert session["excluded_series"] == []

    def test_relabel_leaves_session_incomplete_if_still_missing_modalities(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-002" / "ses-001"
        raw_series = self._make_dicom_series(tmp_path / "raw" / "weird_series_2")

        _write_mapping(tmp_path, {
            "sub-002": {
                "original_id": "P2",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}},
                        "excluded_series": [
                            {
                                "original_path": str(raw_series), "series_description": "xyz", "slice_count": 2,
                                "detected_modality": None, "reason": "unrecognized",
                            }
                        ],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        pm = PipelineManager()
        result = pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-002",
            session_id="ses-001",
            original_path=str(raw_series),
            modality="t2",
        )

        assert result["status"] == "incomplete"
        # still under _incomplete/ — not yet complete
        assert (bids_dir / "_incomplete" / "sub-002" / "ses-001" / "anat" / "t2").exists()
        assert not (bids_dir / "sub-002").exists()

    def test_relabel_raises_and_leaves_state_untouched_on_partial_copy(self, tmp_path, monkeypatch):
        """Regression test: copy_and_anonymize_series catches per-file exceptions
        internally and returns however many files actually succeeded, which can be
        less than len(source_files). If relabel_series ignored that return value it
        would still mark the session complete and move it into the main tree with
        fewer files than expected — silently defeating the whole point of the
        incomplete-patient review feature. Simulate a partial copy failure by
        monkeypatching copy_and_anonymize_series to report fewer files copied than
        were attempted (simplest reliable way — pydicom's force=True reader is too
        lenient to reliably fail on crafted-garbage fixture bytes)."""
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-003" / "ses-001"
        raw_series = self._make_dicom_series(tmp_path / "raw" / "weird_series_3")

        original_series_entry = {
            "original_path": str(raw_series),
            "series_description": "xyz | xyz",
            "slice_count": 2,
            "detected_modality": None,
            "reason": "unrecognized",
        }
        _write_mapping(tmp_path, {
            "sub-003": {
                "original_id": "P3",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [dict(original_series_entry)],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)
        mapping_file = bids_dir / "dataset_mapping.json"
        mapping_before = mapping_file.read_text()

        # source has 2 files, but only 1 "successfully" copies
        monkeypatch.setattr(pipeline_manager, "copy_and_anonymize_series", lambda *a, **kw: 1)

        pm = PipelineManager()
        with pytest.raises(ValueError, match=r"Copy failed: only 1/2 files copied"):
            pm.relabel_series(
                output_path=str(tmp_path),
                patient_id="sub-003",
                session_id="ses-001",
                original_path=str(raw_series),
                modality="t1c",
            )

        # dataset_mapping.json on disk is byte-for-byte unchanged — no partial state persisted
        assert mapping_file.read_text() == mapping_before
        mapping = json.loads(mapping_file.read_text())
        session = mapping["patients"]["sub-003"]["sessions"]["ses-001"]
        assert session["status"] == "incomplete"
        assert "t1c" not in session["series"]
        assert session["excluded_series"] == [original_series_entry]

        # nothing moved out of _incomplete/
        assert incomplete_dir.exists()
        assert not (bids_dir / "sub-003").exists()

    def test_relabel_that_completes_session_appears_once_then_stops(self, tmp_path):
        """Filling the last missing modality makes excluded_series empty and
        status complete. The FIRST fetch after the action must still include
        it (one-time confirmation the action worked, instead of silently
        vanishing) — but manually_reviewed is a one-shot flag: the act of
        being included for that reason clears it, so a SECOND fetch (e.g.
        doctor reopens the queue later) no longer shows a session with
        nothing left to review."""
        raw_series = self._make_dicom_series(tmp_path / "raw" / "candidate_t1c")

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [
                            {
                                "original_path": str(raw_series), "series_description": "candidate t1c",
                                "slice_count": 2, "detected_modality": "t1c", "reason": "unrecognized",
                            }
                        ],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-001",
            session_id="ses-001",
            original_path=str(raw_series),
            modality="t1c",
        )

        first = pm.get_incomplete_patients(str(tmp_path))
        assert len(first) == 1
        assert first[0]["status"] == "complete"
        assert first[0]["excluded_series"] == []

        second = pm.get_incomplete_patients(str(tmp_path))
        assert second == []

    def test_untouched_complete_session_has_manually_reviewed_false_by_default(self, tmp_path):
        """Regression guard: manually_reviewed must default False for session
        dicts that predate this feature (no key present at all), not True."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "complete",
                        "series": {"t1": {}, "t1c": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        assert pm.get_incomplete_patients(str(tmp_path)) == []


class TestRelabelSeriesInputValidation:
    """Regression tests for a path-traversal finding: patient_id/session_id come
    from the API path (not sanitized by FastAPI beyond excluding '/') and used to
    flow into Path()/shutil.move() construction — must be rejected before that,
    not merely relying on the dataset_mapping.json dict lookup to happen to fail."""

    def test_rejects_path_traversal_patient_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid patient_id"):
            pm.relabel_series(
                output_path=str(tmp_path),
                patient_id="..",
                session_id="ses-001",
                original_path="/whatever",
                modality="t1",
            )

    def test_rejects_path_traversal_session_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid session_id"):
            pm.relabel_series(
                output_path=str(tmp_path),
                patient_id="sub-001",
                session_id="..",
                original_path="/whatever",
                modality="t1",
            )

    def test_rejects_unknown_modality(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid modality"):
            pm.relabel_series(
                output_path=str(tmp_path),
                patient_id="sub-001",
                session_id="ses-001",
                original_path="/whatever",
                modality="not_a_real_modality",
            )


class TestDiscardSession:
    def test_discard_sets_status(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.discard_session(str(tmp_path), "sub-001", "ses-001")

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        assert mapping["patients"]["sub-001"]["sessions"]["ses-001"]["status"] == "discarded"

    def test_discarded_session_still_appears_in_review_queue(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {}, "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.discard_session(str(tmp_path), "sub-001", "ses-001")

        result = pm.get_incomplete_patients(str(tmp_path))
        assert len(result) == 1
        assert result[0]["status"] == "discarded"

    def test_discarded_session_keeps_appearing_on_every_fetch(self, tmp_path):
        """Unlike the one-shot 'became complete' confirmation, a discarded
        session is a deliberate, permanent audit-trail record — it must NOT
        disappear after being read once."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {}, "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.discard_session(str(tmp_path), "sub-001", "ses-001")

        first = pm.get_incomplete_patients(str(tmp_path))
        second = pm.get_incomplete_patients(str(tmp_path))
        assert len(first) == 1
        assert len(second) == 1
        assert first[0]["status"] == second[0]["status"] == "discarded"


class TestDiscardSessionInputValidation:
    """Regression tests mirroring TestRelabelSeriesInputValidation — discard_session
    also takes patient_id/session_id from the API path, so it validates against the
    same BIDS-ID patterns before touching dataset_mapping.json, for consistency with
    relabel_series even though discard only does a dict lookup (no path construction)."""

    def test_rejects_path_traversal_patient_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid patient_id"):
            pm.discard_session(str(tmp_path), "..", "ses-001")

    def test_rejects_path_traversal_session_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid session_id"):
            pm.discard_session(str(tmp_path), "sub-001", "..")


class TestMergeSessions:
    def test_merge_pulls_assigned_and_excluded_series_from_donor(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {"t1": {}, "t2": {}},
                        "excluded_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230201", "status": "incomplete",
                        "series": {
                            "t1c": {"original_path": "/raw/donor_t1c", "series_description": "t1c series", "slice_count": 30},
                        },
                        "excluded_series": [
                            {
                                "original_path": "/raw/donor_extra", "series_description": "extra t2fl",
                                "slice_count": 20, "detected_modality": "t2fl", "reason": "unrecognized",
                            },
                        ],
                    },
                },
            },
        })
        pm = PipelineManager()
        result = pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        assert result == {
            "status": "merged",
            "primary_session_id": "ses-001",
            "donor_session_id": "ses-002",
            "pulled_series": 2,
        }

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        primary = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        excluded_paths = {u["original_path"]: u for u in primary["excluded_series"]}
        assert "/raw/donor_t1c" in excluded_paths
        assert excluded_paths["/raw/donor_t1c"]["detected_modality"] == "t1c"
        assert excluded_paths["/raw/donor_t1c"]["reason"] == "from_other_session"
        assert "/raw/donor_extra" in excluded_paths
        assert excluded_paths["/raw/donor_extra"]["detected_modality"] == "t2fl"
        assert excluded_paths["/raw/donor_extra"]["reason"] == "from_other_session"

    def test_merge_conflicting_modality_offers_both_versions(self, tmp_path):
        """Primary already has a t1; donor also has a t1 — both must be
        offered as alternatives, not silently preferred either way."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {"t1": {"original_path": "/raw/primary_t1", "series_description": "primary t1", "slice_count": 10}},
                        "excluded_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230201", "status": "incomplete",
                        "series": {"t1": {"original_path": "/raw/donor_t1", "series_description": "donor t1", "slice_count": 12}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        primary = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        # primary's own t1 assignment is untouched
        assert primary["series"]["t1"]["original_path"] == "/raw/primary_t1"
        # donor's t1 is offered as an alternative, not silently dropped
        assert len(primary["excluded_series"]) == 1
        assert primary["excluded_series"][0]["original_path"] == "/raw/donor_t1"
        assert primary["excluded_series"][0]["detected_modality"] == "t1"

    def test_merge_marks_donor_session_merged_with_pointer(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {"original_date": "20230101", "status": "incomplete", "series": {}, "excluded_series": []},
                    "ses-002": {"original_date": "20230201", "status": "incomplete", "series": {}, "excluded_series": []},
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        donor = mapping["patients"]["sub-001"]["sessions"]["ses-002"]
        assert donor["status"] == "merged"
        assert donor["merged_into_session_id"] == "ses-001"

    def test_merge_is_idempotent_no_duplicate_candidates(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {"original_date": "20230101", "status": "incomplete", "series": {}, "excluded_series": []},
                    "ses-002": {
                        "original_date": "20230201", "status": "incomplete",
                        "series": {"t1c": {"original_path": "/raw/donor_t1c", "series_description": "t1c", "slice_count": 30}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")
        result_second_call = pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        assert result_second_call["pulled_series"] == 0
        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        primary = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        assert len(primary["excluded_series"]) == 1


class TestMergeSessionsInputValidation:
    def test_rejects_path_traversal_patient_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid patient_id"):
            pm.merge_sessions(str(tmp_path), "..", "ses-001", "ses-002")

    def test_rejects_path_traversal_primary_session_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid primary_session_id"):
            pm.merge_sessions(str(tmp_path), "sub-001", "..", "ses-002")

    def test_rejects_path_traversal_donor_session_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid donor_session_id"):
            pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "..")

    def test_rejects_same_session_as_primary_and_donor(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="must be different"):
            pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-001")


class TestGetIncompletePatientsMerged:
    def test_merged_session_appears_permanently_in_review_queue(self, tmp_path):
        """Like discarded, merged is a permanent audit-trail entry — must
        NOT disappear after being read once (unlike the one-shot
        manually_reviewed 'became complete' confirmation)."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {"original_date": "20230101", "status": "incomplete", "series": {}, "excluded_series": []},
                    "ses-002": {"original_date": "20230201", "status": "incomplete", "series": {}, "excluded_series": []},
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        first = pm.get_incomplete_patients(str(tmp_path))
        second = pm.get_incomplete_patients(str(tmp_path))

        by_key_first = {(r["patient_id"], r["session_id"]): r for r in first}
        by_key_second = {(r["patient_id"], r["session_id"]): r for r in second}
        assert by_key_first[("sub-001", "ses-002")]["status"] == "merged"
        assert by_key_first[("sub-001", "ses-002")]["merged_into_session_id"] == "ses-001"
        assert by_key_second[("sub-001", "ses-002")]["status"] == "merged"


class TestRelabelSeriesReplace:
    def _make_dicom_series(self, series_dir: Path, n_files=2):
        series_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (series_dir / f"IM-{i:04d}.dcm").write_bytes(b"fake dicom bytes")
        return series_dir

    def test_replacing_an_already_filled_modality_bumps_old_series_into_excluded(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-001" / "ses-001"
        new_series = self._make_dicom_series(tmp_path / "raw" / "better_t1")

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {
                            "t1": {
                                "original_path": "/raw/old_t1", "slice_count": 150,
                                "series_description": "old_t1_series",
                            },
                            "t2": {}, "t2fl": {},
                        },
                        "excluded_series": [
                            {
                                "original_path": str(new_series), "series_description": "better_t1_series",
                                "slice_count": 2, "detected_modality": "t1", "reason": "lost_deduplication",
                            }
                        ],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        pm = PipelineManager()
        result = pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-001",
            session_id="ses-001",
            original_path=str(new_series),
            modality="t1",
            # MS's required set is t1/t2/t2fl (no t1c) — matches this fixture.
            # Default lesion_type is "glioblastoma", which requires t1c too and
            # would never be satisfied by this fixture.
            lesion_type="multiple_sclerosis",
        )

        assert result["status"] == "complete"  # t1/t2/t2fl all present now

        mapping = json.loads((bids_dir / "dataset_mapping.json").read_text())
        session = mapping["patients"]["sub-001"]["sessions"]["ses-001"]

        # new series is now the active t1
        assert session["series"]["t1"]["original_path"] == str(new_series)

        # old t1 was NOT discarded — it's back in excluded_series
        assert len(session["excluded_series"]) == 1
        bumped = session["excluded_series"][0]
        assert bumped["original_path"] == "/raw/old_t1"
        assert bumped["detected_modality"] == "t1"
        assert bumped["reason"] == "replaced_by_manual_relabel"

    def test_replacing_with_fewer_files_leaves_no_stale_files_from_old_occupant(self, tmp_path):
        """Regression for a final-review Critical finding: target_dir used
        deterministic _0001.dcm-style naming, so replacing a 5-file series
        with a 2-file one only overwrote files 1-2, leaving files 3-5 from
        the OLD series physically mixed into the modality dir while
        dataset_mapping.json reported the session as cleanly complete."""
        bids_dir = tmp_path / "bids_organized"
        target_dir = bids_dir / "_incomplete" / "sub-001" / "ses-001" / "anat" / "t1"
        target_dir.mkdir(parents=True, exist_ok=True)
        # Simulate the OLD occupant's already-copied files (5 of them).
        for i in range(1, 6):
            (target_dir / f"sub-001_ses-001_T1w_{i:04d}.dcm").write_bytes(b"old series bytes")

        new_series = self._make_dicom_series(tmp_path / "raw" / "shorter_t1", n_files=2)

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {
                            "t1": {
                                "original_path": "/raw/old_t1", "slice_count": 5,
                                "series_description": "old_t1_series",
                            },
                            "t2": {}, "t2fl": {},
                        },
                        "excluded_series": [
                            {
                                "original_path": str(new_series), "series_description": "shorter_t1_series",
                                "slice_count": 2, "detected_modality": "t1", "reason": "lost_deduplication",
                            }
                        ],
                    },
                },
            },
        })

        pm = PipelineManager()
        pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-001",
            session_id="ses-001",
            original_path=str(new_series),
            modality="t1",
            lesion_type="multiple_sclerosis",
        )

        # Session became complete -> moved out of _incomplete/ into the main tree.
        final_dir = bids_dir / "sub-001" / "ses-001" / "anat" / "t1"
        remaining_files = sorted(f.name for f in final_dir.glob("*.dcm"))
        assert len(remaining_files) == 2, (
            f"expected exactly 2 files (the new series), found {remaining_files} "
            f"— stale files from the old occupant were not cleared"
        )
        for f in final_dir.glob("*.dcm"):
            assert f.read_bytes() != b"old series bytes"
