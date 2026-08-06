import logging
import sys
import importlib.util
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_unrecognized")
SeriesInfo = reorganize_mod.SeriesInfo
SessionGrouper = reorganize_mod.SessionGrouper
SeriesDeduplicator = reorganize_mod.SeriesDeduplicator

LOGGER = logging.getLogger("test_unrecognized")


def _series(desc, modality, date="20230101"):
    return SeriesInfo(
        original_path=Path(f"/fake/{desc}"),
        patient_id="P1", date=date,
        modality=modality, series_description=desc,
    )


class TestGroupByDateUnrecognized:
    def test_unrecognized_series_routed_to_session_unrecognized_list(self):
        recognized = _series("CE_T1-TFE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([recognized, unrecognized])
        assert len(sessions) == 1
        assert "t1c" in sessions[0].series
        assert len(sessions[0].unrecognized_series) == 1
        assert sessions[0].unrecognized_series[0].series_description == "some_weird_protocol_47"

    def test_unrecognized_series_grouped_by_own_date(self):
        unrecognized_day1 = _series("weird_a", None, date="20230101")
        unrecognized_day2 = _series("weird_b", None, date="20230115")
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([unrecognized_day1, unrecognized_day2])
        assert len(sessions) == 2
        by_date = {s.date: s for s in sessions}
        assert by_date["20230101"].unrecognized_series[0].series_description == "weird_a"
        assert by_date["20230115"].unrecognized_series[0].series_description == "weird_b"


class TestDeduplicateSessionCarriesUnrecognized:
    def test_deduplicate_session_preserves_unrecognized_series(self):
        recognized = _series("CE_T1-TFE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([recognized, unrecognized])[0]

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert len(result.unrecognized_series) == 1
        assert result.unrecognized_series[0].series_description == "some_weird_protocol_47"
