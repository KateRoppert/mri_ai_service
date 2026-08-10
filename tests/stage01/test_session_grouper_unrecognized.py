# tests/stage01/test_session_grouper_unrecognized.py
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_excluded")
SeriesInfo = reorganize_mod.SeriesInfo
SessionGrouper = reorganize_mod.SessionGrouper
SeriesDeduplicator = reorganize_mod.SeriesDeduplicator

LOGGER = logging.getLogger("test_excluded")


def _series(desc, modality, date="20230101"):
    return SeriesInfo(
        original_path=Path(f"/fake/{desc}"),
        patient_id="P1", date=date,
        modality=modality, series_description=desc,
    )


class TestGroupByDateExcluded:
    def test_unrecognized_series_routed_to_session_excluded_list(self):
        recognized = _series("CE_T1-TFE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([recognized, unrecognized])
        assert len(sessions) == 1
        assert "t1c" in sessions[0].series
        assert len(sessions[0].excluded_series) == 1
        assert sessions[0].excluded_series[0].series_description == "some_weird_protocol_47"

    def test_unrecognized_series_grouped_by_own_date(self):
        unrecognized_day1 = _series("weird_a", None, date="20230101")
        unrecognized_day2 = _series("weird_b", None, date="20230115")
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([unrecognized_day1, unrecognized_day2])
        assert len(sessions) == 2
        by_date = {s.date: s for s in sessions}
        assert by_date["20230101"].excluded_series[0].series_description == "weird_a"
        assert by_date["20230115"].excluded_series[0].series_description == "weird_b"


class TestDeduplicateSessionRetainsLosers:
    def test_dedup_losers_added_to_excluded_series(self):
        winner = _series("CE_T1-TFE (3D brain)", "t1c")
        loser = _series("CE_T1-TSE (3D brain)", "t1c")
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([winner, loser])[0]
        assert isinstance(session.series["t1c"], list) and len(session.series["t1c"]) == 2

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert result.series["t1c"] in (winner, loser)  # exactly one is kept
        assert len(result.excluded_series) == 1
        loser_kept = result.excluded_series[0]
        assert loser_kept is not result.series["t1c"]
        assert loser_kept.modality == "t1c"  # detected_modality is just .modality — no new field needed

    def test_deduplicate_session_preserves_prior_excluded_series(self):
        """Modality=None entries from group_by_date must survive dedup untouched,
        alongside any new dedup-loser entries added in the same pass."""
        winner = _series("CE_T1-TFE (3D brain)", "t1c")
        loser = _series("CE_T1-TSE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([winner, loser, unrecognized])[0]

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert len(result.excluded_series) == 2
        descriptions = {s.series_description for s in result.excluded_series}
        assert "some_weird_protocol_47" in descriptions
        assert loser.series_description in descriptions

    def test_no_duplicates_means_no_excluded_series_added(self):
        only = _series("CE_T1-TFE (3D brain)", "t1c")
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([only])[0]

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert result.excluded_series == []
