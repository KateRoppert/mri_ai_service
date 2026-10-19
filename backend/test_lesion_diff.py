import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import lesion_diff
from lesion_diff import compare_labeled_masks, cached_compare_labeled_masks


def _save_labeled(path: Path, labels: dict, shape=(20, 20, 20)):
    """labels: {label_int: (slice_x, slice_y, slice_z)}"""
    data = np.zeros(shape, dtype=np.int16)
    for label, (sx, sy, sz) in labels.items():
        data[sx, sy, sz] = label
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_lesion_with_no_prior_overlap_is_new(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 3), slice(0, 3), slice(0, 3))})       # far away
    _save_labeled(curr_path, {1: (slice(10, 13), slice(10, 13), slice(10, 13))})  # new lesion

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1)

    assert result["new_count"] == 1
    statuses = {l["status"] for l in result["lesions"]}
    assert "new" in statuses


def test_prior_lesion_absent_in_current_is_resolved(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 3), slice(0, 3), slice(0, 3))})
    _save_labeled(curr_path, {})  # empty — lesion resolved

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1)

    assert result["resolved_count"] == 1
    resolved = [l for l in result["lesions"] if l["status"] == "resolved"][0]
    assert resolved["volume_cm3"] == 0.0
    assert resolved["previous_volume_cm3"] > 0.0


def test_matched_lesion_growing_by_relative_threshold(tmp_path):
    # Small lesion: 3x3x3=27 voxels (0.027 cm3) -> grows to 5x5x5=125 voxels
    # (0.125 cm3). Relative growth = (0.125-0.027)/0.027 ≈ 3.6 >> 0.20.
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(5, 8), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(5, 10), slice(5, 10), slice(5, 10))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["growing_count"] == 1
    assert result["new_count"] == 0


def test_matched_lesion_growing_by_absolute_threshold_with_small_relative_growth(tmp_path):
    # Large lesion: 10x10x10=1000 voxels (1.0 cm3) -> grows to 1100 voxels by
    # adding a 10x10x1 slab (100 voxels = 0.1 cm3). Relative growth = 10% < 20%,
    # but absolute growth 0.1 cm3 >= 0.03 cm3 threshold -> still "growing".
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})
    _save_labeled(curr_path, {1: (slice(0, 10), slice(0, 10), slice(0, 11))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["growing_count"] == 1
    assert result["stable_count"] == 0


def test_matched_lesion_with_negligible_change_is_stable(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})
    _save_labeled(curr_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["stable_count"] == 1
    assert result["growing_count"] == 0


def test_sub_threshold_components_excluded_from_both_sides(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    # A single 1mm voxel = 0.001 cm3, far below the 5mm3 (0.005 cm3) floor.
    _save_labeled(prev_path, {1: (slice(0, 1), slice(0, 1), slice(0, 1))})
    _save_labeled(curr_path, {1: (slice(15, 16), slice(15, 16), slice(15, 16))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1, min_lesion_volume_mm3=5.0)

    assert result["lesions"] == []
    assert result["new_count"] == 0
    assert result["resolved_count"] == 0


def test_current_lesion_overlapping_two_prior_lesions_sums_previous_volume(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    # Two separate prior lesions, each 3x3x3=27 voxels (0.027 cm3), adjacent
    # but not touching (gap at x=8). A merged current lesion spans both.
    _save_labeled(prev_path, {
        1: (slice(0, 3), slice(0, 3), slice(0, 3)),
        2: (slice(8, 11), slice(0, 3), slice(0, 3)),
    })
    _save_labeled(curr_path, {1: (slice(0, 11), slice(0, 3), slice(0, 3))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    matched = [l for l in result["lesions"] if l["status"] in ("growing", "stable")][0]
    assert matched["previous_volume_cm3"] == pytest.approx(0.027 + 0.027, abs=1e-4)


def test_shape_mismatch_raises_value_error(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr_different_shape.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.int16), np.eye(4)), str(prev_path))
    nib.save(nib.Nifti1Image(np.zeros((22, 22, 22), dtype=np.int16), np.eye(4)), str(curr_path))

    with pytest.raises(ValueError):
        compare_labeled_masks(prev_path, curr_path)


# --- Caching (cached_compare_labeled_masks) ---------------------------------
# The diff of two label masks is deterministic given the files and params, so it
# must be computed once and reused. These lock in that behaviour and the
# mtime-based invalidation (an expert re-saving an edited mask must recompute).

def _counting_compare(monkeypatch):
    """Wrap compare_labeled_masks with a call counter, via the module global."""
    calls = {"n": 0}
    real = lesion_diff.compare_labeled_masks

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(lesion_diff, "compare_labeled_masks", counting)
    return calls


def test_cached_diff_computes_once_then_reuses(tmp_path, monkeypatch):
    prev = tmp_path / "prev.nii.gz"
    curr = tmp_path / "curr.nii.gz"
    _save_labeled(prev, {1: (slice(10, 13), slice(10, 13), slice(10, 13))})
    _save_labeled(curr, {1: (slice(10, 14), slice(10, 14), slice(10, 14))})
    cache = tmp_path / "cache"

    calls = _counting_compare(monkeypatch)
    r1 = cached_compare_labeled_masks(prev, curr, cache, dilation_voxels=1)
    r2 = cached_compare_labeled_masks(prev, curr, cache, dilation_voxels=1)

    assert calls["n"] == 1              # second call served from cache
    assert r1 == r2
    assert r1 == compare_labeled_masks(prev, curr, dilation_voxels=1)  # same as uncached


def test_cached_diff_invalidates_when_mask_changes(tmp_path, monkeypatch):
    prev = tmp_path / "prev.nii.gz"
    curr = tmp_path / "curr.nii.gz"
    _save_labeled(prev, {1: (slice(10, 13), slice(10, 13), slice(10, 13))})
    _save_labeled(curr, {1: (slice(10, 13), slice(10, 13), slice(10, 13))})
    cache = tmp_path / "cache"

    calls = _counting_compare(monkeypatch)
    cached_compare_labeled_masks(prev, curr, cache, dilation_voxels=1)

    # Expert edits the current mask: new content and a distinct mtime.
    _save_labeled(curr, {1: (slice(10, 15), slice(10, 15), slice(10, 15))})
    os.utime(curr, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))

    cached_compare_labeled_masks(prev, curr, cache, dilation_voxels=1)
    assert calls["n"] == 2              # recomputed after the mask changed


def test_cached_diff_key_depends_on_params(tmp_path, monkeypatch):
    prev = tmp_path / "prev.nii.gz"
    curr = tmp_path / "curr.nii.gz"
    _save_labeled(prev, {1: (slice(10, 13), slice(10, 13), slice(10, 13))})
    _save_labeled(curr, {1: (slice(10, 14), slice(10, 14), slice(10, 14))})
    cache = tmp_path / "cache"

    calls = _counting_compare(monkeypatch)
    cached_compare_labeled_masks(prev, curr, cache, dilation_voxels=1)
    cached_compare_labeled_masks(prev, curr, cache, dilation_voxels=2)  # different param
    assert calls["n"] == 2              # different params must not share a cache entry


# --- coregistration_dice (guardrail signal) ---------------------------------
# The longitudinal diff is only meaningful when the two sessions are co-registered.
# coregistration_dice measures brain overlap (background is 0 on skull-stripped
# preprocessed images) so the endpoint can flag pairs that are not aligned instead
# of reporting the misalignment as lesion activity.

def _save_brain(path, occupied):
    vol = np.zeros((30, 30, 30), dtype=np.float32)
    vol[occupied] = 100.0
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))


def test_coregistration_dice_aligned_is_high(tmp_path):
    a, b = tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"
    occ = (slice(5, 25), slice(5, 25), slice(5, 25))
    _save_brain(a, occ)
    _save_brain(b, occ)
    assert lesion_diff.coregistration_dice(a, b) > 0.95


def test_coregistration_dice_shifted_is_low(tmp_path):
    a, b = tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"
    _save_brain(a, (slice(2, 12), slice(5, 25), slice(5, 25)))    # lower half
    _save_brain(b, (slice(18, 28), slice(5, 25), slice(5, 25)))   # shifted up, no overlap
    assert lesion_diff.coregistration_dice(a, b) < 0.3


def test_coregistration_dice_shape_mismatch_returns_zero(tmp_path):
    a, b = tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((30, 30, 30), np.float32), np.eye(4)), str(a))
    nib.save(nib.Nifti1Image(np.ones((20, 20, 20), np.float32), np.eye(4)), str(b))
    assert lesion_diff.coregistration_dice(a, b) == 0.0


def test_coregistration_dice_missing_file_returns_none(tmp_path):
    a = tmp_path / "a.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((10, 10, 10), np.float32), np.eye(4)), str(a))
    assert lesion_diff.coregistration_dice(a, tmp_path / "missing.nii.gz") is None


# ── Раскрытие пограничных вердиктов (KI-074) ───────────────────────────────
# dilation_voxels не поглощает дрожание модели — его там нет (KI-037 измерен:
# маски побитово идентичны на одном входе). Он принимает клинические решения,
# и в 11 парах сессий из 11 отчётные счёта меняются вместе с ним
# (research/ms_diff_tolerance/). Поэтому диф обязан показывать, НАСКОЛЬКО
# близко вердикт к перевороту, а не только сам вердикт.


def _only(result, status):
    return [l for l in result["lesions"] if l["status"] == status]


def test_margin_mm_reports_the_gap_to_the_counterpart(tmp_path):
    """У «нового» очага зазор — расстояние до ближайшей прежней ткани."""
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    # prev занимает x 2..4, curr — x 12..14 → зазор 12 - 4 = 8 мм
    _save_labeled(prev_path, {1: (slice(2, 5), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(12, 15), slice(5, 8), slice(5, 8))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    new = _only(result, "new")
    assert len(new) == 1
    assert new[0]["margin_mm"] == pytest.approx(8.0)


def test_a_directly_overlapping_lesion_has_zero_margin(tmp_path):
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    box = (slice(5, 8), slice(5, 8), slice(5, 8))
    _save_labeled(prev_path, {1: box})
    _save_labeled(curr_path, {1: box})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    assert result["lesions"][0]["margin_mm"] == pytest.approx(0.0)
    assert result["lesions"][0]["borderline"] is False


def test_borderline_flags_a_verdict_one_voxel_from_flipping(tmp_path):
    """Зазор 4 мм при допуске 3 — вердикт «новый» переворачивается от сдвига
    порога на один воксель, и это должно быть видно."""
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    # prev до x=8, curr от x=12 → зазор 4 мм
    _save_labeled(prev_path, {1: (slice(6, 9), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(12, 15), slice(5, 8), slice(5, 8))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    new = _only(result, "new")
    assert len(new) == 1, "при допуске 3 этот очаг ещё не сопоставлен"
    assert new[0]["margin_mm"] == pytest.approx(4.0)
    assert new[0]["borderline"] is True
    assert result["borderline_count"] >= 1


def test_a_match_one_voxel_from_becoming_new_is_borderline(tmp_path):
    """Обратная сторона: сопоставленный через зазор 2 мм при допуске 3."""
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    # prev до x=10, curr от x=12 → зазор 2 мм; расширение на 3 достаёт
    _save_labeled(prev_path, {1: (slice(8, 11), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(12, 15), slice(5, 8), slice(5, 8))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    assert _only(result, "new") == [], "зазор 2 мм должен быть сопоставлен"
    matched = [l for l in result["lesions"] if l["status"] in ("stable", "growing")]
    assert len(matched) == 1
    assert matched[0]["margin_mm"] == pytest.approx(2.0)
    assert matched[0]["borderline"] is True


def test_a_verdict_far_from_the_threshold_is_not_borderline(tmp_path):
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(2, 5), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(12, 15), slice(5, 8), slice(5, 8))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    assert _only(result, "new")[0]["borderline"] is False
    assert result["borderline_count"] == 0


def test_resolved_lesion_margin_measures_toward_the_current_session(tmp_path):
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    # prev имеет два очага; второй далеко от всего в curr
    _save_labeled(prev_path, {
        1: (slice(5, 8), slice(5, 8), slice(5, 8)),
        2: (slice(5, 8), slice(15, 18), slice(5, 8)),
    })
    _save_labeled(curr_path, {1: (slice(5, 8), slice(5, 8), slice(5, 8))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    resolved = _only(result, "resolved")
    assert len(resolved) == 1
    # prev очаг 2 занимает y 15..17, ближайшая ткань curr — y 7 → зазор 8 мм
    assert resolved[0]["margin_mm"] == pytest.approx(8.0)


def test_matched_prev_count_discloses_swallowed_lesions(tmp_path):
    """Один текущий очаг может проглотить несколько прежних.

    Расширение собирает ВСЕ попавшие прежние метки и суммирует их объёмы, из-за
    чего реально растущий очаг может выглядеть стабильным, а «исчезнувшие»
    пропадают из отчёта. Сколько именно проглочено — факт, который надо
    показывать.
    """
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {
        1: (slice(9, 12), slice(2, 5), slice(5, 8)),
        2: (slice(9, 12), slice(8, 11), slice(5, 8)),
    })
    _save_labeled(curr_path, {1: (slice(12, 15), slice(5, 8), slice(5, 8))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    matched = [l for l in result["lesions"] if l["status"] in ("stable", "growing")]
    assert len(matched) == 1
    assert matched[0]["matched_prev_count"] == 2
    assert _only(result, "resolved") == [], "оба прежних поглощены, не 'исчезли'"


def test_a_new_lesion_matched_nothing(tmp_path):
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(2, 5), slice(2, 5), slice(2, 5))})
    _save_labeled(curr_path, {1: (slice(12, 15), slice(12, 15), slice(12, 15))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)

    assert _only(result, "new")[0]["matched_prev_count"] == 0


def test_cache_is_invalidated_by_the_schema_version(tmp_path, monkeypatch):
    """Ключ кэша включает параметры, но не форму вывода.

    Без версии схемы запись, сделанная до появления margin_mm, вернулась бы
    без него — и потребитель получил бы вердикт без раскрытия, молча.
    """
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(5, 8), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(5, 8), slice(5, 8), slice(5, 8))})
    params = {"dilation_voxels": 3}

    key_now = lesion_diff._diff_cache_key(prev_path, curr_path, params)
    monkeypatch.setattr(lesion_diff, "DIFF_SCHEMA_VERSION", "test-other-version")
    key_other = lesion_diff._diff_cache_key(prev_path, curr_path, params)

    assert key_now != key_other


def test_the_rule_uses_chebyshev_not_euclidean_distance(tmp_path):
    """Зазор в мм и порог в вокселях — разные метрики, и путать их нельзя.

    binary_dilation структурой 3x3x3, применённая N раз, расширяет на N в
    метрике L-infinity. Смещение (1,3,0) даёт евклидовы 3.16 мм, но по
    Чебышёву 3 — очаг СОПОСТАВЛЕН при допуске 3, хотя «3.16 > 3» подсказывает
    обратное. Ровно на этом ошибся живой разбор данных P000010.
    """
    prev_path, curr_path = tmp_path / "prev.nii.gz", tmp_path / "curr.nii.gz"
    # prev занимает x<=11, y<=7;  curr — x>=12, y>=10. Минимальное смещение
    # между их вокселами: по x 12-11=1, по y 10-7=3 → (1,3,0).
    # Евклид sqrt(1+9) = 3.16 мм, Чебышёв = 3.
    _save_labeled(prev_path, {1: (slice(9, 12), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(12, 15), slice(10, 13), slice(5, 8))})

    at3 = compare_labeled_masks(prev_path, curr_path, dilation_voxels=3)
    assert _only(at3, "new") == [], "по Чебышёву 3 <= 3 — очаг сопоставлен"
    matched = [l for l in at3["lesions"] if l["status"] in ("stable", "growing")]
    assert matched[0]["margin_mm"] == pytest.approx(3.16, abs=0.01)
    assert matched[0]["borderline"] is True

    # А вот при допуске 2 он действительно становится новым — то есть вердикт
    # держится именно на значении порога.
    at2 = compare_labeled_masks(prev_path, curr_path, dilation_voxels=2)
    assert len(_only(at2, "new")) == 1
    assert _only(at2, "new")[0]["borderline"] is True, "|3 - 2| = 1 — на грани"
