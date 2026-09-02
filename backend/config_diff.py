"""
Compare a stopped run's saved settings against the current ones.

Resuming adopts whatever the configuration says at that moment, so a run
stopped to change a setting and then resumed would process its remaining
patients differently from the ones already done — with nothing in the output
recording the split. This turns that difference into something the operator
sees before deciding.

Only settings that change results are compared; worker counts and paths do
not belong here.
"""

from typing import Any, Dict, List

# Config path -> operator-facing label. Keeping the list explicit (rather
# than diffing the whole document) keeps the dialog about things that alter
# results, not about incidental noise.
_WATCHED = {
    ("skull_stripping", "method"): "Удаление черепа",
    ("skull_stripping", "reference_modality"): "Опорная модальность (удаление черепа)",
    ("skull_stripping", "fallback_method"): "Запасной инструмент удаления черепа",
    ("registration", "registration_type"): "Тип регистрации",
    ("registration", "reference_modality"): "Опорная модальность (регистрация)",
    ("bias_correction", "shrink_factor"): "Коррекция поля: shrink factor",
    ("resampling", "output_resolution"): "Разрешение ресемплинга",
}


def _step_params(config: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    for step in (config or {}).get("steps", []) or []:
        if step.get("name") == step_name:
            return step.get("params") or {}
    return {}


def diff_configs(snapshot: Dict[str, Any],
                 current: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Result-affecting settings that differ, in the operator's words.

    Returns [] when nothing relevant changed, which is the signal to resume
    without asking.
    """
    differences: List[Dict[str, str]] = []

    for (step_name, param), label in _WATCHED.items():
        was = _step_params(snapshot, step_name).get(param)
        now = _step_params(current, step_name).get(param)
        if was != now:
            differences.append({
                "setting": label,
                "was": "не задано" if was is None else str(was),
                "now": "не задано" if now is None else str(now),
            })

    return differences
