"""Checking a desired modality set, and working out what has to change.

Pure on purpose: dicts in, dicts out, no disk and no config loading. The
API takes the desired FINAL set rather than a list of operations, and this
is what makes that worth doing — the whole set can be checked before a
single file is written, so the mistakes a doctor can actually make (a stale
path from an old screen, a modality this lesion type has no concept of)
cannot half-apply.
"""
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

# A discarded or merged session has already been decided about. Re-opening
# it through this path would silently undo that decision.
EDITABLE_STATUSES = frozenset({"incomplete", "complete"})


class AssignmentError(ValueError):
    """The desired set cannot be applied. Nothing has been written."""


def known_paths(session_data: Mapping[str, Any]) -> set:
    """Every series this session knows about, selected or not."""
    paths = {
        (entry or {}).get("original_path")
        for entry in (session_data.get("series") or {}).values()
    }
    paths |= {
        entry.get("original_path")
        for entry in (session_data.get("excluded_series") or [])
    }
    paths.discard(None)
    paths.discard("")
    return paths


def validate(
    session_data: Mapping[str, Any],
    assignments: Mapping[str, str],
    required: Sequence[str],
) -> None:
    """Raise AssignmentError if the desired set cannot be applied."""
    status = session_data.get("status")
    if status not in EDITABLE_STATUSES:
        raise AssignmentError(
            f"Сессию в состоянии «{status}» редактировать нельзя"
        )

    allowed = set(required)
    for modality in assignments:
        if modality not in allowed:
            raise AssignmentError(
                f"Модальность {modality} не входит в обязательные для этого "
                f"типа поражения: {sorted(allowed)}"
            )

    available = known_paths(session_data)
    for modality, path in assignments.items():
        if path not in available:
            raise AssignmentError(
                f"Серия {path!r} не принадлежит этой сессии"
            )

    seen: Dict[str, str] = {}
    for modality, path in assignments.items():
        if path in seen:
            raise AssignmentError(
                f"Серия {path!r} назначена дважды: {seen[path]} и {modality}"
            )
        seen[path] = modality


@dataclass(frozen=True)
class Changes:
    """What applying the desired set actually requires."""
    assign: Dict[str, str]          # modality -> path to copy in
    clear: Tuple[str, ...]          # modalities to empty
    unchanged: Tuple[str, ...]      # modalities to leave completely alone

    def is_empty(self) -> bool:
        return not self.assign and not self.clear


def plan_changes(
    session_data: Mapping[str, Any], assignments: Mapping[str, str]
) -> Changes:
    """Diff the desired set against the current one."""
    current = {
        modality: (entry or {}).get("original_path")
        for modality, entry in (session_data.get("series") or {}).items()
    }

    assign: Dict[str, str] = {}
    unchanged = []
    for modality, path in assignments.items():
        if current.get(modality) == path:
            unchanged.append(modality)
        else:
            assign[modality] = path

    clear = tuple(sorted(set(current) - set(assignments)))
    return Changes(assign=assign, clear=clear, unchanged=tuple(sorted(unchanged)))
