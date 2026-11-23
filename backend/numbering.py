"""
Decide the BIDS numbering scope for a run, once, at start.

Stage 01 issues subject ids before anything is uploaded, so the scope has to
be fixed before the pipeline starts. Resolving it here — rather than at
upload, where it used to happen — is also what keeps numbering and upload
pointed at the same Kappa dataset even if `current` in
configs/kappa_datasets.yaml is repointed while the run is in flight.

See docs/superpowers/specs/2026-09-21-bids-numbering-per-dataset-design.md.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from kappa_auth import get_session
from kappa_dataset_resolver import dataset_floor, resolve_or_create
from pipeline_monitor import PREPROCESSING_CONFIG
from preprocessing_version import compute_preprocessing_id, register_version
from utils.bids_allocator import dataset_scope, local_scope, pending_scope, set_floor

logger = logging.getLogger(__name__)


async def scope_for_run(
    run_id: str, lesion_type: str, kappa_session_id: Optional[str],
    intends_upload: bool = False,
) -> Tuple[str, Optional[int], Optional[str]]:
    """(scope, dataset_id, warning) for a run about to start.

    Never raises: a run must start even when Kappa is down. `warning` is a
    user-facing message to surface in the start response when set; None means
    nothing worth telling the operator happened.

    `intends_upload` marks a run that WILL go to Kappa once it is reachable —
    every run started from the web UI, including one started while Kappa is
    down. Such a run must never number into `local:<type>`: those numbers are
    fixed and would collide with whatever the dataset already holds. See
    offline() below for what it uses instead.
    """
    def offline() -> Tuple[str, Optional[int], Optional[str]]:
        """Во что нумеровать, когда Kappa недоступна.

        Сначала пробуем узнать датасет из локального конфига: сеть для этого
        не нужна, а `pending:<run_id>` всегда начинает с sub-001 и потому
        гарантированно сталкивается с любым непустым датасетом — перенос
        падал на UNIQUE по (scope, bids_id). Знание датасета убирает и
        перенос, и столкновение: нумерация просто продолжается с нужного
        места.
        """
        if not intends_upload:
            return local_scope(lesion_type), None, None

        dataset_id = _offline_dataset_id(lesion_type)
        if dataset_id is not None:
            return (
                dataset_scope(dataset_id), dataset_id,
                f"Kappa недоступна: нумеруем в датасет {dataset_id} по конфигу, "
                f"данные уйдут туда же при восстановлении связи",
            )
        return (
            pending_scope(run_id), None,
            "Kappa недоступна: номера будут закреплены за датасетом при выгрузке",
        )

    if not kappa_session_id:
        return offline()

    session = get_session(kappa_session_id)
    if not session:
        return offline()

    preprocessing_id = compute_preprocessing_id(str(PREPROCESSING_CONFIG))
    try:
        dataset_id = await resolve_or_create(
            token=session["kappa_token"], user_id=session["user_id"],
            user_type_id=session["user_type_id"], lesion_type=lesion_type,
            preprocessing_id=preprocessing_id,
        )
    except Exception as exc:
        logger.warning("Kappa unreachable at run start: %s", exc)
        return (pending_scope(run_id), None,
                "Kappa недоступна: номера будут закреплены за датасетом при выгрузке")

    if dataset_id is None:
        return (pending_scope(run_id), None,
                "Датасет в Kappa не создан: будет создан при выгрузке")

    scope = dataset_scope(dataset_id)
    floor = await dataset_floor(
        token=session["kappa_token"], user_id=session["user_id"],
        user_type_id=session["user_type_id"], dataset_id=dataset_id,
    )
    if floor is not None:
        set_floor(scope, floor)
    return scope, dataset_id, None


def pin_preprocessing_id(
    config_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """The preprocessing id to pin on a run that starts now.

    One definition of WHICH file the id comes from, shared by the run-start
    path in app.py and the requeue path, so the two cannot drift apart. The
    file is read here, at start, which is the only moment the answer is
    certainly right: by upload time the operator may have edited it (KI-085).

    Returns None when the config is missing or unreadable — a run must still
    start, and the uploader's own fallback covers a NULL id.
    """
    path = Path(config_path) if config_path is not None else PREPROCESSING_CONFIG
    if not path.is_file():
        logger.warning(
            "Preprocessing config not found, run left unpinned: %s", path,
        )
        return None
    # register_version both computes the id and records the settings behind
    # it, so the label written onto the data stays resolvable afterwards. It
    # never raises and returns None when the config cannot be read.
    return register_version(str(path))


def scope_from_dataset_id(dataset_id: Optional[int], lesion_type: str) -> str:
    """Scope for a resumed/requeued run: no Kappa call, just the parent run's
    already-fixed dataset (or local numbering if it never had one)."""
    if dataset_id is not None:
        return dataset_scope(dataset_id)
    return local_scope(lesion_type)


def _offline_dataset_id(lesion_type: str) -> Optional[int]:
    """Датасет для нумерации, когда Kappa не отвечает. Только локальные данные.

    Пользователя определяем точно, а не угадыванием: Kappa кладёт в userName
    тот же логин, которым входили, поэтому отложенный вход (он хранит логин)
    однозначно указывает на user_id из прошлых сессий. Если очереди входа нет,
    но на машине работал ровно один аккаунт — это тоже факт, а не догадка.
    Во всех остальных случаях возвращаем None и не рискуем: нумеровать в чужой
    датасет хуже, чем отложить решение.
    """
    from kappa_auth import sole_known_user_id, user_id_for_login
    from kappa_dataset_mapping import get_dataset_id

    try:
        import kappa_pending_login
        user_id = user_id_for_login(kappa_pending_login.held_login())
        if user_id is None:
            user_id = sole_known_user_id()
        if user_id is None:
            return None
        return get_dataset_id(user_id, lesion_type, "current")
    except Exception as exc:  # noqa: BLE001 — запуск важнее нумерации
        logger.warning("Не удалось определить датасет офлайн: %s", exc)
        return None
