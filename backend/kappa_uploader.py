"""
Загрузка итоговых результатов пайплайна в Kappa.
Один датасет (по lesion_type + preprocessing_id).
Одна сущность = одна сессия (4 preprocessed NIfTI + 1 маска сегментации).
dsEntityInfo = quality reports + volume report + lobar report + параметры запуска.
Дедупликация по study_hash (PatientID + StudyInstanceUID из DICOM-метаданных).
"""
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kappa_client import (
    get_dataset_entities,
    list_user_datasets,
    update_entity_status,
    upload_entity,
)
import session_identity
from kappa_dataset_mapping import get_dataset_id
from preprocessing_version import compute_preprocessing_id
from utils.config_loader import load_lesion_type_config

logger = logging.getLogger(__name__)

# Same fallback as pipeline_manager when lesion_types.yaml has no entry.
_DEFAULT_REQUIRED_MODALITIES = ["t1", "t1c", "t2", "t2fl"]


def session_file_paths(session_data: Dict[str, Any]) -> List[Path]:
    """The files that make up one session's Kappa entity.

    Both the upload and the replacement need this set. Keeping one
    definition is what stops them drifting apart — a report added to the
    entity has to reach a replaced entity too.

    The native-space mask is deliberately excluded: it lives in the
    patient's own geometry and is not part of what the dataset holds.
    """
    paths = list(session_data["preprocessed"])
    paths.extend(m for m in session_data["masks"] if "_native_" not in m.name)
    if session_data.get("lesion_labels_mask"):
        paths.append(session_data["lesion_labels_mask"])
    return paths


def _entity_info(entity: Dict[str, Any]) -> Dict[str, Any]:
    info = entity.get("dsEntityInfo") or {}
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except ValueError:
            info = {}
    return info if isinstance(info, dict) else {}


class KappaUnavailable(Exception):
    """The dataset listing could not be read (KI-069).

    Every duplicate and name-clash check reads that listing; treating a failed
    read as an empty dataset switched them off — attempts re-POSTed delivered
    sessions (dataset 353: 29 entities for 13 names), or flagged them all as
    name clashes. A failed read ends the attempt as a network failure instead.
    """


class KappaUploader:
    """
    Загружает итоговые результаты пайплайна в Kappa.
    Один экземпляр на один запуск пайплайна (run_id).
    """

    def __init__(
        self,
        run_id: str,
        output_path: str,
        token: str,
        user_id: int,
        user_type_id: int,
        lesion_type: str,
        preprocessing_config_path: str,
        dataset_id: Optional[int] = None,
        superseding_sessions=frozenset(),
    ):
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.token = token
        self.user_id = user_id
        self.user_type_id = user_type_id
        self.lesion_type = lesion_type
        self.preprocessing_config_path = preprocessing_config_path
        # Fixed by backend/numbering.py at run start, when the run has a Kappa
        # session. None for CLI runs, and for runs started before this existed.
        self.dataset_id = dataset_id

        # Sessions rebuilt from a corrected modality set. They hash the same
        # as what Kappa holds (study_hash covers the DICOM study, not the
        # images), so without this they would pass as duplicates and the
        # correction would never arrive.
        self.superseding_sessions = set(superseding_sessions or ())

        # Вычисляем preprocessing_id
        self.preprocessing_id = compute_preprocessing_id(preprocessing_config_path)
        logger.info(
            "KappaUploader: run=%s, lesion=%s, prep_id=%s",
            run_id, lesion_type, self.preprocessing_id,
        )

    async def upload_results(self) -> Dict[str, Any]:
        """
        Основной метод: загрузить все сессии в Kappa.
        Возвращает отчёт о загрузке.
        """
        # 1. Получаем dataset_id
        dataset_id = await self._resolve_dataset_id()
        if dataset_id is None:
            return {"error": "Failed to resolve dataset_id"}

        # Numbers issued while Kappa was unreachable (pending:<run_id>) belong
        # to this dataset now that it exists — move them before anything else
        # touches the dataset, so a name-clash check below sees the true state.
        self._bind_pending_scope(dataset_id)

        # 2. Находим все сессии. Знаменатель — всё, что вышло из этапа 01, а
        # загружается только готовое: сессия, потерянная на 05/06, не должна
        # ни уехать в Kappa без маски, ни выпасть из счёта.
        sessions = self._discover_sessions()
        universe = self._session_universe(sessions)
        if not universe:
            logger.warning("No sessions found in %s", self.output_path)
            return {"uploaded": 0, "sessions": []}

        results = []
        ready_sessions: Dict[str, Dict[str, Any]] = {}
        for session_key in universe:
            ready, reason = self._readiness(session_key, sessions.get(session_key))
            if ready:
                ready_sessions[session_key] = sessions[session_key]
                continue
            logger.warning("Not uploading %s: %s", session_key, reason)
            results.append({
                "session": session_key,
                "success": False,
                "error": "not_processed",
                "message": reason,
            })
        sessions = ready_sessions

        if not sessions:
            return {
                "dataset_id": dataset_id,
                "uploaded": 0,
                "total": len(universe),
                "sessions": results,
            }

        # 3. Проверяем дубликаты. Hashes and names come from one listing per
        # attempt, so they cannot disagree; without a listing nothing is
        # decided (KI-069).
        self._attempt_entities = None
        try:
            existing_hashes = await self._get_existing_study_hashes(dataset_id)
            # Same sub-NNN, different study: two machines numbered into this
            # dataset independently while both were offline. Cannot renumber —
            # the id is already in the file names — so refuse loudly instead of
            # silently pairing a number with the wrong patient's files.
            existing_names = await self._get_existing_entity_names(dataset_id)
        except KappaUnavailable:
            logger.warning(
                "Dataset %d listing unavailable — nothing uploaded, the "
                "attempt is retried (run=%s)", dataset_id, self.run_id,
            )
            return {
                "dataset_id": dataset_id,
                "error": "entities_unavailable",
                "uploaded": 0,
                "total": len(universe),
                "sessions": results,
            }

        # 4. Загружаем каждую готовую сессию
        for session_key, session_data in sessions.items():
            self._reconciled_name = None
            # Вычисляем study_hash для дедупликации
            study_hash = self._compute_study_hash(session_data)

            if session_key in existing_names and study_hash not in existing_hashes:
                logger.warning(
                    "Name clash in dataset %d: %s already exists with a "
                    "different study — numbers were issued twice for this "
                    "dataset", dataset_id, session_key,
                )
                results.append({
                    "session": session_key,
                    "success": False,
                    "error": "name_clash",
                    "message": (
                        f"В датасете уже есть {session_key} с другими данными "
                        f"— номер выдан дважды, загрузка пропущена"
                    ),
                })
                continue

            if (study_hash and study_hash in existing_hashes
                    and session_key in self.superseding_sessions):
                logger.warning(
                    "Session %s supersedes what dataset %d holds — needs a "
                    "human decision, not a silent skip", session_key, dataset_id,
                )
                results.append({
                    "session": session_key,
                    "success": False,
                    "error": "supersedes",
                    "message": (
                        f"{session_key} пересчитан с исправленным набором "
                        f"модальностей — в Kappa лежит прежняя версия"
                    ),
                })
                continue

            if study_hash and study_hash in existing_hashes:
                # KI-035: the entity may predate _readiness and lack a
                # modality and the mask. A complete rerun completes it
                # instead of passing as a duplicate.
                entity, state = self._entity_completeness(study_hash)
                if state in ("incomplete", "incomplete_expert") \
                        and entity.get("dsEntityName") != session_key:
                    # Files carry the local name; written into an entity
                    # named otherwise, it would look incomplete again on
                    # every attempt and be rewritten forever.
                    state = "incomplete_expert"
                if state == "incomplete":
                    results.append(await self._complete_entity(
                        dataset_id, session_key, session_data, entity))
                    continue
                if state == "incomplete_expert":
                    logger.warning(
                        "Session %s would complete entity %s, which carries an "
                        "expert mask — needs a human decision",
                        session_key, entity.get("dsEntityId"),
                    )
                    results.append({
                        "session": session_key,
                        "success": False,
                        "error": "supersedes",
                        "message": (
                            f"В Kappa лежит неполная версия {session_key} с "
                            f"экспертной маской — дополнение заменит снимки, "
                            f"на которых она нарисована"
                        ),
                    })
                    continue

                logger.warning(
                    "Duplicate detected, skipping upload: %s (study_hash=%s already in dataset %d)",
                    session_key, study_hash, dataset_id,
                )

                # POST не делаем (сущность уже в Каппе — иначе задвоим), но
                # убеждаемся, что она есть в локальных таблицах. Это и есть
                # путь восстановления запусков, чьи сущности осели в Каппе
                # после 504, но не попали в локальный реестр.
                reconciled_id = await self._reconcile_session(
                    dataset_id, session_key, session_data, study_hash,
                    attempts=1,
                )

                actual = getattr(self, "_reconciled_name", None) or session_key
                results.append({
                    "session": session_key,
                    "kappa_name": actual,
                    "success": bool(reconciled_id),
                    "skipped_upload": True,
                    "reconciled": bool(reconciled_id),
                    "entity_id": reconciled_id,
                    "error": None if reconciled_id else "duplicate",
                    "message": (
                        (f"Эти данные уже в датасете под именем {actual}"
                         if actual != session_key
                         else f"Сессия {session_key} уже в датасете")
                        + f"; локальная запись "
                        + f"{'восстановлена' if reconciled_id else 'не найдена'}"
                    ),
                })
                continue

            result = await self._upload_session(
                dataset_id, session_key, session_data
            )
            results.append(result)
            await asyncio.sleep(2)  # Пауза между запросами

        uploaded = sum(1 for r in results if r.get("success"))
        logger.info(
            "Upload complete: %d/%d sessions, run=%s",
            uploaded, len(universe), self.run_id,
        )

        return {
            "dataset_id": dataset_id,
            "uploaded": uploaded,
            "total": len(universe),
            "sessions": results,
        }

    async def _resolve_dataset_id(self) -> Optional[int]:
        """Dataset this run uploads to.

        Prefers the one fixed at run start (backend/numbering.py) — resolving
        again here could pick a different dataset if `current` moved in
        configs/kappa_datasets.yaml while the run was in flight, splitting
        numbering and upload across two datasets. Falls back to resolving now
        for runs started before dataset_id existed, and for CLI/legacy paths.
        """
        if self.dataset_id is not None:
            return self.dataset_id

        # Warn if Kappa already has an empty dataset for this lesion_type —
        # that usually means an operator created one manually and forgot to
        # register it in kappa_datasets.yaml (KI-015 operational footgun).
        # Only worth checking when we are about to create one ourselves.
        if get_dataset_id(self.user_id, self.lesion_type, self.preprocessing_id) is None:
            await self._warn_if_empty_dataset_exists()

        from kappa_dataset_resolver import resolve_or_create
        return await resolve_or_create(
            token=self.token,
            user_id=self.user_id,
            user_type_id=self.user_type_id,
            lesion_type=self.lesion_type,
            preprocessing_id=self.preprocessing_id,
        )

    async def _warn_if_empty_dataset_exists(self) -> None:
        """
        Log a WARNING if Kappa already has an empty dataset whose name contains
        this lesion_type. That indicates a manually-created dataset was left
        unregistered in kappa_datasets.yaml — the auto-create below will produce
        a duplicate (KI-015).
        """
        try:
            all_datasets = await list_user_datasets(
                self.token, self.user_id, self.user_type_id
            )
        except Exception:
            return  # network error — don't block the upload

        for ds in all_datasets:
            name = ds.get("datasetName", "")
            ds_id = ds.get("datasetId")
            if not name or ds_id is None:
                continue
            if self.lesion_type.lower() not in name.lower():
                continue

            try:
                entities = await get_dataset_entities(
                    self.token, self.user_id, self.user_type_id, ds_id
                )
            except Exception:
                continue

            if entities is not None and len(entities) == 0:
                logger.warning(
                    "Empty dataset '%s' (id=%d) found in Kappa for lesion_type='%s'. "
                    "If it was created manually, either delete it or register its id "
                    "in configs/kappa_datasets.yaml to prevent creating a duplicate. "
                    "Proceeding with auto-create.",
                    name, ds_id, self.lesion_type,
                )

    def _discover_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Сканирует preprocessed/ и segmentation/ и группирует по сессиям.
        Возвращает: {session_key: {preprocessed: [...], masks: [...], quality: [...], ...}}
        """
        sessions: Dict[str, Dict[str, Any]] = {}

        preprocessed_dir = self.output_path / "preprocessed"
        segmentation_dir = self.output_path / "segmentation"
        quality_dir = self.output_path / "quality_reports"

        if not preprocessed_dir.exists() or not segmentation_dir.exists():
            logger.warning(
                "preprocessed or segmentation dir not found in %s",
                self.output_path,
            )
            return sessions

        # Находим preprocessed NIfTI, группируем по сессии
        for nifti in sorted(preprocessed_dir.rglob("*.nii.gz")):
            session_key = self._extract_session_key(nifti)
            if not session_key:
                continue
            sessions.setdefault(session_key, {
                "preprocessed": [],
                "masks": [],
                "quality_reports": [],
                "volume_report": None,
                "lobar_report": None,
                "lesion_stats_report": None,
                "mcdonald_report": None,
                "lesion_labels_mask": None,
            })
            sessions[session_key]["preprocessed"].append(nifti)

        # Находим маски сегментации
        for mask in sorted(segmentation_dir.rglob("*_segmask.nii.gz")):
            session_key = self._extract_session_key(mask)
            if session_key and session_key in sessions:
                sessions[session_key]["masks"].append(mask)

        # Находим native маски (из inverse transform)
        for mask in sorted(segmentation_dir.rglob("*_segmask_native_*.nii.gz")):
            session_key = self._extract_session_key(mask)
            if session_key and session_key in sessions:
                sessions[session_key]["masks"].append(mask)

        # Находим quality reports
        if quality_dir.exists():
            for qr in sorted(quality_dir.rglob("*_quality.json")):
                session_key = self._extract_session_key(qr)
                if session_key and session_key in sessions:
                    sessions[session_key]["quality_reports"].append(qr)

        # Находим volume reports (JSON, с fallback на txt)
        for vr in sorted(segmentation_dir.rglob("*_volume_report.json")):
            session_key = self._extract_session_key(vr)
            if session_key and session_key in sessions:
                sessions[session_key]["volume_report"] = vr

        # Fallback на txt если JSON не нашёлся
        for vr in sorted(segmentation_dir.rglob("*_volume_report.txt")):
            session_key = self._extract_session_key(vr)
            if session_key and session_key in sessions and sessions[session_key]["volume_report"] is None:
                sessions[session_key]["volume_report"] = vr

        # Находим lobar reports
        for lr in sorted(segmentation_dir.rglob("*_lobar_report.json")):
            session_key = self._extract_session_key(lr)
            if session_key and session_key in sessions:
                sessions[session_key]["lobar_report"] = lr

        # Находим lesion_stats reports (МС — количество и объёмы очагов)
        for ls in sorted(segmentation_dir.rglob("*_lesion_stats_report.json")):
            session_key = self._extract_session_key(ls)
            if session_key and session_key in sessions:
                sessions[session_key]["lesion_stats_report"] = ls

        # Находим mcdonald reports (МС — классификация очагов по McDonald-зонам)
        for mr in sorted(segmentation_dir.rglob("*_mcdonald_report.json")):
            session_key = self._extract_session_key(mr)
            if session_key and session_key in sessions:
                sessions[session_key]["mcdonald_report"] = mr

        # Find labeled lesion masks (МС — для hover объёма очага)
        for lbl in sorted(segmentation_dir.rglob("*_segmask_labels.nii.gz")):
            session_key = self._extract_session_key(lbl)
            if session_key and session_key in sessions:
                sessions[session_key]["lesion_labels_mask"] = lbl

        logger.info("Discovered %d sessions", len(sessions))
        for sk, sd in sessions.items():
            logger.info(
                "  %s: %d preprocessed, %d masks, %d quality, volume=%s, lobar=%s",
                sk,
                len(sd["preprocessed"]),
                len(sd["masks"]),
                len(sd["quality_reports"]),
                sd["volume_report"] is not None,
                sd["lobar_report"] is not None,
            )

        return sessions

    def _session_universe(self, discovered: Dict[str, Dict[str, Any]]) -> List[str]:
        """Every session this run was supposed to deliver, sorted.

        That is every complete session stage 01 produced
        (bids_organized/sub-*/ses-*), plus anything found downstream (layouts
        without bids_organized/, e.g. old or CLI runs). bids_organized/
        _incomplete/ is left out: those sessions wait in the doctor's
        incomplete-patients queue and never entered processing.
        """
        keys = set(discovered)
        bids_dir = self.output_path / "bids_organized"
        if bids_dir.is_dir():
            for sub in bids_dir.glob("sub-*"):
                if not sub.is_dir():
                    continue
                for ses in sub.glob("ses-*"):
                    if ses.is_dir():
                        keys.add(f"{sub.name}_{ses.name}")
        return sorted(keys)

    def _readiness(
        self, session_key: str, session_data: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """Is the session complete enough for Kappa, and if not, why not.

        Complete = every required modality of the lesion type is preprocessed
        and the main (atlas-space) segmentation mask exists. Checked by file
        name only, so it costs no disk access.
        """
        preprocessed = (session_data or {}).get("preprocessed") or []
        if not preprocessed:
            return False, "нет данных после предобработки"

        try:
            required = load_lesion_type_config(self.lesion_type)["required_modalities"]
        except KeyError:
            required = _DEFAULT_REQUIRED_MODALITIES
        names = {p.name for p in preprocessed}
        missing = [m for m in required if f"{session_key}_{m}.nii.gz" not in names]
        has_mask = any(
            "_native_" not in m.name for m in (session_data.get("masks") or [])
        )

        if not missing and has_mask:
            return True, None
        parts = []
        if missing:
            parts.append(", ".join(missing))
        if not has_mask:
            parts.append("маски сегментации")
        return False, "нет " + " и ".join(parts)

    def _extract_session_key(self, filepath: Path) -> Optional[str]:
        """Извлечь session_key (sub-XXX_ses-XXX) из пути или имени файла."""
        # Ищем в частях пути, ИСКЛЮЧАЯ имя файла (последний элемент)
        parts = filepath.parts[:-1]  # только директории
        sub = None
        ses = None
        for part in parts:
            if part.startswith("sub-"):
                sub = part
            elif part.startswith("ses-"):
                ses = part

        if sub and ses:
            return f"{sub}_{ses}"

        # Fallback: парсим из имени файла
        name = filepath.stem  # без расширения
        if name.endswith(".nii"):
            name = name[:-4]  # убираем .nii от .nii.gz
        name_parts = name.split("_")
        for i, p in enumerate(name_parts):
            if p.startswith("sub-") and i + 1 < len(name_parts):
                if name_parts[i + 1].startswith("ses-"):
                    return f"{p}_{name_parts[i + 1]}"

        return None

    def _compute_study_hash(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Стабильный хэш сессии: sha256 от PatientID:StudyInstanceUID.

        Делегирует в session_identity: этот же хэш считает локальный
        регистратор при завершении прогона, а register_patient апсертит по
        нему. Две реализации разошлись бы молча и раздвоили каждую сессию в
        реестре.
        """
        return session_identity.study_hash(
            self.output_path, session_data.get("preprocessed", [])
        )

    def _bind_pending_scope(self, dataset_id: int) -> None:
        """Move numbers issued under pending:<run_id> onto the real dataset.

        A pending scope exists only when Kappa was unreachable (or the
        mapping had nothing) at run start (backend/numbering.py); this binds
        those numbers to whichever dataset upload_results() resolves, which
        for a pending scope is always a dataset CREATED for this run — never
        merged into one that already holds numbers, which could collide with
        them. A no-op when nothing was pending (the common case: the scope
        was already a real dataset at run start).
        """
        from utils.bids_allocator import dataset_scope, pending_scope, rebind_scope
        moved = rebind_scope(pending_scope(self.run_id), dataset_scope(dataset_id),
                             getattr(self, "_allocation_db", None))
        if moved:
            logger.info("Bound %d pending allocations of run %s to dataset %d",
                        moved, self.run_id, dataset_id)

    async def _get_existing_entity_names(self, dataset_id: int) -> set:
        """Entity names already in the dataset (e.g. "sub-001_ses-001").

        Used for the name-clash check: two machines can each number into this
        dataset while both are offline, and would otherwise silently upload
        two different patients under the same sub-NNN.
        """
        entities = await self._fetch_dataset_entities(dataset_id)
        return {e.get("dsEntityName") for e in entities if e.get("dsEntityName")}

    def _entity_completeness(self, study_hash: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """What Kappa holds for this study (KI-035), from this attempt's listing.

        Returns (entity, state): "complete", "incomplete" (a required modality
        or the AI mask is missing, no expert mask), "incomplete_expert", or
        "unknown" when there is no listing or no file list — never rewrite an
        entity on missing information.
        """
        entities = [
            e for e in (getattr(self, "_attempt_entities", None) or [])
            if _entity_info(e).get("study_hash") == study_hash
        ]
        if not entities:
            return None, "unknown"
        try:
            required = load_lesion_type_config(self.lesion_type)["required_modalities"]
        except KeyError:
            required = _DEFAULT_REQUIRED_MODALITIES
        from kappa_entity_files import EXPERT_MASK_RE

        verdicts = []
        for entity in entities:
            files = entity.get("files")
            if not isinstance(files, list):
                verdicts.append((entity, "unknown"))
                continue
            name = entity.get("dsEntityName") or ""
            names = {f.get("fileName") or "" for f in files}
            expert = any(EXPERT_MASK_RE.search(n) for n in names)
            ai_mask = any("_segmask" in n and "_native_" not in n
                          and not EXPERT_MASK_RE.search(n) for n in names)
            complete = ai_mask and all(f"{name}_{m}.nii.gz" in names for m in required)
            if complete:
                verdicts.append((entity, "complete"))
            else:
                verdicts.append((entity, "incomplete_expert" if expert else "incomplete"))
        # Any complete copy settles it; otherwise the most cautious verdict.
        for wanted in ("complete", "unknown", "incomplete_expert", "incomplete"):
            for entity, state in verdicts:
                if state == wanted:
                    return entity, state
        return None, "unknown"

    async def _complete_entity(
        self, dataset_id: int, session_key: str,
        session_data: Dict[str, Any], entity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make an incomplete entity hold this run's whole session (KI-035).

        Every file is replaced, not only the missing ones: the entity's images
        come from an earlier run, and this run's mask belongs to this run's
        images. Expert masks are never touched (replace_entity_contents).
        No POST: the study is already in the dataset.
        """
        import kappa_entity_files

        entity_id = entity.get("dsEntityId")
        name = entity.get("dsEntityName") or session_key
        files = session_file_paths(session_data)
        logger.warning(
            "Completing incomplete entity %s (%s) in dataset %d with %d files of %s",
            entity_id, name, dataset_id, len(files), session_key,
        )
        try:
            outcome = await kappa_entity_files.replace_entity_contents(
                self.token, self.user_id, self.user_type_id,
                dataset_id, entity_id, files,
            )
        except Exception as exc:  # noqa: BLE001 — reported, retried next attempt
            logger.exception("Completing %s failed: %s", entity_id, exc)
            outcome = {"failed": [type(exc).__name__], "delete_job": None}

        if outcome.get("failed") or outcome.get("delete_job") == "failed":
            return {
                "session": session_key,
                "kappa_name": name,
                "success": False,
                "error": "upload failed",
                "entity_id": entity_id,
                "message": (
                    f"Не удалось дополнить {name}: "
                    f"{', '.join(outcome.get('failed') or ['удаление устаревших файлов'])}"
                ),
            }

        status = entity.get("dsEntityStatus")
        if isinstance(status, int) and status < 3:
            try:
                await update_entity_status(
                    token=self.token, user_id=self.user_id,
                    user_type_id=self.user_type_id, dataset_id=dataset_id,
                    entity_id=entity_id, status=3,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set status after completing %s: %s",
                               entity_id, exc)
        # The description still lists the first upload's files and
        # preprocessing; the files are the truth, but keep the two in step.
        try:
            info = {**_entity_info(entity),
                    **self._build_entity_info(session_key, session_data)}
            updated = await kappa_entity_files.update_entity_info(
                self.token, self.user_id, self.user_type_id, dataset_id, entity_id,
                info, remark=f"KI-035: completed by run {self.run_id[:8]}",
            )
        except Exception as exc:  # noqa: BLE001 — the files are already in place
            logger.warning("Building/sending dsEntityInfo for %s failed: %s",
                           entity_id, exc)
            updated = False
        if not updated:
            logger.warning("Entity %s completed, but its dsEntityInfo was not updated",
                           entity_id)
        self._do_local_registration(name, session_data, entity_id, dataset_id)
        return {
            "session": session_key,
            "kappa_name": name,
            "success": True,
            "completed": True,
            "entity_id": entity_id,
            "files": len(files),
            "message": f"Неполная сущность {name} дополнена данными этого прогона",
        }

    async def _fetch_dataset_entities(self, dataset_id: int) -> list:
        """The dataset listing, read once per upload attempt.

        Raises KappaUnavailable when Kappa did not answer: an unread listing
        is not an empty dataset (KI-069).
        """
        cached = getattr(self, "_attempt_entities", None)
        if cached is not None:
            return cached
        import kappa_client

        entities = await kappa_client.get_dataset_entities(
            token=self.token, user_id=self.user_id,
            user_type_id=self.user_type_id, dataset_id=dataset_id,
        )
        if entities is None:
            raise KappaUnavailable(f"dataset {dataset_id} listing unavailable")
        self._attempt_entities = entities
        return entities

    async def _get_existing_study_hashes(self, dataset_id: int) -> set:
        """Получить множество study_hash уже загруженных сущностей."""
        entities = await self._fetch_dataset_entities(dataset_id)

        if not entities:
            return set()

        # Отладка: смотрим формат ответа
        if entities:
            logger.debug(
                "First entity sample: %s",
                json.dumps(entities[0], ensure_ascii=False, default=str)[:500],
            )

        existing = set()
        for entity in entities:
            info = entity.get("dsEntityInfo") or {}
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except Exception:
                    info = {}

            study_hash = info.get("study_hash")
            if study_hash:
                existing.add(study_hash)

        logger.info("Found %d existing study_hashes in dataset %d", len(existing), dataset_id)
        return existing

    async def _upload_session(
        self,
        dataset_id: int,
        session_key: str,
        session_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Загрузить одну сессию как сущность."""

        # Собираем файлы: preprocessed + основная маска + labels.
        # Правило одно для выгрузки и для замены — см. session_file_paths.
        file_paths = session_file_paths(session_data)

        if not file_paths:
            return {"session": session_key, "success": False, "error": "no files"}

        # Формируем dsEntityInfo
        entity_info = self._build_entity_info(session_key, session_data)

        from kappa_entity_files import file_category

        result = await upload_entity(
            token=self.token,
            user_id=self.user_id,
            user_type_id=self.user_type_id,
            dataset_id=dataset_id,
            entity_name=session_key,
            file_paths=file_paths,
            entity_info=entity_info,
            files_category={p.name: file_category(p.name) for p in file_paths},
        )

        entity_id = self._extract_entity_id(result) if result is not None else None

        # Чистый успех: Каппа вернула entity_id в ответе.
        if entity_id:
            logger.info(
                "Session uploaded: %s (%d files)", session_key, len(file_paths)
            )
            await update_entity_status(
                token=self.token,
                user_id=self.user_id,
                user_type_id=self.user_type_id,
                dataset_id=dataset_id,
                entity_id=entity_id,
                status=3,  # Labeled
            )
            self._do_local_registration(
                session_key, session_data, entity_id, dataset_id
            )
            return {
                "session": session_key,
                "success": True,
                "files": len(file_paths),
                "entity_id": entity_id,
                "response": result,
            }

        # Ответа с entity_id нет (типичный случай — 504 Gateway Timeout от Каппы:
        # её nginx сдался по своему таймауту, но бэкенд сущность мог создать).
        # НЕ ретраим POST (не идемпотентен → плодит дубли). Вместо этого
        # идемпотентным GET проверяем, не появилась ли сущность в Каппе,
        # и регистрируем её локально по реальному entity_id.
        study_hash = self._compute_study_hash(session_data)
        recovered_id = await self._reconcile_session(
            dataset_id, session_key, session_data, study_hash
        )
        if recovered_id:
            return {
                "session": session_key,
                "success": True,
                "recovered": True,
                "files": len(file_paths),
                "entity_id": recovered_id,
            }

        logger.error("Failed to upload session: %s", session_key)
        return {"session": session_key, "success": False, "error": "upload failed"}

    def _do_local_registration(
        self,
        session_key: str,
        session_data: Dict[str, Any],
        entity_id: Optional[str],
        dataset_id: int,
    ) -> None:
        """
        Записать сущность в локальные таблицы (patient_registry + mask_versions).
        Идемпотентно: register_patient апсертит по study_hash, register_ai_mask
        пропускает, если версия 1 уже есть.
        """
        self._register_patient(session_key, session_data, entity_id, dataset_id)

        if entity_id:
            from mask_service import register_ai_mask
            main_masks = [
                m for m in session_data["masks"]
                if "_native_" not in m.name
            ]
            if main_masks:
                register_ai_mask(
                    entity_id=entity_id,
                    dataset_id=dataset_id,
                    file_path=str(main_masks[0]),
                )

    async def _find_kappa_entity_by_hash(
        self, dataset_id: int, study_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Найти в Каппе сущность с данным study_hash (самую свежую, если их несколько).
        Возвращает entity-dict или None.
        """
        entities = await get_dataset_entities(
            token=self.token,
            user_id=self.user_id,
            user_type_id=self.user_type_id,
            dataset_id=dataset_id,
        )
        if not entities:
            return None

        matches = []
        for entity in entities:
            info = entity.get("dsEntityInfo") or {}
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except Exception:
                    info = {}
            if info.get("study_hash") == study_hash:
                matches.append(entity)

        if not matches:
            return None

        matches.sort(key=lambda e: e.get("createdOn", ""), reverse=True)
        return matches[0]

    async def _reconcile_session(
        self,
        dataset_id: int,
        session_key: str,
        session_data: Dict[str, Any],
        study_hash: Optional[str],
        attempts: int = 3,
        delay: float = 6.0,
    ) -> Optional[str]:
        """
        Сверка с Каппой после неоднозначной загрузки (504): если сущность
        всё же создана — регистрируем её локально и возвращаем entity_id.
        Только GET-запросы (идемпотентно), повторный POST не делается.
        """
        if not study_hash:
            return None

        for attempt in range(attempts):
            entity = await self._find_kappa_entity_by_hash(dataset_id, study_hash)
            if entity:
                entity_id = entity.get("dsEntityId")
                if entity_id:
                    logger.warning(
                        "Reconciled entity after ambiguous upload: session=%s, "
                        "entity=%s (study_hash=%s)",
                        session_key, entity_id, study_hash,
                    )
                    # Поднимаем статус до Labeled только если он ниже —
                    # не откатываем уже верифицируемые/верифицированные сущности.
                    status = entity.get("dsEntityStatus")
                    if isinstance(status, int) and status < 3:
                        try:
                            await update_entity_status(
                                token=self.token,
                                user_id=self.user_id,
                                user_type_id=self.user_type_id,
                                dataset_id=dataset_id,
                                entity_id=entity_id,
                                status=3,
                            )
                        except Exception as e:
                            logger.warning("Failed to set status on reconcile: %s", e)

                    # Под каким именем сущность лежит в Kappa — то и есть
                    # правда. Регистрация под локальным session_key плодила
                    # в реестре второго «sub-001» для другого пациента:
                    # ровно та двусмысленность, ради устранения которой
                    # нумерация и делалась по датасетам.
                    actual_name = entity.get("dsEntityName") or session_key
                    if actual_name != session_key:
                        logger.warning(
                            "Session %s is in dataset %d under the name %s — "
                            "registering it under the dataset's name",
                            session_key, dataset_id, actual_name,
                        )
                    self._reconciled_name = actual_name
                    self._do_local_registration(
                        actual_name, session_data, entity_id, dataset_id
                    )
                    return entity_id

            if attempt < attempts - 1:
                await asyncio.sleep(delay)

        return None

    def _kappa_labels(self) -> List[str]:
        """dsEntityInfo.labels for this lesion type (configs/lesion_types.yaml
        kappa_labels, i.e. the segmentation service's mask classes)."""
        try:
            labels = load_lesion_type_config(self.lesion_type).get("kappa_labels")
        except KeyError:
            labels = None
        return list(labels) if labels else [self.lesion_type]

    def _build_entity_info(
        self,
        session_key: str,
        session_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Сформировать dsEntityInfo для сущности."""

        info: Dict[str, Any] = {
            "bids_id": session_key,
            # Mask classes; Kappa refuses statuses 3 and 5 without them (KI-075).
            "labels": self._kappa_labels(),
            "study_hash": self._compute_study_hash(session_data),
            "pipeline_run_id": self.run_id[:8],
            "lesion_type": self.lesion_type,
            "preprocessing_id": self.preprocessing_id,
            "modalities": [
                f.stem.replace(".nii", "").split("_")[-1]
                for f in session_data["preprocessed"]
            ],
            # Note: no file_count here on purpose — it would be a stale snapshot.
            # Expert mask versions accumulate as new entity files over time, so
            # the live entity.files list is the only correct source of the count.
            "data_files": [
                f.name for f in session_data["preprocessed"]
            ],
            "prediction_files": [
                m.name for m in session_data["masks"]
                if "_native_" not in m.name
            ],
        }

        # Quality reports
        if session_data["quality_reports"]:
            info["quality_reports"] = []
            for qr_path in session_data["quality_reports"]:
                try:
                    with open(qr_path, "r") as f:
                        qr = json.load(f)
                    info["quality_reports"].append({
                        "modality": qr.get("modality"),
                        "quality_score": qr.get("quality_score"),
                        "quality_category": qr.get("quality_category"),
                        "metrics": qr.get("metrics"),
                    })
                except Exception as e:
                    logger.warning("Failed to read quality report %s: %s", qr_path, e)

        # Volume report (GBM-specific NCR/ED/NET/ET tumor-subregion breakdown —
        # not meaningful for MS binary lesion masks, which carry no such
        # distinction; MS volumes are already reported correctly via
        # lesion_stats_report/mcdonald_report)
        if self.lesion_type != "multiple_sclerosis" and session_data["volume_report"]:
            vr_path = session_data["volume_report"]
            if vr_path.suffix == ".json":
                try:
                    with open(vr_path, "r") as f:
                        vr = json.load(f)
                    info["volume_report"] = {
                        "total_tumor_cm3": vr.get("total_tumor", {}).get("volume_cm3"),
                        "classes": {
                            cls_data["name"].split("(")[0].strip(): {
                                "voxels": cls_data.get("voxel_count"),
                                "cm3": cls_data.get("volume_cm3"),
                            }
                            for cls_data in vr.get("classes", {}).values()
                        },
                    }
                except Exception as e:
                    logger.warning("Failed to read volume report JSON: %s", e)
            else:
                info["volume_report"] = self._parse_volume_report(vr_path)

        # Lobar report
        if session_data["lobar_report"]:
            try:
                with open(session_data["lobar_report"], "r") as f:
                    lr = json.load(f)
                info["lobar_report"] = {
                    "total_lesion_cm3": lr.get("total_lesion_volume_cm3"),
                    "lobes": {
                        lobe_id: {
                            "name_ru": lobe_data.get("name_ru"),
                            "color": lobe_data.get("color"),
                            "cm3": lobe_data.get("total_volume_cm3"),
                            "percent": lobe_data.get("percent_of_lesion"),
                        }
                        for lobe_id, lobe_data in lr.get("lobes", {}).items()
                    },
                }
            except Exception as e:
                logger.warning("Failed to read lobar report: %s", e)

        # Lesion stats report (МС — количество и объёмы очагов)
        if session_data.get("lesion_stats_report"):
            try:
                with open(session_data["lesion_stats_report"], "r") as f:
                    ls = json.load(f)
                info["lesion_stats"] = {
                    "lesion_count": ls.get("lesion_count"),
                    "total_volume_cm3": ls.get("total_volume_cm3"),
                    "mean_lesion_volume_cm3": ls.get("mean_lesion_volume_cm3"),
                    "lesion_volumes_cm3": ls.get("lesion_volumes_cm3", []),
                    "lesion_volumes_by_label": ls.get("lesion_volumes_by_label", {}),
                }
                if session_data.get("lesion_labels_mask"):
                    info["lesion_labels_file"] = session_data["lesion_labels_mask"].name
            except Exception as e:
                logger.warning("Failed to read lesion stats report: %s", e)

        # McDonald zone report (МС — классификация очагов по зонам)
        if session_data.get("mcdonald_report"):
            try:
                with open(session_data["mcdonald_report"], "r") as f:
                    mr = json.load(f)
                info["mcdonald_report"] = {
                    "total_lesion_count": mr.get("total_lesion_count"),
                    "zones": mr.get("zones", {}),
                    "lesion_zones_by_label": mr.get("lesion_zones_by_label", {}),
                }
            except Exception as e:
                logger.warning("Failed to read mcdonald report: %s", e)

        return info

    def _parse_volume_report(self, report_path: Path) -> Optional[Dict[str, Any]]:
        """Парсит текстовый volume report в JSON."""
        try:
            text = report_path.read_text(encoding="utf-8")
            result = {"classes": {}}

            for line in text.splitlines():
                line = line.strip()

                if line.startswith("1. NCR"):
                    result["classes"]["NCR"] = self._parse_volume_line(line)
                elif line.startswith("2. ED"):
                    result["classes"]["ED"] = self._parse_volume_line(line)
                elif line.startswith("3. NET"):
                    result["classes"]["NET"] = self._parse_volume_line(line)
                elif line.startswith("4. ET"):
                    result["classes"]["ET"] = self._parse_volume_line(line)
                elif line.startswith("TUMOR CORE") or line.startswith("TOTAL TUMOR"):
                    # "TOTAL TUMOR" was renamed to "TUMOR CORE (NCR+NET+ET)" to
                    # match the clinical definition (TC, without edema). Keep the
                    # old prefix as a fallback for existing reports.
                    result["total_tumor_cm3"] = self._parse_volume_line(line).get("cm3")

            return result if result["classes"] else None

        except Exception as e:
            logger.warning("Failed to parse volume report %s: %s", report_path, e)
            return None

    @staticmethod
    def _parse_volume_line(line: str) -> Dict[str, float]:
        """Извлечь voxels и cm3 из строки volume report."""
        parts = line.split()
        numbers = []
        for p in reversed(parts):
            try:
                numbers.append(float(p))
            except ValueError:
                if numbers:
                    break
        numbers.reverse()

        if len(numbers) >= 3:
            return {"voxels": int(numbers[0]), "mm3": numbers[1], "cm3": numbers[2]}
        elif len(numbers) >= 1:
            return {"cm3": numbers[-1]}
        return {}

    @staticmethod
    def _extract_entity_id(response_text: str) -> Optional[str]:
        """Извлечь ds_entity_id из ответа Kappa."""
        try:
            data = json.loads(response_text)
            return data.get("ds_entity_id")
        except Exception:
            # Ответ может быть не JSON (soft error от 400)
            return None

    def _register_patient(
        self,
        session_key: str,
        session_data: Dict[str, Any],
        kappa_entity_id: Optional[str],
        kappa_dataset_id: int,
    ) -> None:
        """Зарегистрировать пациента в локальном реестре."""
        logger.info("Registering patient: session=%s, entity=%s", session_key, kappa_entity_id)
        try:
            from patient_registry import register_patient

            # Читаем метаданные для получения оригинальной информации
            meta = self._read_session_metadata(session_data)

            study_hash = self._compute_study_hash(session_data) or ""

            register_patient(
                bids_id=session_key,
                study_hash=study_hash,
                original_patient_id=meta.get("PatientID", ""),
                patient_name=meta.get("PatientName", ""),
                scan_date=meta.get("StudyDate", ""),
                study_instance_uid=meta.get("StudyInstanceUID", ""),
                kappa_entity_id=kappa_entity_id,
                kappa_dataset_id=kappa_dataset_id,
                pipeline_run_id=self.run_id,
                lesion_type=self.lesion_type,
                preprocessing_id=self.preprocessing_id,
            )
        except Exception as e:
            logger.error("Failed to register patient: %s", e, exc_info=True)

    def _read_session_metadata(self, session_data: Dict[str, Any]) -> Dict[str, str]:
        """Поля DICOM-метаданных сессии. См. session_identity."""
        return session_identity.registry_fields(
            self.output_path, session_data.get("preprocessed", [])
        )