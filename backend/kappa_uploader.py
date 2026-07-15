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
    create_dataset,
    get_dataset_entities,
    list_user_datasets,
    update_entity_status,
    upload_entity,
)
from kappa_dataset_mapping import get_dataset_id, set_dataset_id
from preprocessing_version import compute_preprocessing_id

logger = logging.getLogger(__name__)


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
    ):
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.token = token
        self.user_id = user_id
        self.user_type_id = user_type_id
        self.lesion_type = lesion_type
        self.preprocessing_config_path = preprocessing_config_path

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

        # 2. Находим все сессии
        sessions = self._discover_sessions()
        if not sessions:
            logger.warning("No sessions found in %s", self.output_path)
            return {"uploaded": 0, "sessions": []}

        # 3. Проверяем дубликаты
        existing_hashes = await self._get_existing_study_hashes(dataset_id)

        # 4. Загружаем каждую сессию
        results = []
        for session_key, session_data in sessions.items():
            # Вычисляем study_hash для дедупликации
            study_hash = self._compute_study_hash(session_data)

            if study_hash and study_hash in existing_hashes:
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

                results.append({
                    "session": session_key,
                    "success": bool(reconciled_id),
                    "skipped_upload": True,
                    "reconciled": bool(reconciled_id),
                    "entity_id": reconciled_id,
                    "error": None if reconciled_id else "duplicate",
                    "message": (
                        f"Сессия {session_key} уже в датасете; локальная запись "
                        f"{'восстановлена' if reconciled_id else 'не найдена'}"
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
            uploaded, len(results), self.run_id,
        )

        return {
            "dataset_id": dataset_id,
            "uploaded": uploaded,
            "total": len(results),
            "sessions": results,
        }

    async def _resolve_dataset_id(self) -> Optional[int]:
        """Получить dataset_id из маппинга или создать новый."""
        dataset_id = get_dataset_id(self.lesion_type, self.preprocessing_id)

        if dataset_id is not None:
            logger.info(
                "Using existing dataset: lesion=%s, prep=%s, id=%d",
                self.lesion_type, self.preprocessing_id, dataset_id,
            )
            return dataset_id

        # Warn if Kappa already has an empty dataset for this lesion_type —
        # that usually means an operator created one manually and forgot to
        # register it in kappa_datasets.yaml (KI-015 operational footgun).
        await self._warn_if_empty_dataset_exists()

        # Создаём новый датасет в Каппе
        logger.info(
            "Creating new dataset: lesion=%s, prep=%s",
            self.lesion_type, self.preprocessing_id,
        )
        short_id = self.preprocessing_id[:8]
        dataset_name = f"{self.lesion_type}_{short_id}"

        new_id = await create_dataset(
            token=self.token,
            user_id=self.user_id,
            user_type_id=self.user_type_id,
            dataset_name=dataset_name,
            dataset_short_info=(
                f"Lesion: {self.lesion_type}, "
                f"Preprocessing: {self.preprocessing_id}"
            ),
            dataset_type=1,  # Image dataset
            dataset_tags=f"mri,{self.lesion_type},segmentation",
        )

        if new_id is not None:
            set_dataset_id(self.lesion_type, self.preprocessing_id, new_id)
            logger.info("New dataset created: id=%d", new_id)

        return new_id

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
        """
        Вычислить стабильный хэш сессии из DICOM-метаданных.
        Используем PatientID + StudyInstanceUID.
        """
        metadata_dir = self.output_path / "metadata"
        if not metadata_dir.exists():
            return None

        # Ищем JSON в metadata по той же структуре sub/ses
        for prep_file in session_data.get("preprocessed", []):
            session_key = self._extract_session_key(prep_file)
            if not session_key:
                continue
            sub, ses = session_key.split("_", 1)
            modality = prep_file.stem.replace(".nii", "").split("_")[-1]

            meta_pattern = metadata_dir / sub / ses / "anat" / modality
            if meta_pattern.exists():
                for meta_file in meta_pattern.glob("*.json"):
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                        ident = meta.get("identification", {})
                        patient_id = ident.get("PatientID", "")
                        study_uid = ident.get("StudyInstanceUID", "")
                        if patient_id and study_uid:
                            raw = f"{patient_id}:{study_uid}"
                            return hashlib.sha256(raw.encode()).hexdigest()[:16]
                    except Exception as e:
                        logger.warning("Failed to read metadata %s: %s", meta_file, e)
                        continue

        return None

    async def _get_existing_study_hashes(self, dataset_id: int) -> set:
        """Получить множество study_hash уже загруженных сущностей."""
        from kappa_client import get_dataset_entities

        entities = await get_dataset_entities(
            token=self.token,
            user_id=self.user_id,
            user_type_id=self.user_type_id,
            dataset_id=dataset_id,
        )

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

        # Собираем файлы: preprocessed + основная маска
        file_paths = list(session_data["preprocessed"])

        # Добавляем только основную маску (без native)
        main_masks = [
            m for m in session_data["masks"]
            if "_native_" not in m.name
        ]
        file_paths.extend(main_masks)

        # Include the labeled lesion mask so validation hover works Kappa-only
        if session_data.get("lesion_labels_mask"):
            file_paths.append(session_data["lesion_labels_mask"])

        if not file_paths:
            return {"session": session_key, "success": False, "error": "no files"}

        # Формируем dsEntityInfo
        entity_info = self._build_entity_info(session_key, session_data)

        result = await upload_entity(
            token=self.token,
            user_id=self.user_id,
            user_type_id=self.user_type_id,
            dataset_id=dataset_id,
            entity_name=session_key,
            file_paths=file_paths,
            entity_info=entity_info,
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

                    self._do_local_registration(
                        session_key, session_data, entity_id, dataset_id
                    )
                    return entity_id

            if attempt < attempts - 1:
                await asyncio.sleep(delay)

        return None

    def _build_entity_info(
        self,
        session_key: str,
        session_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Сформировать dsEntityInfo для сущности."""

        info: Dict[str, Any] = {
            "bids_id": session_key,
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
        """Прочитать DICOM-метаданные сессии из папки metadata/."""
        metadata_dir = self.output_path / "metadata"
        if not metadata_dir.exists():
            return {}

        for prep_file in session_data.get("preprocessed", []):
            session_key = self._extract_session_key(prep_file)
            if not session_key:
                continue
            sub, ses = session_key.split("_", 1)
            modality = prep_file.stem.replace(".nii", "").split("_")[-1]

            meta_dir = metadata_dir / sub / ses / "anat" / modality
            if meta_dir.exists():
                for meta_file in meta_dir.glob("*.json"):
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                        ident = meta.get("identification", {})
                        # PatientName может быть списком символов
                        patient_name = ident.get("PatientName", "")
                        if isinstance(patient_name, list):
                            patient_name = "".join(str(c) for c in patient_name)
                        return {
                            "PatientID": ident.get("PatientID", ""),
                            "PatientName": patient_name,
                            "StudyDate": ident.get("StudyDate", ""),
                            "StudyInstanceUID": ident.get("StudyInstanceUID", ""),
                        }
                    except Exception:
                        continue
        return {}