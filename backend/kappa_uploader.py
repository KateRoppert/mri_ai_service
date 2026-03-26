"""
Модуль загрузки результатов пайплайна в Kappa.
Отслеживает завершение этапов, создаёт датасеты и загружает сущности.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

from kappa_client import create_dataset, upload_entity

logger = logging.getLogger(__name__)

# Папки, которые мониторим. Ключ — имя папки, значение — описание для датасета.
STAGE_FOLDERS = {
    "bids_organized": {
        "description": "DICOM файлы после BIDS-реорганизации и деперсонализации",
        "tags": "mri,dicom,bids",
        "zip": True,
    },
    "metadata": {
        "description": "Метаданные сессий (извлечённые из DICOM)",
        "tags": "mri,metadata,json",
        "zip": False,
    },
    "nifti": {
        "description": "NIfTI файлы после конвертации из DICOM",
        "tags": "mri,nifti,conversion",
        "zip": False,
    },
    "quality_reports": {
        "description": "Отчёты о качестве изображений",
        "tags": "mri,quality,assessment",
        "zip": False,
    },
    "preprocessed": {
        "description": "Предобработанные NIfTI (коррекция, регистрация, нормализация)",
        "tags": "mri,nifti,preprocessed",
        "zip": False,
    },
    "transformations": {
        "description": "Матрицы трансформаций и маски мозга",
        "tags": "mri,transformations,registration",
        "zip": False,
    },
    "segmentation": {
        "description": "Маски сегментации, отчёты об объёмах и лобарной локализации",
        "tags": "mri,segmentation,lesion",
        "zip": False,
    },
}


class KappaUploader:
    """
    Управляет загрузкой результатов пайплайна в Kappa.
    Один экземпляр на один запуск пайплайна (run_id).
    """

    def __init__(
        self,
        run_id: str,
        output_path: str,
        token: str,
        user_id: int,
        user_type_id: int,
    ):
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.token = token
        self.user_id = user_id
        self.user_type_id = user_type_id

        self._dataset_ids: Dict[str, int] = {}
        self._uploaded_files: Set[str] = set()
        self._uploaded_entities: Set[str] = set()
        self._lock = asyncio.Lock()

    async def _ensure_dataset(self, folder_name: str) -> Optional[int]:
        """Создать датасет для папки, если ещё не создан."""
        if folder_name in self._dataset_ids:
            return self._dataset_ids[folder_name]

        config = STAGE_FOLDERS[folder_name]
        # Используем первые 8 символов run_id чтобы уложиться в лимит имени Kappa
        short_id = self.run_id[:8]
        dataset_name = f"{short_id}_{folder_name}"

        dataset_id = await create_dataset(
            token=self.token,
            user_id=self.user_id,
            user_type_id=self.user_type_id,
            dataset_name=dataset_name,
            dataset_short_info=config["description"],
            dataset_type=1,
            dataset_tags=config["tags"],
        )

        if dataset_id is not None:
            self._dataset_ids[folder_name] = dataset_id
            logger.info(
                "Kappa dataset ready: folder=%s, dataset_id=%s, run=%s",
                folder_name, dataset_id, self.run_id,
            )
        else:
            logger.error(
                "Failed to create/find Kappa dataset: folder=%s, run=%s",
                folder_name, self.run_id,
            )

        await asyncio.sleep(1)
        return dataset_id

    def _discover_sessions(self, folder_path: Path) -> Dict[str, list]:
        """
        Сканирует папку и группирует файлы по сессиям.
        Возвращает {session_key: [file_paths]}, где session_key = "sub-XXX_ses-XXX".
        Файлы в incomplete_data и корне группируются в сессию "_meta".
        """
        sessions: Dict[str, list] = {}

        if not folder_path.exists():
            return sessions

        for fpath in sorted(folder_path.rglob("*")):
            if not fpath.is_file():
                continue

            if str(fpath) in self._uploaded_files:
                continue

            rel = fpath.relative_to(folder_path)
            parts = rel.parts

            session_key = "_meta"
            for i, part in enumerate(parts):
                if part.startswith("sub-") and i + 1 < len(parts) and parts[i + 1].startswith("ses-"):
                    session_key = f"{part}_{parts[i + 1]}"
                    break

            sessions.setdefault(session_key, []).append(fpath)

        return sessions

    async def upload_folder(self, folder_name: str) -> int:
        """
        Загрузить новые файлы из указанной папки.
        Возвращает количество загруженных сущностей.
        """
        if folder_name not in STAGE_FOLDERS:
            logger.warning("Unknown folder: %s", folder_name)
            return 0

        folder_path = self.output_path / folder_name
        if not folder_path.exists():
            return 0

        async with self._lock:
            sessions = self._discover_sessions(folder_path)
            if not sessions:
                return 0

            dataset_id = await self._ensure_dataset(folder_name)
            if dataset_id is None:
                return 0

            config = STAGE_FOLDERS[folder_name]
            use_zip = config["zip"]
            uploaded_count = 0

            for session_key, file_paths in sessions.items():
                entity_key = f"{folder_name}:{session_key}"
                if entity_key in self._uploaded_entities:
                    continue

                entity_info = {
                    "run_id": self.run_id,
                    "pipeline_stage": folder_name,
                    "session": session_key,
                    "file_count": len(file_paths),
                }

                should_zip = use_zip and session_key != "_meta"
                if should_zip:
                    modalities = sorted(set(
                        f.parent.name for f in file_paths
                        if f.parent.name not in ("anat", folder_name)
                        and not f.parent.name.startswith("sub-")
                        and not f.parent.name.startswith("ses-")
                    ))
                    if modalities:
                        entity_info["modalities"] = modalities
                    entity_info["archive_format"] = "zip"

                result = await upload_entity(
                    token=self.token,
                    user_id=self.user_id,
                    user_type_id=self.user_type_id,
                    dataset_id=dataset_id,
                    entity_name=session_key,
                    file_paths=file_paths,
                    entity_info=entity_info,
                    zip_as_archive=should_zip,
                )

                if result is not None:
                    for fpath in file_paths:
                        self._uploaded_files.add(str(fpath))
                    self._uploaded_entities.add(entity_key)
                    uploaded_count += 1
                    logger.info(
                        "Uploaded entity: %s/%s (%d files), run=%s",
                        folder_name, session_key, len(file_paths), self.run_id,
                    )
                else:
                    logger.error(
                        "Failed to upload entity: %s/%s, run=%s",
                        folder_name, session_key, self.run_id,
                    )

                await asyncio.sleep(2)

            return uploaded_count

    async def upload_all_new(self) -> Dict[str, int]:
        """
        Проверить все папки и загрузить новые файлы.
        Возвращает {folder_name: uploaded_count}.
        """
        results = {}
        for folder_name in STAGE_FOLDERS:
            count = await self.upload_folder(folder_name)
            if count > 0:
                results[folder_name] = count
        return results