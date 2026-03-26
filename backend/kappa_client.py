"""
Клиент для работы с Kappa Data API.
Создание датасетов и загрузка сущностей (файлов).
"""
import json
import logging
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

KAPPA_DATA_URL = "https://kappa.nsu.ru:8061/data-micro-services/v1"


async def _find_dataset_id(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_name: str,
) -> Optional[int]:
    """
    Найти dataset_id по имени через GET /datasets/{user_id}/{user_type_id}.
    """
    url = f"{KAPPA_DATA_URL}/datasets/{user_id}/{user_type_id}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(url, headers=headers)

        if not response.is_success:
            logger.warning("Failed to get datasets list: status=%s", response.status_code)
            return None

        datasets = response.json()
        if not isinstance(datasets, list):
            logger.warning("Unexpected datasets response format: %s", type(datasets))
            return None

        for ds in datasets:
            if ds.get("datasetName") == dataset_name:
                return ds.get("datasetId")

        logger.warning("Dataset not found by name: %s", dataset_name)
        return None

    except Exception as exc:
        logger.exception("Error finding dataset ID: %s", exc)
        return None


async def create_dataset(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_name: str,
    dataset_short_info: str = "",
    dataset_type: int = 0,
    dataset_tags: str = "",
) -> Optional[int]:
    """
    Создать новый датасет в Kappa.
    Возвращает dataset_id или None при ошибке.
    """
    url = f"{KAPPA_DATA_URL}/datasets/new/{user_id}/{user_type_id}"
    payload = {
        "userId": user_id,
        "datasetName": dataset_name,
        "datasetType": dataset_type,
        "datasetShortInfo": dataset_short_info,
        "datasetTags": dataset_tags,
    }
    headers = {"Authorization": f"Bearer {token}"}

    try:
        logger.debug("Creating dataset: url=%s, payload=%s", url, payload)

        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.is_success:
            result = response.json()
            logger.info("Dataset created: name=%s, response=%s", dataset_name, result)
            # API возвращает строку-подтверждение, не ID.
            # Получаем ID через список датасетов пользователя.
            actual_id = await _find_dataset_id(token, user_id, user_type_id, dataset_name)
            return actual_id
        else:
            logger.warning(
                "Failed to create dataset: status=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
            # Возможно, датасет с таким именем уже существует
            if response.status_code == 400:
                logger.info("Trying to find existing dataset: %s", dataset_name)
                existing_id = await _find_dataset_id(
                    token, user_id, user_type_id, dataset_name
                )
                if existing_id is not None:
                    logger.info(
                        "Found existing dataset: name=%s, id=%s",
                        dataset_name, existing_id,
                    )
                return existing_id
            return None
    except Exception as exc:
        logger.exception("Error creating dataset: %s", exc)
        return None


def _zip_files(file_paths: List[Path], archive_name: str) -> Path:
    """
    Упаковать список файлов в zip-архив во временную директорию.
    Сохраняет структуру: parent_dir/filename (чтобы модальности различались).
    """
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / f"{archive_name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in file_paths:
            # Сохраняем parent/filename (например t1/sub-001_ses-001_T1w_0001.dcm)
            arcname = f"{fpath.parent.name}/{fpath.name}"
            zf.write(fpath, arcname)

    logger.info(
        "Created zip: %s (%d files, %.1f MB)",
        zip_path, len(file_paths), zip_path.stat().st_size / 1024 / 1024,
    )
    return zip_path


async def upload_entity(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_id: int,
    entity_name: str,
    file_paths: List[Path],
    entity_info: Optional[Dict[str, Any]] = None,
    entity_source: str = "mri-ai-service",
    labeling_algo: str = "default",
    zip_as_archive: bool = False,
) -> Optional[str]:
    """
    Загрузить сущность (один или несколько файлов) в датасет Kappa.
    Если zip_as_archive=True, файлы упаковываются в zip перед загрузкой.
    Возвращает ответ сервера или None при ошибке.
    """
    url = (
        f"{KAPPA_DATA_URL}/datasets/datasetEntities/new"
        f"/{user_id}/{user_type_id}/{dataset_id}"
    )

    collected_on = datetime.now(timezone.utc).isoformat()

    new_dataset_entity = {
        "datasetId": dataset_id,
        "userId": user_id,
        "dsEntityName": entity_name,
        "entitySource": entity_source,
        "collectedOn": collected_on,
        "labelingAlgo": labeling_algo,
        "dsEntityInfo": entity_info or {},
    }

    entity_json = json.dumps(new_dataset_entity, ensure_ascii=False)

    # multipart: form field + файлы
    data = {"new_dataset_entity": entity_json}

    zip_path = None
    files = []
    open_handles = []
    try:
        # Если нужен архив — упаковываем
        if zip_as_archive and len(file_paths) > 0:
            zip_path = _zip_files(file_paths, entity_name)
            actual_files = [zip_path]
        else:
            actual_files = file_paths

        for fpath in actual_files:
            if not fpath.exists():
                logger.warning("File not found, skipping: %s", fpath)
                continue
            f = open(fpath, "rb")
            open_handles.append(f)
            # Определяем content-type
            suffix = fpath.suffix.lower()
            if suffix == ".json":
                ct = "application/json"
            elif suffix == ".gz":
                ct = "application/gzip"
            elif suffix == ".zip":
                ct = "application/zip"
            elif suffix in (".png", ".jpg", ".jpeg"):
                ct = f"image/{suffix.lstrip('.')}"
            elif suffix == ".dcm":
                ct = "application/dicom"
            else:
                ct = "application/octet-stream"
            files.append(("files", (fpath.name, f, ct)))

        if not files:
            logger.warning("No valid files to upload for entity: %s", entity_name)
            return None

        headers = {"Authorization": f"Bearer {token}"}

        # Retry с увеличивающимся таймаутом для больших файлов
        last_exc = None
        for attempt in range(3):
            write_timeout = 300.0 * (attempt + 1)  # 300, 600, 900 сек
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, read=300.0, write=write_timeout),
                    verify=False,
                ) as client:
                    response = await client.post(
                        url, data=data, files=files, headers=headers
                    )
                last_exc = None
                break  # Успешно отправлено
            except httpx.WriteTimeout as exc:
                last_exc = exc
                logger.warning(
                    "WriteTimeout on attempt %d/%d (write_timeout=%.0fs), entity=%s",
                    attempt + 1, 3, write_timeout, entity_name,
                )
                # Пересоздаём file handles для повторной попытки
                for f in open_handles:
                    f.seek(0)

        if last_exc is not None:
            logger.exception("All upload attempts failed for entity: %s", entity_name)
            return None

    except Exception as exc:
        logger.exception("Error uploading entity: %s", exc)
        return None
    finally:
        for f in open_handles:
            f.close()
        if zip_path and zip_path.exists():
            zip_path.unlink()
            zip_path.parent.rmdir()