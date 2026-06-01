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


async def list_user_datasets(
    token: str,
    user_id: int,
    user_type_id: int,
) -> list:
    """
    Return all datasets for the given user.
    Each element is a dict with at least 'datasetId' and 'datasetName'.
    Returns an empty list on error.
    """
    url = f"{KAPPA_DATA_URL}/datasets/{user_id}/{user_type_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            response = await client.get(url, headers=headers)
        if response.is_success:
            data = response.json()
            return data if isinstance(data, list) else []
        logger.warning("list_user_datasets: status=%s", response.status_code)
        return []
    except Exception as exc:
        logger.exception("Error listing datasets: %s", exc)
        return []


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
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
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
    dataset_type: int = 1,
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
            elif suffix == ".txt":
                ct = "text/plain"
            elif suffix == ".mat":
                ct = "application/octet-stream"
            else:
                ct = "application/octet-stream"
            files.append(("files", (fpath.name, f, ct)))

        if not files:
            logger.warning("No valid files to upload for entity: %s", entity_name)
            return None

        headers = {"Authorization": f"Bearer {token}"}

        # Retry с увеличивающимся таймаутом для больших файлов
        response = None
        last_exc = None
        for attempt in range(3):
            write_timeout = 300.0 * (attempt + 1)
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, read=300.0, write=write_timeout),
                    verify=False,
                ) as client:
                    response = await client.post(
                        url, data=data, files=files, headers=headers
                    )
                last_exc = None
                break
            except (httpx.WriteTimeout, httpx.ConnectTimeout) as exc:
                last_exc = exc
                logger.warning(
                    "Timeout on attempt %d/%d (write_timeout=%.0fs), entity=%s: %s",
                    attempt + 1, 3, write_timeout, entity_name, type(exc).__name__,
                )
                # Перематываем file handles для повторной попытки
                for f in open_handles:
                    f.seek(0)

        if last_exc is not None:
            logger.error("All upload attempts failed for entity: %s", entity_name)
            return None

        # Обработка ответа
        if response.is_success:
            logger.info(
                "Entity uploaded: name=%s, dataset_id=%s, files=%d",
                entity_name, dataset_id, len(files),
            )
            return response.text
        elif response.status_code == 400:
            # Каппа иногда возвращает 400, но файл при этом сохраняется
            logger.warning(
                "Entity likely uploaded with post-processing error: "
                "name=%s, dataset_id=%s, status=%s, body=%s",
                entity_name, dataset_id, response.status_code,
                response.text[:300],
            )
            return f'{{"warning": "400 but likely saved", "entity_name": "{entity_name}"}}'
        else:
            logger.warning(
                "Failed to upload entity: status=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
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

async def update_entity_status(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_id: int,
    entity_id: str,
    status: int = 3,  # 3 = Labeled
) -> bool:
    """
    Обновить статус сущности датасета.
    Статусы: 0=Deactive, 1=Active, 2=New, 3=Labeled,
             4=Under Verification, 5=Verified.
    """
    url = (
        f"{KAPPA_DATA_URL}/datasets/datasetEntities"
        f"/{user_id}/{user_type_id}/{dataset_id}/{entity_id}"
    )

    update_payload = json.dumps({"dsEntityStatus": status, "remark": ""})

    # API требует multipart/form-data с update_dataset_entity + files
    data = {"update_dataset_entity": update_payload}

    # Пустой файл-заглушка (files обязателен в API)
    files = [("files", ("empty.json", b"{}", "application/json"))]

    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            response = await client.put(
                url, data=data, files=files, headers=headers
            )

        if response.is_success:
            logger.info(
                "Entity status updated: entity=%s, status=%d", entity_id, status
            )
            return True
        else:
            logger.warning(
                "Failed to update entity status: status_code=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
            return False
    except Exception as exc:
        logger.exception("Error updating entity status: %s", exc)
        return False

async def replace_entity_file(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_id: int,
    entity_id: str,
    file_path: Path,
    content_type: str = "application/gzip",
) -> Optional[str]:
    """
    Заменить файл(ы) сущности в Каппе (PUT с новым файлом).
    Используется для загрузки отредактированной маски экспертом.
    Возвращает kappa_file_id загруженного файла или None при ошибке.
    """
    url = (
        f"{KAPPA_DATA_URL}/datasets/datasetEntities"
        f"/{user_id}/{user_type_id}/{dataset_id}/{entity_id}"
    )

    update_payload = json.dumps({"remark": "mask updated by expert"})
    data = {"update_dataset_entity": update_payload}

    headers = {"Authorization": f"Bearer {token}"}
    f = None
    target_filename = file_path.name

    try:
        if not file_path.exists():
            logger.error("File not found for replacement: %s", file_path)
            return None

        f = open(file_path, "rb")
        files = [("files", (target_filename, f, content_type))]

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=300.0, write=300.0),
            verify=False,
        ) as client:
            response = await client.put(
                url, data=data, files=files, headers=headers
            )

        if response.is_success:
            logger.info(
                "Entity file replaced: entity=%s, file=%s",
                entity_id, target_filename,
            )
        else:
            logger.warning(
                "Failed to replace entity file: status=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
            return None
    except Exception as exc:
        logger.exception("Error replacing entity file: %s", exc)
        return None
    finally:
        if f:
            f.close()

    # Получаем file_id загруженного файла из деталей сущности
    try:
        details = await get_entity_details(
            token=token,
            user_id=user_id,
            user_type_id=user_type_id,
            dataset_id=dataset_id,
            entity_id=entity_id,
        )
        if details and "files" in details:
            for file_info in details["files"]:
                if file_info.get("fileName") == target_filename:
                    file_id = file_info.get("fileId")
                    logger.info(
                        "Resolved kappa_file_id: %s for %s",
                        file_id, target_filename,
                    )
                    return file_id
            logger.warning(
                "File %s not found in entity details after upload",
                target_filename,
            )
    except Exception as exc:
        logger.warning("Failed to resolve kappa_file_id: %s", exc)

    # Файл загружен, но file_id не удалось получить
    return "uploaded_no_file_id"


async def get_dataset_entities(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_id: int,
) -> Optional[list]:
    """
    Получить список сущностей датасета.
    """
    url = (
        f"{KAPPA_DATA_URL}/datasets/datasetEntities"
        f"/{user_id}/{user_type_id}/{dataset_id}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            response = await client.get(url, headers=headers)

        if response.is_success:
            data = response.json()
            logger.debug("Got %d entities for dataset %d", len(data) if isinstance(data, list) else 0, dataset_id)
            return data if isinstance(data, list) else []
        else:
            logger.warning(
                "Failed to get entities: status=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
            return None
    except Exception as exc:
        logger.exception("Error getting dataset entities: %s", exc)
        return None

async def get_entity_details(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_id: int,
    entity_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Получить детали сущности датасета (включая список файлов).
    """
    url = (
        f"{KAPPA_DATA_URL}/datasets/datasetEntities"
        f"/{user_id}/{user_type_id}/{dataset_id}/{entity_id}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            response = await client.get(url, headers=headers)

        if response.is_success:
            return response.json()
        else:
            logger.warning(
                "Failed to get entity details: status=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
            return None
    except Exception as exc:
        logger.exception("Error getting entity details: %s", exc)
        return None

async def download_entity_file(
    token: str,
    user_id: int,
    user_type_id: int,
    dataset_id: int,
    file_id: str,
) -> Optional[bytes]:
    """
    Скачать файл сущности по file_id.
    Возвращает bytes или None при ошибке.
    """
    url = (
        f"{KAPPA_DATA_URL}/datasets/datasetEntities/files"
        f"/{user_id}/{user_type_id}/{dataset_id}/{file_id}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=300.0),
            verify=False,
        ) as client:
            response = await client.get(url, headers=headers)

        if response.is_success:
            logger.info("File downloaded: %s (%d bytes)", file_id, len(response.content))
            return response.content
        else:
            logger.warning(
                "Failed to download file: status=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
            return None
    except Exception as exc:
        logger.exception("Error downloading file: %s", exc)
        return None