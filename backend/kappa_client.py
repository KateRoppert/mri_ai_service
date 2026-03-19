"""
Клиент для работы с Kappa Data API.
Создание датасетов и загрузка сущностей (файлов).
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

KAPPA_DATA_URL = "https://kappa.nsu.ru:8061/data-micro-services/v1"


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
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.is_success:
            result = response.json()
            logger.info(
                "Dataset created: name=%s, response=%s", dataset_name, result
            )
            # Ответ — строка или число (dataset_id)
            return int(result) if isinstance(result, (str, int)) else result
        else:
            logger.warning(
                "Failed to create dataset: status=%s, body=%s",
                response.status_code,
                response.text[:300],
            )
            return None
    except Exception as exc:
        logger.exception("Error creating dataset: %s", exc)
        return None


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
) -> Optional[str]:
    """
    Загрузить сущность (один или несколько файлов) в датасет Kappa.
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

    files = []
    open_handles = []
    try:
        for fpath in file_paths:
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

        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            response = await client.post(
                url, data=data, files=files, headers=headers
            )

        if response.is_success:
            logger.info(
                "Entity uploaded: name=%s, dataset_id=%s, files=%d",
                entity_name,
                dataset_id,
                len(files),
            )
            return response.text
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