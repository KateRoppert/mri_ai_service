"""
Сервис валидаций.
Хранит историю действий пользователей (confirm/reject/revoke) по сущностям Каппы.
Вычисляет текущее состояние сущности на основе последних решений каждого пользователя.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database import SessionLocal
from registry_models import Validation

logger = logging.getLogger(__name__)


def record_action(
    entity_id: str,
    dataset_id: int,
    user_id: int,
    action: str,
    user_name: Optional[str] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Записать действие пользователя (confirm/reject/revoke).
    Возвращает созданную запись.
    """
    if action not in ("confirm", "reject", "revoke"):
        raise ValueError(f"Invalid action: {action}")

    db = SessionLocal()
    try:
        record = Validation(
            entity_id=entity_id,
            dataset_id=dataset_id,
            user_id=user_id,
            user_name=user_name,
            action=action,
            comment=comment,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            "Validation recorded: entity=%s, user=%s, action=%s",
            entity_id, user_id, action,
        )
        return _to_dict(record)
    finally:
        db.close()


def get_entity_history(entity_id: str) -> List[Dict[str, Any]]:
    """Получить полную историю действий по сущности (в хронологическом порядке)."""
    db = SessionLocal()
    try:
        records = db.query(Validation).filter(
            Validation.entity_id == entity_id
        ).order_by(Validation.created_at.asc()).all()
        return [_to_dict(r) for r in records]
    finally:
        db.close()


def get_current_votes(entity_id: str) -> Dict[str, Any]:
    """
    Вычислить текущее состояние голосования по сущности.
    
    Логика:
    - Берём последнее действие каждого пользователя
    - Если последнее действие 'revoke' — голос не учитывается
    - Считаем количество подтверждений и отклонений
    
    Returns:
        {
            "confirms": [{"user_id": ..., "user_name": ..., "created_at": ...}],
            "rejects": [{"user_id": ..., "user_name": ..., "created_at": ...}],
            "confirms_count": int,
            "rejects_count": int,
            "total_votes": int,
        }
    """
    db = SessionLocal()
    try:
        # Получаем все действия по сущности, сортируя по времени
        records = db.query(Validation).filter(
            Validation.entity_id == entity_id
        ).order_by(Validation.created_at.asc()).all()

        # Группируем по user_id, оставляя последнее действие
        last_action_by_user: Dict[int, Validation] = {}
        for record in records:
            last_action_by_user[record.user_id] = record

        confirms = []
        rejects = []

        for user_id, record in last_action_by_user.items():
            if record.action == "confirm":
                confirms.append({
                    "user_id": user_id,
                    "user_name": record.user_name,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                })
            elif record.action == "reject":
                rejects.append({
                    "user_id": user_id,
                    "user_name": record.user_name,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                })
            # revoke — голос не учитывается

        return {
            "confirms": confirms,
            "rejects": rejects,
            "confirms_count": len(confirms),
            "rejects_count": len(rejects),
            "total_votes": len(confirms) + len(rejects),
        }
    finally:
        db.close()


def get_user_current_vote(entity_id: str, user_id: int) -> Optional[str]:
    """
    Получить текущий голос конкретного пользователя по сущности.
    Возвращает 'confirm', 'reject' или None (если не голосовал или отозвал).
    """
    db = SessionLocal()
    try:
        last_action = db.query(Validation).filter(
            Validation.entity_id == entity_id,
            Validation.user_id == user_id,
        ).order_by(Validation.created_at.desc()).first()

        if not last_action:
            return None
        if last_action.action == "revoke":
            return None
        return last_action.action
    finally:
        db.close()


def _to_dict(record: Validation) -> Dict[str, Any]:
    """Преобразовать SQLAlchemy объект в dict."""
    return {
        "id": record.id,
        "entity_id": record.entity_id,
        "dataset_id": record.dataset_id,
        "user_id": record.user_id,
        "user_name": record.user_name,
        "action": record.action,
        "comment": record.comment,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }