"""
Тесты для сервиса версионирования масок.
Запуск: python test_mask_service.py
"""
from mask_service import (
    register_ai_mask,
    register_expert_mask,
    get_current_mask,
    get_mask_history,
    get_mask_by_version,
)
from database import SessionLocal
from registry_models import MaskVersion, init_registry_tables


def cleanup():
    """Удалить тестовые записи."""
    db = SessionLocal()
    try:
        db.query(MaskVersion).filter(
            MaskVersion.entity_id.like("test-%")
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def test_ai_mask():
    """Регистрация маски от ИИ."""
    cleanup()

    result = register_ai_mask(
        entity_id="test-entity-mask-001",
        dataset_id=133,
        file_path="/app/output/segmentation/sub-001_ses-001_t1_segmask.nii.gz",
    )

    assert result["version"] == 1
    assert result["source"] == "ai"
    assert result["uploaded_by_name"] == "AI Pipeline"
    print(f"OK: AI mask registered, version={result['version']}")

    # Повторная регистрация не создаёт дубликат
    result2 = register_ai_mask(
        entity_id="test-entity-mask-001",
        dataset_id=133,
        file_path="/app/output/segmentation/sub-001_ses-001_t1_segmask.nii.gz",
    )
    assert result2["version"] == 1
    print(f"OK: duplicate AI mask ignored")

    cleanup()


def test_expert_masks():
    """Загрузка масок экспертами."""
    cleanup()

    # ИИ маска
    register_ai_mask(
        entity_id="test-entity-mask-002",
        dataset_id=133,
        file_path="/app/output/seg/mask_v1.nii.gz",
    )

    # Эксперт 1
    v2 = register_expert_mask(
        entity_id="test-entity-mask-002",
        dataset_id=133,
        file_path="/app/output/seg/mask_v2.nii.gz",
        user_id=26,
        user_name="Kate",
    )
    assert v2["version"] == 2
    assert v2["source"] == "expert"
    print(f"OK: expert mask v2")

    # Эксперт 2
    v3 = register_expert_mask(
        entity_id="test-entity-mask-002",
        dataset_id=133,
        file_path="/app/output/seg/mask_v3.nii.gz",
        user_id=27,
        user_name="Ivan",
    )
    assert v3["version"] == 3
    print(f"OK: expert mask v3")

    cleanup()


def test_current_and_history():
    """Получение актуальной маски и истории."""
    cleanup()

    entity_id = "test-entity-mask-003"

    register_ai_mask(entity_id=entity_id, dataset_id=133, file_path="/path/v1.nii.gz")
    register_expert_mask(entity_id=entity_id, dataset_id=133, file_path="/path/v2.nii.gz", user_id=26, user_name="Kate")
    register_expert_mask(entity_id=entity_id, dataset_id=133, file_path="/path/v3.nii.gz", user_id=27, user_name="Ivan")

    # Актуальная = последняя
    current = get_current_mask(entity_id)
    assert current["version"] == 3
    assert current["uploaded_by_name"] == "Ivan"
    print(f"OK: current mask is v3 by Ivan")

    # История
    history = get_mask_history(entity_id)
    assert len(history) == 3
    assert history[0]["version"] == 1
    assert history[2]["version"] == 3
    print(f"OK: history has {len(history)} versions")

    # По версии
    v2 = get_mask_by_version(entity_id, 2)
    assert v2["uploaded_by_name"] == "Kate"
    print(f"OK: v2 is by Kate")

    cleanup()


if __name__ == "__main__":
    init_registry_tables()

    print("=== test_ai_mask ===")
    test_ai_mask()

    print("\n=== test_expert_masks ===")
    test_expert_masks()

    print("\n=== test_current_and_history ===")
    test_current_and_history()

    print("\n=== Все тесты пройдены ===")