"""
Тесты для локального реестра пациентов.
Запуск: python test_patient_registry.py
"""
from patient_registry import (
    register_patient,
    find_by_study_hash,
    find_by_patient_id,
    find_by_kappa_entity,
    get_all_records,
    REGISTRY_FILE,
)


def test_register_and_find():
    """Регистрация и поиск пациента."""
    # Сохраняем оригинал
    original = REGISTRY_FILE.read_text() if REGISTRY_FILE.exists() else None

    try:
        record = register_patient(
            bids_id="sub-001_ses-001",
            study_hash="test_hash_abc123",
            original_patient_id="UPENN-GBM-00001",
            patient_name="John Doe",
            scan_date="20020206",
            study_instance_uid="1.3.6.1.4.1.14519.5.2.1.test",
            kappa_entity_id="entity-uuid-test",
            kappa_dataset_id=133,
            pipeline_run_id="run-test-001",
            lesion_type="glioblastoma",
            preprocessing_id="1e2b93ad",
        )

        assert record["study_hash"] == "test_hash_abc123"
        assert record["original_patient_id"] == "UPENN-GBM-00001"
        print(f"OK: registered {record['study_hash']}")

        # Поиск по study_hash
        found = find_by_study_hash("test_hash_abc123")
        assert found is not None
        assert found["patient_name"] == "John Doe"
        print(f"OK: found by study_hash")

        # Поиск по patient_id
        found_list = find_by_patient_id("UPENN-GBM-00001")
        assert len(found_list) >= 1
        print(f"OK: found {len(found_list)} records by patient_id")

        # Поиск по kappa_entity_id
        found_kappa = find_by_kappa_entity("entity-uuid-test")
        assert found_kappa is not None
        print(f"OK: found by kappa_entity_id")

    finally:
        # Восстанавливаем
        if original is not None:
            REGISTRY_FILE.write_text(original)
        elif REGISTRY_FILE.exists():
            REGISTRY_FILE.unlink()


def test_update_existing():
    """Обновление записи при повторной регистрации."""
    original = REGISTRY_FILE.read_text() if REGISTRY_FILE.exists() else None

    try:
        # Первая регистрация без kappa_entity_id
        record1 = register_patient(
            bids_id="sub-002_ses-001",
            study_hash="test_hash_update",
            original_patient_id="PATIENT-002",
            pipeline_run_id="run-1",
        )
        assert record1["kappa_entity_id"] is None
        print(f"OK: registered without kappa_entity_id")

        # Обновление с kappa_entity_id
        record2 = register_patient(
            bids_id="sub-002_ses-001",
            study_hash="test_hash_update",
            original_patient_id="PATIENT-002",
            kappa_entity_id="new-entity-uuid",
            kappa_dataset_id=133,
        )
        assert record2["kappa_entity_id"] == "new-entity-uuid"
        print(f"OK: updated kappa_entity_id")

        # Проверяем что запись одна
        all_matching = [r for r in get_all_records() if r["study_hash"] == "test_hash_update"]
        assert len(all_matching) == 1
        print(f"OK: no duplicate records")

    finally:
        if original is not None:
            REGISTRY_FILE.write_text(original)
        elif REGISTRY_FILE.exists():
            REGISTRY_FILE.unlink()


def test_no_duplicate():
    """Повторная регистрация не создаёт дубликат."""
    original = REGISTRY_FILE.read_text() if REGISTRY_FILE.exists() else None

    try:
        register_patient(
            bids_id="sub-003_ses-001",
            study_hash="test_hash_nodup",
            original_patient_id="PATIENT-003",
        )
        register_patient(
            bids_id="sub-003_ses-001",
            study_hash="test_hash_nodup",
            original_patient_id="PATIENT-003",
        )

        all_records = get_all_records()
        matching = [r for r in all_records if r["study_hash"] == "test_hash_nodup"]
        assert len(matching) == 1, f"Expected 1 record, got {len(matching)}"
        print(f"OK: no duplicates on re-register")

    finally:
        if original is not None:
            REGISTRY_FILE.write_text(original)
        elif REGISTRY_FILE.exists():
            REGISTRY_FILE.unlink()


if __name__ == "__main__":
    print("=== test_register_and_find ===")
    test_register_and_find()

    print("\n=== test_update_existing ===")
    test_update_existing()

    print("\n=== test_no_duplicate ===")
    test_no_duplicate()

    print("\n=== Все тесты пройдены ===")