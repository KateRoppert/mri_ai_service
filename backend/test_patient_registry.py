"""
Тесты для patient_registry (SQLite) и validation_service.
Запуск: python test_patient_registry.py
"""
from patient_registry import (
    register_patient,
    find_by_study_hash,
    find_by_patient_id,
    find_by_kappa_entity,
    find_by_run_id,
    get_all_records,
    ensure_tables,
)
from validation_service import (
    record_action,
    get_entity_history,
    get_current_votes,
    get_user_current_vote,
)
from database import SessionLocal
from registry_models import PatientRegistry, Validation


def cleanup_test_data():
    """Удалить тестовые записи."""
    db = SessionLocal()
    try:
        db.query(PatientRegistry).filter(
            PatientRegistry.study_hash.like("test_%")
        ).delete(synchronize_session=False)
        db.query(Validation).filter(
            Validation.entity_id.like("test-%")
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def test_register_and_find():
    """Регистрация и поиск пациента."""
    cleanup_test_data()

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
    print(f"OK: registered {record['study_hash']}")

    found = find_by_study_hash("test_hash_abc123")
    assert found is not None
    assert found["patient_name"] == "John Doe"
    print(f"OK: found by study_hash")

    found_list = find_by_patient_id("UPENN-GBM-00001")
    assert len(found_list) >= 1
    print(f"OK: found {len(found_list)} by patient_id")

    found_kappa = find_by_kappa_entity("entity-uuid-test")
    assert found_kappa is not None
    print(f"OK: found by kappa_entity_id")

    found_run = find_by_run_id("run-test-001")
    assert len(found_run) == 1
    print(f"OK: found by run_id")

    cleanup_test_data()


def test_update_existing():
    """Обновление kappa_entity_id при повторной регистрации."""
    cleanup_test_data()

    record1 = register_patient(
        bids_id="sub-002_ses-001",
        study_hash="test_hash_update",
        original_patient_id="PATIENT-002",
        pipeline_run_id="run-1",
    )
    assert record1["kappa_entity_id"] is None
    print(f"OK: registered without kappa_entity_id")

    record2 = register_patient(
        bids_id="sub-002_ses-001",
        study_hash="test_hash_update",
        original_patient_id="PATIENT-002",
        kappa_entity_id="new-entity-uuid",
        kappa_dataset_id=133,
    )
    assert record2["kappa_entity_id"] == "new-entity-uuid"
    print(f"OK: updated kappa_entity_id")

    all_matching = [r for r in get_all_records() if r["study_hash"] == "test_hash_update"]
    assert len(all_matching) == 1
    print(f"OK: no duplicate records")

    cleanup_test_data()


def test_validation_single_user():
    """Один пользователь: confirm → revoke → reject."""
    cleanup_test_data()

    entity_id = "test-entity-001"

    # Подтверждение
    record_action(entity_id=entity_id, dataset_id=133, user_id=26, user_name="Kate", action="confirm")
    vote = get_user_current_vote(entity_id, 26)
    assert vote == "confirm", f"Expected confirm, got {vote}"
    print(f"OK: confirm recorded")

    # Отзыв
    record_action(entity_id=entity_id, dataset_id=133, user_id=26, action="revoke")
    vote = get_user_current_vote(entity_id, 26)
    assert vote is None, f"Expected None after revoke, got {vote}"
    print(f"OK: revoke removes vote")

    # Отклонение
    record_action(entity_id=entity_id, dataset_id=133, user_id=26, action="reject")
    vote = get_user_current_vote(entity_id, 26)
    assert vote == "reject"
    print(f"OK: reject recorded")

    # Проверяем историю
    history = get_entity_history(entity_id)
    assert len(history) == 3
    print(f"OK: history has {len(history)} entries")

    cleanup_test_data()


def test_validation_multiple_users():
    """Несколько пользователей голосуют."""
    cleanup_test_data()

    entity_id = "test-entity-002"

    record_action(entity_id=entity_id, dataset_id=133, user_id=1, user_name="Alice", action="confirm")
    record_action(entity_id=entity_id, dataset_id=133, user_id=2, user_name="Bob", action="confirm")
    record_action(entity_id=entity_id, dataset_id=133, user_id=3, user_name="Carol", action="reject")

    votes = get_current_votes(entity_id)
    assert votes["confirms_count"] == 2
    assert votes["rejects_count"] == 1
    assert votes["total_votes"] == 3
    print(f"OK: 2 confirms + 1 reject")

    # Alice отзывает свой голос
    record_action(entity_id=entity_id, dataset_id=133, user_id=1, action="revoke")
    votes = get_current_votes(entity_id)
    assert votes["confirms_count"] == 1, f"Expected 1 confirm after Alice revoke, got {votes['confirms_count']}"
    assert votes["rejects_count"] == 1
    print(f"OK: revoke reduces confirms to 1")

    # Alice меняет решение — голосует против
    record_action(entity_id=entity_id, dataset_id=133, user_id=1, user_name="Alice", action="reject")
    votes = get_current_votes(entity_id)
    assert votes["confirms_count"] == 1, f"Expected 1 confirm, got {votes['confirms_count']}"
    assert votes["rejects_count"] == 2, f"Expected 2 rejects, got {votes['rejects_count']}"
    print(f"OK: Alice changed to reject — 1 confirm + 2 rejects")

    cleanup_test_data()


def test_invalid_action():
    """Неправильное действие выбрасывает ошибку."""
    try:
        record_action(entity_id="test-x", dataset_id=133, user_id=1, action="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print(f"OK: invalid action rejected")


if __name__ == "__main__":
    ensure_tables()

    print("=== test_register_and_find ===")
    test_register_and_find()

    print("\n=== test_update_existing ===")
    test_update_existing()

    print("\n=== test_validation_single_user ===")
    test_validation_single_user()

    print("\n=== test_validation_multiple_users ===")
    test_validation_multiple_users()

    print("\n=== test_invalid_action ===")
    test_invalid_action()

    print("\n=== Все тесты пройдены ===")