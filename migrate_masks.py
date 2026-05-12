"""
Одноразовый скрипт миграции: консолидация масок в директорию ИИ-маски.

Для каждого entity_id в mask_versions:
1. Находит директорию ИИ-маски (version 1)
2. Копирует экспертные маски из других директорий в неё
3. Переименовывает файлы по схеме {base}_segmask_v{version}.nii.gz
4. Обновляет file_path и file_name в БД

Запуск:
    cd /path/to/mri_ai_service
    python migrate_masks.py [--dry-run]
"""
import shutil
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from database import SessionLocal
from registry_models import MaskVersion


def migrate(dry_run=False):
    db = SessionLocal()
    try:
        # Получаем все entity_id
        entity_ids = [
            r[0] for r in db.query(MaskVersion.entity_id).distinct().all()
        ]

        for entity_id in entity_ids:
            print(f"\n=== Entity: {entity_id} ===")

            # Находим ИИ-маску (version 1)
            ai_mask = db.query(MaskVersion).filter(
                MaskVersion.entity_id == entity_id,
                MaskVersion.version == 1,
            ).first()

            if not ai_mask:
                print("  SKIP: no AI mask (version 1)")
                continue

            ai_dir = Path(ai_mask.file_path).parent
            ai_base = ai_mask.file_name.replace(".nii.gz", "").replace("_segmask", "")
            # ai_base = "sub-001_ses-001_t1"
            base_name = f"{ai_base}_segmask"

            print(f"  AI mask dir: {ai_dir}")
            print(f"  Base name: {base_name}")

            if not ai_dir.exists():
                print(f"  WARNING: AI mask dir does not exist!")
                continue

            # Обрабатываем все версии
            versions = db.query(MaskVersion).filter(
                MaskVersion.entity_id == entity_id,
            ).order_by(MaskVersion.version.asc()).all()

            for v in versions:
                current_path = Path(v.file_path)
                current_dir = current_path.parent

                # Правильное имя файла
                if v.version == 1:
                    correct_name = f"{base_name}.nii.gz"
                else:
                    correct_name = f"{base_name}_v{v.version}.nii.gz"

                correct_path = ai_dir / correct_name

                needs_move = current_dir != ai_dir
                needs_rename = v.file_name != correct_name

                if not needs_move and not needs_rename:
                    print(f"  v{v.version}: OK ({v.file_name})")
                    continue

                print(f"  v{v.version}: {v.file_name}")
                if needs_move:
                    print(f"    MOVE: {current_path} → {correct_path}")
                elif needs_rename:
                    print(f"    RENAME: {v.file_name} → {correct_name}")

                if dry_run:
                    print(f"    [DRY RUN] skipped")
                    continue

                # Проверяем, что исходный файл существует
                if not current_path.exists():
                    print(f"    WARNING: source file not found, updating DB only")
                    v.file_path = str(correct_path)
                    v.file_name = correct_name
                    continue

                # Копируем (не перемещаем — безопаснее)
                if needs_move or needs_rename:
                    shutil.copy2(str(current_path), str(correct_path))
                    print(f"    COPIED: {correct_path}")

                # Обновляем БД
                v.file_path = str(correct_path)
                v.file_name = correct_name

            db.commit()
            print(f"  DB updated for {len(versions)} versions")

    finally:
        db.close()


if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    if is_dry_run:
        print("=== DRY RUN MODE (no changes) ===\n")
    else:
        print("=== MIGRATION MODE (will copy files and update DB) ===\n")

    migrate(dry_run=is_dry_run)
    print("\nDone!")