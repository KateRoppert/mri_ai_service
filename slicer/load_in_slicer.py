"""
Скрипт для 3D Slicer: загрузка данных пациента и открытие Segment Editor.

Вызывается автоматически агентом:
    Slicer --python-script load_in_slicer.py --python-arg params.json

params.json содержит:
    image_paths: список путей к preprocessed MRI
    mask_path: путь к маске сегментации
    native_image_paths: список путей к нативным MRI (опционально)
    native_mask_path: путь к нативной маске (опционально)
    patient_id: ID пациента
    session_id: ID сессии
"""
import json
import sys
import os

def main():
    # Получаем путь к файлу параметров
    # Slicer передаёт аргументы через sys.argv
    params_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--python-arg" and i + 1 < len(sys.argv):
            params_file = sys.argv[i + 1]
            break
        # Также пробуем последний аргумент
        if arg.endswith(".json") and os.path.exists(arg):
            params_file = arg

    if not params_file or not os.path.exists(params_file):
        print(f"ERROR: params file not found: {params_file}")
        print(f"sys.argv: {sys.argv}")
        return

    with open(params_file, "r") as f:
        params = json.load(f)

    image_paths = params.get("image_paths", [])
    mask_path = params.get("mask_path", "")
    native_image_paths = params.get("native_image_paths", [])
    native_mask_path = params.get("native_mask_path")
    patient_id = params.get("patient_id", "")
    session_id = params.get("session_id", "")

    print(f"Loading patient data: {patient_id} / {session_id}")
    print(f"  Images: {image_paths}")
    print(f"  Mask: {mask_path}")

    # Загружаем preprocessed изображения
    volume_nodes = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            print(f"  Loading volume: {img_path}")
            node = slicer.util.loadVolume(img_path)
            if node:
                volume_nodes.append(node)
                # Извлекаем модальность из имени файла
                basename = os.path.basename(img_path).replace(".nii.gz", "")
                parts = basename.split("_")
                modality = parts[-1] if parts else basename
                node.SetName(f"{patient_id}_{modality}")

    # Загружаем нативные изображения (если есть)
    for img_path in native_image_paths:
        if os.path.exists(img_path):
            print(f"  Loading native volume: {img_path}")
            node = slicer.util.loadVolume(img_path)
            if node:
                basename = os.path.basename(img_path).replace(".nii.gz", "")
                parts = basename.split("_")
                modality = parts[-1] if parts else basename
                node.SetName(f"{patient_id}_{modality}_native")

    # Загружаем маску как сегментацию
    segmentation_node = None
    if mask_path and os.path.exists(mask_path):
        print(f"  Loading segmentation mask: {mask_path}")

        # Загружаем маску как labelmap
        labelmap_node = slicer.util.loadLabelVolume(mask_path)
        if labelmap_node:
            labelmap_node.SetName(f"{patient_id}_segmask")

            # Конвертируем labelmap в сегментацию
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode"
            )
            segmentation_node.SetName(f"{patient_id}_segmentation")

            # Импортируем labelmap в сегментацию
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmap_node, segmentation_node
            )

            # Удаляем промежуточный labelmap
            slicer.mrmlScene.RemoveNode(labelmap_node)

            # Настраиваем имена сегментов (классы опухоли)
            segment_names = {
                1: "Некротическое ядро (NCR)",
                2: "Отёк (ED)",
                3: "Неусиливающаяся опухоль (NET)",
                4: "Усиливающаяся опухоль (ET)",
            }
            segmentation = segmentation_node.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segment = segmentation.GetNthSegment(i)
                label_value = i + 1
                if label_value in segment_names:
                    segment.SetName(segment_names[label_value])

    # Настраиваем визуализацию
    if volume_nodes:
        # Первый том — основной (background)
        main_volume = volume_nodes[0]

        # Настраиваем отображение в slice views
        layout_manager = slicer.app.layoutManager()

        # Устанавливаем основной том как background
        for view_name in ["Red", "Yellow", "Green"]:
            slice_widget = layout_manager.sliceWidget(view_name)
            if slice_widget:
                slice_logic = slice_widget.sliceLogic()
                composite = slice_logic.GetSliceCompositeNode()
                composite.SetBackgroundVolumeID(main_volume.GetID())
                slice_logic.FitSliceToAll()

    # Открываем Segment Editor
    if segmentation_node and volume_nodes:
        print("  Opening Segment Editor...")

        # Переключаемся на модуль Segment Editor
        slicer.util.selectModule("SegmentEditor")

        # Устанавливаем активную сегментацию и том
        editor_widget = slicer.modules.SegmentEditorWidget
        if hasattr(editor_widget, "editor"):
            editor = editor_widget.editor
        else:
            # Для некоторых версий Slicer
            editor_node = slicer.mrmlScene.GetFirstNodeByClass(
                "vtkMRMLSegmentEditorNode"
            )
            if not editor_node:
                editor_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentEditorNode"
                )

            editor_node.SetAndObserveSegmentationNode(segmentation_node)
            editor_node.SetAndObserveMasterVolumeNode(volume_nodes[0])

    # Удаляем временный файл параметров
    try:
        os.remove(params_file)
    except Exception:
        pass

    print(f"Patient {patient_id} loaded successfully!")
    print(f"  Volumes: {len(volume_nodes)}")
    print(f"  Segmentation: {'yes' if segmentation_node else 'no'}")


# Slicer выполняет скрипт в своём Python-окружении
# где slicer модуль уже импортирован
try:
    import slicer
    # Вызываем main после полной инициализации Slicer
    slicer.app.processEvents()
    main()
except ImportError:
    print("ERROR: This script must be run inside 3D Slicer")
    print("Usage: Slicer --python-script load_in_slicer.py --python-arg params.json")
    sys.exit(1)