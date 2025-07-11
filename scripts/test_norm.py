# import SimpleITK as sitk
# import numpy as np

# input_path = "/home/roppert/work/pipeline_results/BrainMetShare/nifti/sub-001/ses-001/anat/t1_gd.nii.gz"
# output_path = "/home/roppert/work/pipeline_results/BrainMetShare/preprocessed/t1_gd_preprocessed.nii.gz"

# img = sitk.ReadImage(input_path)
# arr = sitk.GetArrayFromImage(img)
# mask = arr > 0

# print("Original stats:")
# print(f"Min: {np.min(arr[mask])}, Max: {np.max(arr[mask])}")
# print(f"Mean: {np.mean(arr[mask])}, Std: {np.std(arr[mask])}")

# normalized = (arr - np.mean(arr[mask])) / np.std(arr[mask])
# print("\nNormalized stats:")
# print(f"Min: {np.min(normalized[mask])}, Max: {np.max(normalized[mask])}")
# print(f"Mean: {np.mean(normalized[mask]):.2f}, Std: {np.std(normalized[mask]):.2f}")

# out_img = sitk.GetImageFromArray(normalized)
# out_img.CopyInformation(img)
# sitk.WriteImage(out_img, output_path)

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def calculate_stats(image_data):
    """Вычисляет статистические характеристики для ненулевых вокселей (включая отрицательные)"""
    # Создаем маску для ненулевых значений (исключаем только нули)
    mask = image_data != 0
    masked_data = image_data[mask]
    
    # Если нет ненулевых вокселей, возвращаем нулевые статистики
    if masked_data.size == 0:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'std': 0,
            'nonzero_voxels': 0,
            'total_voxels': image_data.size
        }
    
    return {
        'min': np.min(masked_data),
        'max': np.max(masked_data),
        'mean': np.mean(masked_data),
        'std': np.std(masked_data),
        'nonzero_voxels': masked_data.size,
        'total_voxels': image_data.size
    }

def generate_report(stats1, stats2, filename1, filename2):
    """Генерирует текстовый отчет со статистикой"""
    report = "Отчет по анализу МРТ изображений (ненулевые воксели, включая отрицательные)\n"
    report += "=" * 70 + "\n\n"
    
    # Статистика для исходного изображения
    report += f"Исходное изображение: {filename1}\n"
    report += "-" * 70 + "\n"
    report += f"Общее количество вокселей: {stats1['total_voxels']}\n"
    report += f"Количество ненулевых вокселей: {stats1['nonzero_voxels']} ({stats1['nonzero_voxels']/stats1['total_voxels']*100:.2f}%)\n"
    report += f"Минимальная интенсивность: {stats1['min']:.4f}\n"
    report += f"Максимальная интенсивность: {stats1['max']:.4f}\n"
    report += f"Средняя интенсивность: {stats1['mean']:.4f}\n"
    report += f"Стандартное отклонение: {stats1['std']:.4f}\n\n"
    
    # Статистика для нормализованного изображения
    report += f"Нормализованное изображение: {filename2}\n"
    report += "-" * 70 + "\n"
    report += f"Общее количество вокселей: {stats2['total_voxels']}\n"
    report += f"Количество ненулевых вокселей: {stats2['nonzero_voxels']} ({stats2['nonzero_voxels']/stats2['total_voxels']*100:.2f}%)\n"
    report += f"Минимальная интенсивность: {stats2['min']:.4f}\n"
    report += f"Максимальная интенсивность: {stats2['max']:.4f}\n"
    report += f"Средняя интенсивность: {stats2['mean']:.4f}\n"
    report += f"Стандартное отклонение: {stats2['std']:.4f}\n\n"
    
    # Сравнение изменений
    report += "Сравнение изменений после нормализации:\n"
    report += "-" * 70 + "\n"
    report += f"Изменение диапазона интенсивностей: {stats1['max']-stats1['min']:.2f} → {stats2['max']-stats2['min']:.2f}\n"
    report += f"Изменение стандартного отклонения: {stats1['std']:.2f} → {stats2['std']:.2f} ({stats2['std']/stats1['std']*100 if stats1['std'] != 0 else 0:.1f}% от исходного)\n"
    report += f"Отношение сигнал/шум (среднее/отклонение): {stats1['mean']/stats1['std']:.2f} → {stats2['mean']/stats2['std']:.2f}\n\n"
    
    report += "=" * 70 + "\n"
    return report

def plot_histograms(original, normalized, save_path):
    """Создает и сохраняет гистограммы распределения интенсивностей для ненулевых вокселей"""
    # Применяем маски для ненулевых значений
    original_masked = original[original != 0]
    normalized_masked = normalized[normalized != 0]
    
    plt.figure(figsize=(16, 8))
    
    # Гистограмма исходного изображения
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(original_masked.flatten(), bins=300, alpha=0.7, color='blue', log=True)
    plt.title('Исходное изображение (ненулевые воксели)', fontsize=14)
    plt.xlabel('Интенсивность', fontsize=12)
    plt.ylabel('Логарифм частоты (log)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Добавляем вертикальные линии для статистик
    stats_original = calculate_stats(original)
    plt.axvline(stats_original['mean'], color='red', linestyle='dashed', linewidth=1.5, label=f'Среднее: {stats_original["mean"]:.2f}')
    plt.axvline(stats_original['mean'] - stats_original['std'], color='orange', linestyle='dotted', linewidth=1)
    plt.axvline(stats_original['mean'] + stats_original['std'], color='orange', linestyle='dotted', linewidth=1, label=f'±1 STD: {stats_original["std"]:.2f}')
    plt.legend()
    
    # Гистограмма нормализованного изображения
    plt.subplot(1, 2, 2)
    n, bins, patches = plt.hist(normalized_masked.flatten(), bins=300, alpha=0.7, color='green', log=True)
    plt.title('Нормализованное изображение (ненулевые воксели)', fontsize=14)
    plt.xlabel('Интенсивность', fontsize=12)
    plt.ylabel('Логарифм частоты (log)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Добавляем вертикальные линии для статистик
    stats_normalized = calculate_stats(normalized)
    plt.axvline(stats_normalized['mean'], color='red', linestyle='dashed', linewidth=1.5, label=f'Среднее: {stats_normalized["mean"]:.2f}')
    plt.axvline(stats_normalized['mean'] - stats_normalized['std'], color='orange', linestyle='dotted', linewidth=1)
    plt.axvline(stats_normalized['mean'] + stats_normalized['std'], color='orange', linestyle='dotted', linewidth=1, label=f'±1 STD: {stats_normalized["std"]:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 't2f_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Запрос путей у пользователя
    print("=" * 70)
    print("Анализ МРТ изображений до и после нормализации")
    print("=" * 70)

    # Запрос путей у пользователя
    original_path = "/home/roppert/work/pipeline_results/BRATS_MEN_25/nifti/BraTS-MEN-00084-000-t2f.nii.gz"
    normalized_path = "/home/roppert/work/pipeline_results/BRATS_MEN_25/intensity_norm/BraTS-MEN-00084-000-t2f.nii.gz"
    output_dir = "/home/roppert/work/pipeline_results/BRATS_MEN_25/validation/intensity_norm"

    # Создание папки для результатов
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Загрузка изображений
        print("\nЗагрузка изображений...")
        original_img = nib.load(original_path)
        normalized_img = nib.load(normalized_path)
        
        # Получение данных изображений
        original_data = original_img.get_fdata()
        normalized_data = normalized_img.get_fdata()

        # Вычисление статистики для ненулевых вокселей
        print("Расчет статистики для исходного изображения...")
        stats_original = calculate_stats(original_data)
        print("Расчет статистики для нормализованного изображения...")
        stats_normalized = calculate_stats(normalized_data)

        # Генерация и сохранение отчета
        report = generate_report(
            stats_original, 
            stats_normalized,
            os.path.basename(original_path),
            os.path.basename(normalized_path)
        )
        
        report_path = os.path.join(output_dir, 't2f_stats.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Создание и сохранение гистограмм
        print("Создание гистограмм...")
        plot_histograms(original_data, normalized_data, output_dir)
        
        print("\n" + "=" * 70)
        print(f"Результаты успешно сохранены в папку: {output_dir}")
        print(f"Текстовый отчет: {report_path}")
        print(f"Гистограммы: {os.path.join(output_dir, 't1c_histograms.png')}")
        print("=" * 70)
    
    except Exception as e:
        print(f"\nОшибка: {str(e)}")
        print("Возможные причины:")
        print("- Неправильный путь к файлам")
        print("- Файлы повреждены или имеют несовместимый формат")
        print("- Отсутствуют необходимые библиотеки (nibabel, numpy, matplotlib)")

if __name__ == "__main__":
    main()
