import os
import json
from pathlib import Path

# Папка с результатами MRIQC
base_dir = Path("mriqc_output")
output_dir = Path("mriqc_interpretation")
output_dir.mkdir(exist_ok=True)

# Шаблон интерпретации
def interpret_metrics(metrics, filename):
    lines = [f"Интерпретация результатов MRIQC для файла {filename}\n"]

    cjv = metrics.get("cjv", None)
    if cjv is not None:
        if cjv > 1:
            lines.append(f"- CJV (коэффициент вариации между тканями): {cjv:.3f} — высокий, может указывать на шум или движение.")
        else:
            lines.append(f"- CJV: {cjv:.3f} — в пределах нормы, сигнал и шум сбалансированы.")

    cnr = metrics.get("cnr", None)
    if cnr is not None:
        if cnr < 1:
            lines.append(f"- CNR (контраст между серым и белым веществом): {cnr:.3f} — низкий, может быть трудно различить ткани.")
        else:
            lines.append(f"- CNR: {cnr:.3f} — приемлемый контраст между тканями.")

    efc = metrics.get("efc", None)
    if efc is not None:
        lines.append(f"- EFC (энтропия фурье-коэффициентов): {efc:.3f} — {'высокое значение может говорить о шуме' if efc > 0.5 else 'в пределах нормы'}.")

    inu = metrics.get("inu_med", None)
    if inu is not None:
        if inu < 0.4 or inu > 1.5:
            lines.append(f"- INU (неоднородность интенсивности): {inu:.3f} — возможна неоднородность из-за сканера или движения.")
        else:
            lines.append(f"- INU: {inu:.3f} — равномерное освещение изображения.")

    snr_total = metrics.get("snr_total", None)
    if snr_total is not None:
        if snr_total < 5:
            lines.append(f"- SNR (отношение сигнал/шум): {snr_total:.3f} — низкое, изображение может быть шумным.")
        else:
            lines.append(f"- SNR: {snr_total:.3f} — хорошее качество сигнала.")

    warnings = metrics.get("provenance", {}).get("warnings", {})
    if warnings:
        for k, v in warnings.items():
            if v:
                lines.append(f"- Предупреждение: {k.replace('_', ' ')} — присутствует.")

    lines.append("\nОбщий вывод:")
    if cjv and cjv > 1 or snr_total and snr_total < 5:
        lines.append("⚠️ Возможны артефакты или шум. Рекомендуется дополнительная проверка.")
    else:
        lines.append("✅ Качество изображения в целом хорошее.")

    return "\n".join(lines)


# Поиск и обработка всех JSON файлов в структуре mriqc_output
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".json") and not file.startswith("dataset_description"):
            filepath = Path(root) / file
            with open(filepath, "r") as f:
                try:
                    metrics = json.load(f)
                    interpretation = interpret_metrics(metrics, file)
                    # Сохраняем результат
                    output_file = output_dir / f"{file.replace('.json', '')}_interpretation.txt"
                    with open(output_file, "w") as out_f:
                        out_f.write(interpretation)
                except Exception as e:
                    print(f"Ошибка при обработке {file}: {e}")

