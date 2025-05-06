import os
import subprocess

def run_dciodvfy(input_dir, report_dir):
    summary = {}

    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        output_root = os.path.join(report_dir, rel_path)
        os.makedirs(output_root, exist_ok=True)

        dir_summary = []

        for file in files:
            if not file.lower().endswith(".dcm"):
                continue

            input_file = os.path.join(root, file)
            output_file = os.path.join(output_root, file.replace('.dcm', '_report.txt'))

            try:
                result = subprocess.run(['dciodvfy', input_file], capture_output=True, text=True, timeout=10)
                report = result.stdout + result.stderr

                with open(output_file, 'w') as f:
                    f.write(report)

                # Подсчёт ошибок и предупреждений
                errors = sum("Error" in line for line in report.splitlines())
                warnings = sum("Warning" in line for line in report.splitlines())

                dir_summary.append((file, errors, warnings))

            except Exception as e:
                with open(output_file, 'w') as f:
                    f.write(f"Error running dciodvfy: {str(e)}")
                dir_summary.append((file, -1, -1))  # ошибка при проверке

        if dir_summary:
            summary_file = os.path.join(output_root, "summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Результаты проверки DICOM в каталоге: {rel_path}\n\n")
                for filename, errors, warnings in dir_summary:
                    if errors == -1:
                        f.write(f"{filename}: ❌ Ошибка при запуске dciodvfy\n")
                    else:
                        f.write(f"{filename}: ⚠️ {errors} ошибок, 🔶 {warnings} предупреждений\n")

            summary[rel_path] = dir_summary

    return summary

input_dicom_bids_dir = "bids_data_dicom"
output_reports_dir = "dciodvfy_reports"
run_dciodvfy(input_dicom_bids_dir, output_reports_dir)
