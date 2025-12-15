#!/usr/bin/env python3
"""
Тест параллельных запросов к серверу сегментации

Этот скрипт отправляет несколько запросов одновременно для проверки
работы GPU pool и параллельной обработки на сервере.

ИСПОЛЬЗОВАНИЕ:
    python test_parallel_requests.py --num-requests 2 --config /path/to/segmentation_config.yaml

ТРЕБОВАНИЯ:
    pip install requests
"""

import argparse
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import yaml
import sys

class ParallelRequestTester:
    """Класс для тестирования параллельных запросов"""
    
    def __init__(self, server_url: str, timeout: int = 1800):
        self.server_url = server_url
        self.timeout = timeout
        self.results = []
        
    def send_single_request(self, request_id: int, files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Отправляет один запрос на сегментацию
        
        Args:
            request_id: Номер запроса для идентификации
            files: Словарь с путями к файлам {modality: path}
            
        Returns:
            Словарь с результатами запроса
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"📤 REQUEST {request_id}: STARTING")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        print(f"   Server: {self.server_url}")
        print(f"{'='*60}")
        
        try:
            # Проверяем, что файлы существуют
            for modality, filepath in files.items():
                if not filepath.exists():
                    raise FileNotFoundError(f"File not found: {filepath}")
            
            # Подготовка multipart данных с уникальными именами
            files_data = {}
            for modality, filepath in files.items():
                key = f'file_{modality}'
                # Добавляем request_id к имени файла для уникальности
                unique_name = f"req{request_id}_{filepath.name}"
                files_data[key] = (unique_name, open(filepath, 'rb'), 'application/gzip')
            
            # Формируем запрос
            url = f"{self.server_url}/v1/inference_async"
            data = {
                'model': 'Unet',
                'client_id': f'test_request_{request_id}'
            }
            
            print(f"🚀 REQUEST {request_id}: Sending POST to {url}")
            print(f"   Client ID: test_request_{request_id}")
            
            # Отправляем запрос
            response = requests.post(url, files=files_data, data=data, timeout=10)
            
            # Закрываем файлы
            for f in files_data.values():
                f[1].close()
            
            if response.status_code != 200:
                raise Exception(f"Server returned status {response.status_code}: {response.text}")
            
            response_data = response.json()
            task_id = response_data.get('task_id')
            
            print(f"✅ REQUEST {request_id}: Task created")
            print(f"   Task ID: {task_id}")
            print(f"   Status: {response_data.get('status')}")
            
            # Ожидаем завершения
            print(f"⏳ REQUEST {request_id}: Waiting for completion...")
            
            result = self.wait_for_completion(task_id, request_id)
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"✅ REQUEST {request_id}: COMPLETED")
            print(f"   Total time: {elapsed:.2f}s")
            print(f"   GPU used: {result.get('gpu_id', 'unknown')}")
            print(f"   Inference time: {result.get('inference_time', 'unknown')}")
            print(f"{'='*60}")
            
            return {
                'request_id': request_id,
                'task_id': task_id,
                'success': True,
                'elapsed_time': elapsed,
                'gpu_id': result.get('gpu_id'),
                'inference_time': result.get('inference_time'),
                'result': result
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"❌ REQUEST {request_id}: FAILED")
            print(f"   Error: {e}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"{'='*60}")
            
            return {
                'request_id': request_id,
                'success': False,
                'elapsed_time': elapsed,
                'error': str(e)
            }
    
    def wait_for_completion(self, task_id: str, request_id: int, poll_interval: float = 2.0) -> Dict[str, Any]:
        """
        Опрашивает статус задачи до завершения
        
        Args:
            task_id: ID задачи на сервере
            request_id: Номер запроса для логирования
            poll_interval: Интервал опроса в секундах
            
        Returns:
            Финальный статус задачи
        """
        status_url = f"{self.server_url}/get_status/{task_id}"
        last_progress = -1
        
        while True:
            try:
                response = requests.get(status_url, timeout=10)
                
                if response.status_code != 200:
                    time.sleep(poll_interval)
                    continue
                
                status_data = response.json()
                current_status = status_data.get('status')
                current_progress = status_data.get('progress', 0)
                
                # Выводим прогресс только если изменился
                if current_progress != last_progress:
                    gpu_info = f"GPU {status_data.get('gpu_id')}" if 'gpu_id' in status_data else "GPU unknown"
                    print(f"   REQUEST {request_id}: {current_status} - {current_progress}% ({gpu_info})")
                    last_progress = current_progress
                
                if current_status == 'completed':
                    return status_data
                elif current_status == 'failed':
                    error = status_data.get('error', 'Unknown error')
                    raise Exception(f"Task failed: {error}")
                
                time.sleep(poll_interval)
                
            except requests.RequestException as e:
                print(f"   REQUEST {request_id}: Connection error while polling: {e}")
                time.sleep(poll_interval)
    
    def run_parallel_test(self, num_requests: int, files: Dict[str, Path], max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Запускает несколько запросов параллельно
        
        Args:
            num_requests: Количество запросов
            files: Файлы для сегментации
            max_workers: Максимальное количество параллельных потоков (None = num_requests)
            
        Returns:
            Список результатов
        """
        if max_workers is None:
            max_workers = num_requests
        
        print("\n" + "="*80)
        print(f"🚀 PARALLEL TEST STARTING")
        print(f"   Requests: {num_requests}")
        print(f"   Max parallel: {max_workers}")
        print(f"   Server: {self.server_url}")
        print("="*80)
        
        overall_start = time.time()
        
        # Запускаем запросы параллельно
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.send_single_request, i, files): i 
                for i in range(1, num_requests + 1)
            }
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        overall_elapsed = time.time() - overall_start
        
        # Анализ результатов
        self.print_summary(results, overall_elapsed)
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]], overall_time: float):
        """Выводит сводку по результатам тестирования"""
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"Total requests:     {len(results)}")
        print(f"Successful:         {len(successful)}")
        print(f"Failed:             {len(failed)}")
        print(f"Overall time:       {overall_time:.2f}s")
        print()
        
        if successful:
            # Статистика по времени
            times = [r['elapsed_time'] for r in successful]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print("Time statistics:")
            print(f"  Average:   {avg_time:.2f}s")
            print(f"  Min:       {min_time:.2f}s")
            print(f"  Max:       {max_time:.2f}s")
            print()
            
            # Статистика по GPU
            gpu_ids = [r.get('gpu_id') for r in successful if r.get('gpu_id') is not None]
            if gpu_ids:
                unique_gpus = set(gpu_ids)
                print(f"GPUs used: {sorted(unique_gpus)}")
                for gpu in sorted(unique_gpus):
                    count = gpu_ids.count(gpu)
                    print(f"  GPU {gpu}: {count} tasks")
                print()
            
            # Детали по запросам
            print("Request details:")
            for r in sorted(successful, key=lambda x: x['request_id']):
                gpu_info = f"GPU {r.get('gpu_id')}" if r.get('gpu_id') is not None else "GPU unknown"
                print(f"  Request {r['request_id']}: {r['elapsed_time']:.2f}s ({gpu_info})")
        
        if failed:
            print("\nFailed requests:")
            for r in failed:
                print(f"  Request {r['request_id']}: {r.get('error', 'Unknown error')}")
        
        print("="*80)


def load_config(config_path: Path) -> dict:
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_test_files(input_dir: Path) -> Dict[str, Path]:
    """
    Ищет тестовые файлы в указанной директории
    
    Ожидаемая структура:
    input_dir/
        sub-XXX/
            ses-YYY/
                anat/
                    sub-XXX_ses-YYY_T1w.nii.gz
                    sub-XXX_ses-YYY_T1wCE.nii.gz
                    sub-XXX_ses-YYY_T2w.nii.gz
                    sub-XXX_ses-YYY_FLAIR.nii.gz
    """
    print(f"\n🔍 Searching for test files in: {input_dir}")
    
    # Ищем первую сессию с полным набором файлов
    for sub_dir in sorted(input_dir.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            anat_dir = ses_dir / "anat"
            if not anat_dir.exists():
                continue
            
            # Ищем файлы
            t1_files = list(anat_dir.glob("*_T1w.nii.gz"))
            t1c_files = list(anat_dir.glob("*_T1wCE.nii.gz"))
            t2_files = list(anat_dir.glob("*_T2w.nii.gz"))
            flair_files = list(anat_dir.glob("*_FLAIR.nii.gz"))
            
            if t1_files and t1c_files and t2_files and flair_files:
                files = {
                    't1': t1_files[0],
                    't1c': t1c_files[0],
                    't2': t2_files[0],
                    't2fl': flair_files[0]
                }
                
                print(f"✅ Found complete set in: {ses_dir}")
                for modality, filepath in files.items():
                    print(f"   {modality}: {filepath.name}")
                
                return files
    
    raise FileNotFoundError("No complete set of test files found")


def main():
    parser = argparse.ArgumentParser(
        description="Test parallel requests to segmentation server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/segmentation_config.yaml"),
        help="Path to segmentation config file"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=2,
        help="Number of parallel requests to send"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel threads (default: same as num-requests)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with test files (optional, for manual file selection)"
    )
    parser.add_argument(
        "--t1",
        type=Path,
        help="Path to T1 file (if not using --input-dir)"
    )
    parser.add_argument(
        "--t1c",
        type=Path,
        help="Path to T1CE file (if not using --input-dir)"
    )
    parser.add_argument(
        "--t2",
        type=Path,
        help="Path to T2 file (if not using --input-dir)"
    )
    parser.add_argument(
        "--t2fl",
        type=Path,
        help="Path to FLAIR file (if not using --input-dir)"
    )
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию
        config = load_config(args.config)
        
        # Получаем server_url из активного профиля
        active_profile = config['segmentation']['active_profile']
        profile = config['segmentation']['profiles'][active_profile]
        server_url = profile['server_url']
        
        print(f"Using profile: {active_profile}")
        print(f"Server URL: {server_url}")
        
        # Получаем тестовые файлы
        if args.t1 and args.t1c and args.t2 and args.t2fl:
            # Явно указаны файлы
            files = {
                't1': args.t1,
                't1c': args.t1c,
                't2': args.t2,
                't2fl': args.t2fl
            }
        elif args.input_dir:
            # Ищем в директории
            files = get_test_files(args.input_dir)
        else:
            print("❌ Error: Please specify either --input-dir or all four file arguments (--t1, --t1c, --t2, --t2fl)")
            sys.exit(1)
        
        # Проверяем доступность сервера
        print(f"\n🔍 Checking server availability...")
        try:
            response = requests.get(f"{server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server is accessible")
            else:
                print(f"⚠️  Server returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Server not accessible: {e}")
            sys.exit(1)
        
        # Запускаем тест
        tester = ParallelRequestTester(server_url)
        results = tester.run_parallel_test(
            num_requests=args.num_requests,
            files=files,
            max_workers=args.max_workers
        )
        
        # Возвращаем код выхода в зависимости от результатов
        failed_count = len([r for r in results if not r['success']])
        sys.exit(0 if failed_count == 0 else 1)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()