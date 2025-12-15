#!/usr/bin/env python3
"""
GPU Monitor - мониторинг загрузки GPU в реальном времени

Этот скрипт отслеживает использование GPU во время тестирования параллелизма.
Запускайте его в отдельном терминале во время выполнения test_parallel_requests.py

ИСПОЛЬЗОВАНИЕ:
    # На сервере (barguzin):
    python gpu_monitor.py --interval 1 --duration 300
    
    # Или просто используйте watch + nvidia-smi:
    watch -n 1 nvidia-smi
"""

import subprocess
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import re

class GPUMonitor:
    """Класс для мониторинга GPU"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.history = []
        
    def get_gpu_info(self) -> Optional[List[Dict[str, any]]]:
        """
        Получает информацию о GPU через nvidia-smi
        
        Returns:
            Список словарей с информацией о каждом GPU или None при ошибке
        """
        try:
            # Запрос информации о GPU
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_total': int(parts[4]),
                        'temperature': int(parts[5])
                    })
            
            return gpus
            
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None
    
    def get_gpu_processes(self) -> Optional[List[Dict[str, any]]]:
        """
        Получает список процессов, использующих GPU
        
        Returns:
            Список словарей с информацией о процессах или None при ошибке
        """
        try:
            cmd = [
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    processes.append({
                        'pid': int(parts[0]),
                        'name': parts[1],
                        'gpu_uuid': parts[2],
                        'memory': int(parts[3])
                    })
            
            return processes
            
        except Exception as e:
            print(f"Error getting GPU processes: {e}")
            return None
    
    def print_snapshot(self, timestamp: datetime, gpus: List[Dict], processes: Optional[List[Dict]] = None):
        """Выводит снимок состояния GPU"""
        
        print("\n" + "="*80)
        print(f"⏰ {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print("="*80)
        
        if not gpus:
            print("❌ No GPU data available")
            return
        
        for gpu in gpus:
            util_bar = self.make_bar(gpu['utilization'], 100, width=20)
            mem_used_gb = gpu['memory_used'] / 1024
            mem_total_gb = gpu['memory_total'] / 1024
            mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100 if gpu['memory_total'] > 0 else 0
            mem_bar = self.make_bar(int(mem_percent), 100, width=20)
            
            print(f"\n🎮 GPU {gpu['index']}: {gpu['name']}")
            print(f"   Utilization: {util_bar} {gpu['utilization']:3d}%")
            print(f"   Memory:      {mem_bar} {mem_used_gb:.1f}/{mem_total_gb:.1f} GB ({mem_percent:.1f}%)")
            print(f"   Temperature: {gpu['temperature']}°C")
        
        if processes:
            print(f"\n📋 Active processes: {len(processes)}")
            for proc in processes:
                mem_gb = proc['memory'] / 1024
                print(f"   PID {proc['pid']}: {proc['name']} ({mem_gb:.1f} GB)")
        
        print("="*80)
    
    def make_bar(self, value: int, max_value: int, width: int = 20) -> str:
        """Создает ASCII прогресс-бар"""
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f'[{bar}]'
    
    def monitor(self, duration: Optional[float] = None, output_file: Optional[str] = None):
        """
        Запускает мониторинг GPU
        
        Args:
            duration: Длительность мониторинга в секундах (None = бесконечно)
            output_file: Файл для сохранения истории (CSV формат)
        """
        print("\n" + "="*80)
        print("🚀 GPU MONITORING STARTED")
        print("="*80)
        print(f"Interval: {self.interval}s")
        if duration:
            print(f"Duration: {duration}s")
        print("Press Ctrl+C to stop")
        print("="*80)
        
        start_time = time.time()
        
        # Открываем файл для записи если указан
        csv_file = None
        if output_file:
            csv_file = open(output_file, 'w')
            csv_file.write("timestamp,gpu_id,utilization,memory_used,memory_total,temperature\n")
        
        try:
            while True:
                current_time = datetime.now()
                elapsed = time.time() - start_time
                
                # Проверяем, не истекло ли время
                if duration and elapsed >= duration:
                    break
                
                # Получаем данные
                gpus = self.get_gpu_info()
                processes = self.get_gpu_processes()
                
                # Выводим на экран
                self.print_snapshot(current_time, gpus or [], processes)
                
                # Сохраняем в историю
                if gpus:
                    self.history.append({
                        'timestamp': current_time,
                        'gpus': gpus,
                        'processes': processes
                    })
                    
                    # Записываем в CSV
                    if csv_file:
                        for gpu in gpus:
                            csv_file.write(
                                f"{current_time.isoformat()},"
                                f"{gpu['index']},"
                                f"{gpu['utilization']},"
                                f"{gpu['memory_used']},"
                                f"{gpu['memory_total']},"
                                f"{gpu['temperature']}\n"
                            )
                        csv_file.flush()
                
                # Ждем до следующей итерации
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Monitoring stopped by user")
        
        finally:
            if csv_file:
                csv_file.close()
                print(f"\n💾 Data saved to: {output_file}")
        
        # Выводим сводку
        self.print_summary()
    
    def print_summary(self):
        """Выводит сводку по собранным данным"""
        if not self.history:
            return
        
        print("\n" + "="*80)
        print("📊 MONITORING SUMMARY")
        print("="*80)
        
        # Собираем статистику по каждому GPU
        gpu_stats = {}
        
        for snapshot in self.history:
            for gpu in snapshot['gpus']:
                gpu_id = gpu['index']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = {
                        'utilization': [],
                        'memory_used': [],
                        'temperature': []
                    }
                
                gpu_stats[gpu_id]['utilization'].append(gpu['utilization'])
                gpu_stats[gpu_id]['memory_used'].append(gpu['memory_used'])
                gpu_stats[gpu_id]['temperature'].append(gpu['temperature'])
        
        # Выводим статистику
        for gpu_id, stats in sorted(gpu_stats.items()):
            print(f"\n🎮 GPU {gpu_id}:")
            
            util = stats['utilization']
            print(f"   Utilization:")
            print(f"     Average: {sum(util) / len(util):.1f}%")
            print(f"     Max:     {max(util)}%")
            print(f"     Min:     {min(util)}%")
            
            mem = stats['memory_used']
            print(f"   Memory (GB):")
            print(f"     Average: {sum(mem) / len(mem) / 1024:.1f}")
            print(f"     Max:     {max(mem) / 1024:.1f}")
            print(f"     Min:     {min(mem) / 1024:.1f}")
            
            temp = stats['temperature']
            print(f"   Temperature:")
            print(f"     Average: {sum(temp) / len(temp):.1f}°C")
            print(f"     Max:     {max(temp)}°C")
            print(f"     Min:     {min(temp)}°C")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU usage in real-time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration of monitoring in seconds (default: infinite)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for data logging"
    )
    
    args = parser.parse_args()
    
    # Проверяем наличие nvidia-smi
    try:
        subprocess.run(['nvidia-smi', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: nvidia-smi not found. This script requires NVIDIA GPU and drivers.")
        return 1
    
    # Запускаем мониторинг
    monitor = GPUMonitor(interval=args.interval)
    monitor.monitor(duration=args.duration, output_file=args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())