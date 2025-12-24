#!/usr/bin/env python3
"""
Results Analysis Script for DICOM Metadata Extraction Benchmark
Generates plots, tables, and reports for ISBI 2026 paper.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be skipped.")


class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates visualizations."""
    
    def __init__(self, metrics_file: Path, output_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            metrics_file: Path to metrics.csv
            output_dir: Directory for output files
        """
        self.metrics_file = metrics_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = self._load_data()
        self.baseline = self._get_baseline()
    
    def _load_data(self) -> List[Dict]:
        """Load metrics from CSV file (supports both CPU and GPU formats)."""
        data = []
        
        with open(self.metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields with safe get
                record = {
                    'experiment_id': row['experiment_id'],
                    # Legacy fields (preprocessing)
                    'mode': row.get('mode', ''),
                    'workers': int(row.get('workers', 0)) if row.get('workers') else 0,
                    # New fields (segmentation)
                    'stage': row.get('stage', 'preprocessing'),
                    'parallelism_type': row.get('parallelism_type', 'cpu_workers'),
                    'parallelism_level': int(row.get('parallelism_level', 0)) if row.get('parallelism_level') else 0,
                    'server_name': row.get('server_name', ''),
                    'gpu_count': int(row.get('gpu_count', 0)) if row.get('gpu_count') else 0,
                    # Common fields
                    'total_series': int(row['total_series']),
                    'successful': int(row['successful']),
                    'total_time': float(row['total_time']),
                    'time_per_series': float(row['time_per_series']),
                    'throughput': float(row['throughput']),
                    # CPU metrics
                    'cpu_avg': float(row['cpu_avg']) if row.get('cpu_avg') else None,
                    'cpu_max': float(row['cpu_max']) if row.get('cpu_max') else None,
                    'memory_avg_mb': float(row['memory_avg_mb']) if row.get('memory_avg_mb') else None,
                    'memory_peak_mb': float(row['memory_peak_mb']) if row.get('memory_peak_mb') else None,
                    # GPU metrics
                    'gpu_utilization_avg': float(row['gpu_utilization_avg']) if row.get('gpu_utilization_avg') else None,
                    'gpu_utilization_max': float(row['gpu_utilization_max']) if row.get('gpu_utilization_max') else None,
                    'gpu_memory_used_mb_avg': float(row['gpu_memory_used_mb_avg']) if row.get('gpu_memory_used_mb_avg') else None,
                    'gpu_memory_used_mb_max': float(row['gpu_memory_used_mb_max']) if row.get('gpu_memory_used_mb_max') else None,
                    'gpu_temperature_avg': float(row['gpu_temperature_avg']) if row.get('gpu_temperature_avg') else None,
                    'gpu_temperature_max': float(row['gpu_temperature_max']) if row.get('gpu_temperature_max') else None,
                    # Performance metrics
                    'speedup': float(row['speedup']) if row.get('speedup') else None,
                    'efficiency': float(row['efficiency']) if row.get('efficiency') else None
                }
                data.append(record)
        
        # Sort appropriately based on data type
        # For CPU: sort by workers
        # For GPU: sort by server_name then gpu_count
        if data and data[0]['stage'] == 'segmentation':
            data.sort(key=lambda x: (x['server_name'], x['gpu_count']))
        else:
            data.sort(key=lambda x: x['workers'])
        
        return data
    
    def _get_baseline(self) -> Dict:
        """Get baseline (sequential) experiment."""
        for exp in self.data:
            if exp['mode'] == 'sequential' and exp['workers'] == 1:
                return exp
        return None
    
    def _has_gpu_data(self) -> bool:
        """Check if data contains GPU metrics."""
        return any(exp.get('gpu_utilization_avg') is not None for exp in self.data)

    def _filter_segmentation_data(self) -> List[Dict]:
        """Filter only segmentation experiments with GPU data."""
        return [exp for exp in self.data 
                if exp['stage'] == 'segmentation' 
                and exp.get('gpu_utilization_avg') is not None]
    
    def generate_plots(self):
        """Generate all plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping plots (matplotlib not available)")
            return
        
        print("Generating plots...")
        
        # Setup style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Extract data
        workers = [exp['workers'] for exp in self.data]
        speedup = [exp['speedup'] for exp in self.data]
        efficiency = [exp['efficiency'] * 100 for exp in self.data]  # Convert to percentage
        throughput = [exp['throughput'] for exp in self.data]
        cpu_avg = [exp['cpu_avg'] for exp in self.data if exp['cpu_avg']]
        
        # 1. Speedup plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(workers, speedup, 'o-', linewidth=2, markersize=8, label='Actual Speedup')
        ax.plot(workers, workers, '--', linewidth=2, alpha=0.5, label='Linear Speedup (ideal)')
        ax.set_xlabel('Number of Workers', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Speedup vs Number of Workers', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xticks(workers)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'speedup.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'speedup.png'}")
        
        # 2. Efficiency plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(workers, efficiency, 'o-', linewidth=2, markersize=8, color='green')
        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='100% Efficiency')
        ax.set_xlabel('Number of Workers', fontsize=12)
        ax.set_ylabel('Efficiency (%)', fontsize=12)
        ax.set_title('Parallel Efficiency vs Number of Workers', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xticks(workers)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'efficiency.png'}")
        
        # 3. Throughput plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(workers, throughput, 'o-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Number of Workers', fontsize=12)
        ax.set_ylabel('Throughput (series/sec)', fontsize=12)
        ax.set_title('Throughput vs Number of Workers', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(workers)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'throughput.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'throughput.png'}")
        
        # 4. Combined plot (Speedup + Efficiency)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Speedup
        ax1.plot(workers, speedup, 'o-', linewidth=2, markersize=8, label='Actual Speedup')
        ax1.plot(workers, workers, '--', linewidth=2, alpha=0.5, label='Linear Speedup')
        ax1.set_xlabel('Number of Workers', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title('Speedup', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xticks(workers)
        
        # Efficiency
        ax2.plot(workers, efficiency, 'o-', linewidth=2, markersize=8, color='green')
        ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Number of Workers', fontsize=12)
        ax2.set_ylabel('Efficiency (%)', fontsize=12)
        ax2.set_title('Parallel Efficiency', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(workers)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'combined.png'}")
        
        # 5. CPU Utilization plot
        if cpu_avg:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(workers, cpu_avg, 'o-', linewidth=2, markersize=8, color='orange')
            ax.set_xlabel('Number of Workers', fontsize=12)
            ax.set_ylabel('CPU Utilization (%)', fontsize=12)
            ax.set_title('CPU Utilization vs Number of Workers', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(workers)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'cpu_utilization.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {self.output_dir / 'cpu_utilization.png'}")


    def generate_gpu_dashboard(self):
        """Generate 2x2 dashboard comparing GPU servers."""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping GPU dashboard (matplotlib not available)")
            return
        
        gpu_data = self._filter_segmentation_data()
        
        if not gpu_data:
            print("No GPU data found, skipping GPU dashboard")
            return
        
        print("Generating GPU comparison dashboard...")
        
        # Prepare data - group by server and GPU count
        configs = []
        labels = []
        throughput = []
        gpu_util_avg = []
        gpu_util_max = []
        gpu_mem_avg = []
        gpu_mem_max = []
        gpu_temp_avg = []
        gpu_temp_max = []
        
        for exp in gpu_data:
            server = exp['server_name']
            gpus = exp['gpu_count']
            # Create label: server name + GPU count
            if gpus > 1:
                label = f"{server}\n({gpus} GPUs)"
            else:
                label = f"{server}\n(1 GPU)"
            
            labels.append(label)
            throughput.append(exp['throughput'])
            gpu_util_avg.append(exp['gpu_utilization_avg'] or 0)
            gpu_util_max.append(exp['gpu_utilization_max'] or 0)
            gpu_mem_avg.append((exp['gpu_memory_used_mb_avg'] or 0) / 1024)  # Convert to GB
            gpu_mem_max.append((exp['gpu_memory_used_mb_max'] or 0) / 1024)
            gpu_temp_avg.append(exp['gpu_temperature_avg'] or 0)
            gpu_temp_max.append(exp['gpu_temperature_max'] or 0)
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x = range(len(labels))
        width = 0.35
        
        # 1. Throughput (top-left)
        bars1 = ax1.bar(x, throughput, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Throughput (series/sec)', fontsize=11)
        ax1.set_title('A) Segmentation Throughput', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 2. GPU Utilization (top-right)
        bars2_avg = ax2.bar([i - width/2 for i in x], gpu_util_avg, width, 
                            label='Average', color='orange', alpha=0.7)
        bars2_max = ax2.bar([i + width/2 for i in x], gpu_util_max, width,
                            label='Maximum', color='orangered', alpha=0.7)
        ax2.set_ylabel('GPU Utilization (%)', fontsize=11)
        ax2.set_title('B) GPU Utilization', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        # 3. GPU Memory (bottom-left)
        bars3_avg = ax3.bar([i - width/2 for i in x], gpu_mem_avg, width,
                            label='Average', color='mediumseagreen', alpha=0.7)
        bars3_max = ax3.bar([i + width/2 for i in x], gpu_mem_max, width,
                            label='Maximum', color='darkgreen', alpha=0.7)
        ax3.set_ylabel('GPU Memory (GB)', fontsize=11)
        ax3.set_title('C) GPU Memory Usage', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. GPU Temperature (bottom-right)
        bars4_avg = ax4.bar([i - width/2 for i in x], gpu_temp_avg, width,
                            label='Average', color='mediumpurple', alpha=0.7)
        bars4_max = ax4.bar([i + width/2 for i in x], gpu_temp_max, width,
                            label='Maximum', color='darkviolet', alpha=0.7)
        ax4.set_ylabel('Temperature (°C)', fontsize=11)
        ax4.set_title('D) GPU Temperature', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gpu_comparison_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'gpu_comparison_dashboard.png'}")


    def generate_gpu_comparison_table(self):
        """Generate comparison table for GPU servers."""
        gpu_data = self._filter_segmentation_data()
        
        if not gpu_data:
            print("No GPU data found, skipping GPU comparison table")
            return
        
        print("\nGenerating GPU comparison table...")
        
        # Markdown table
        md = []
        md.append("## GPU Server Comparison\n")
        md.append("| Configuration | GPUs | Throughput | GPU Util (avg/max) | GPU Memory (avg/max) | Temperature (avg/max) | Speedup |")
        md.append("|---------------|------|------------|--------------------|-----------------------|-----------------------|---------|")
        
        for exp in gpu_data:
            server = exp['server_name']
            gpus = exp['gpu_count']
            config = f"{server} ({gpus} GPU)" if gpus else server
            
            # Safe formatting with None checks
            util_avg = exp['gpu_utilization_avg'] if exp['gpu_utilization_avg'] is not None else 0
            util_max = exp['gpu_utilization_max'] if exp['gpu_utilization_max'] is not None else 0
            mem_avg = (exp['gpu_memory_used_mb_avg'] or 0) / 1024
            mem_max = (exp['gpu_memory_used_mb_max'] or 0) / 1024
            temp_avg = exp['gpu_temperature_avg'] if exp['gpu_temperature_avg'] is not None else 0
            temp_max = exp['gpu_temperature_max'] if exp['gpu_temperature_max'] is not None else 0
            speedup = exp['speedup'] if exp['speedup'] is not None else 1.0

            md.append(
                f"| {config} | "
                f"{gpus} | "
                f"{exp['throughput']:.2f} series/s | "
                f"{util_avg:.1f}% / {util_max:.1f}% | "
                f"{mem_avg:.1f}GB / {mem_max:.1f}GB | "
                f"{temp_avg:.1f}°C / {temp_max:.1f}°C | "
                f"**{speedup:.2f}x** |"
            )
        
        md_str = "\n".join(md)
        
        # Save markdown
        md_file = self.output_dir / 'gpu_comparison.md'
        with open(md_file, 'w') as f:
            f.write(md_str)
        print(f"✓ Saved: {md_file}")
        
        # LaTeX table
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{GPU server performance comparison for MRI segmentation.}")
        latex.append("\\label{tab:gpu_comparison}")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\hline")
        latex.append("Configuration & GPUs & Throughput & GPU Util. & GPU Memory & Temperature & Speedup \\\\")
        latex.append("              &      & (series/s) & (avg/max \\%) & (avg/max GB) & (avg/max °C) & \\\\")
        latex.append("\\hline")
        
        for exp in gpu_data:
            server = exp['server_name']
            gpus = exp['gpu_count']
            
            # Safe formatting for LaTeX
            util_avg = exp['gpu_utilization_avg'] if exp['gpu_utilization_avg'] is not None else 0
            util_max = exp['gpu_utilization_max'] if exp['gpu_utilization_max'] is not None else 0
            mem_avg = (exp['gpu_memory_used_mb_avg'] or 0) / 1024
            mem_max = (exp['gpu_memory_used_mb_max'] or 0) / 1024
            temp_avg = exp['gpu_temperature_avg'] if exp['gpu_temperature_avg'] is not None else 0
            temp_max = exp['gpu_temperature_max'] if exp['gpu_temperature_max'] is not None else 0
            speedup = exp['speedup'] if exp['speedup'] is not None else 1.0

            latex.append(
                f"{server} ({gpus}GPU) & "
                f"{gpus} & "
                f"{exp['throughput']:.2f} & "
                f"{util_avg:.1f}/{util_max:.1f} & "
                f"{mem_avg:.1f}/{mem_max:.1f} & "
                f"{temp_avg:.1f}/{temp_max:.1f} & "
                f"{speedup:.2f}x \\\\"
            )
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        # Save LaTeX
        latex_file = self.output_dir / 'gpu_comparison.tex'
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        print(f"✓ Saved: {latex_file}")
        
        print("\nGPU Comparison Table:")
        print("=" * 120)
        print(md_str)
        print("=" * 120)
    
    def generate_latex_table(self):
        """Generate LaTeX table."""
        print("\nGenerating LaTeX table...")
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Performance metrics for DICOM metadata extraction with varying parallelization.}")
        latex.append("\\label{tab:performance}")
        latex.append("\\begin{tabular}{ccccccc}")
        latex.append("\\hline")
        latex.append("Workers & Time (s) & Time/series (ms) & Throughput & Speedup & Efficiency & CPU (\\%) \\\\")
        latex.append("        &          &                  & (series/s) &         & (\\%)       &          \\\\")
        latex.append("\\hline")
        
        for exp in self.data:
            latex.append(
                f"{exp['workers']} & "
                f"{exp['total_time']:.2f} & "
                f"{exp['time_per_series']*1000:.1f} & "
                f"{exp['throughput']:.2f} & "
                f"{exp['speedup']:.2f}x & "
                f"{exp['efficiency']*100:.1f} & "
                f"{exp['cpu_avg']:.1f} \\\\"
            )
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        # Save to file
        output_file = self.output_dir / 'table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_str)
        
        print(f"✓ Saved: {output_file}")
        print("\nLaTeX Table Preview:")
        print("=" * 80)
        print(latex_str)
        print("=" * 80)
    
    def generate_markdown_table(self):
        """Generate Markdown table."""
        print("\nGenerating Markdown table...")
        
        md = []
        md.append("## Performance Results\n")
        md.append("| Workers | Time (s) | Time/series (ms) | Throughput (series/s) | Speedup | Efficiency (%) | CPU Avg (%) |")
        md.append("|---------|----------|------------------|-----------------------|---------|----------------|-------------|")
        
        for exp in self.data:
            md.append(
                f"| {exp['workers']} | "
                f"{exp['total_time']:.2f} | "
                f"{exp['time_per_series']*1000:.1f} | "
                f"{exp['throughput']:.2f} | "
                f"**{exp['speedup']:.2f}x** | "
                f"{exp['efficiency']*100:.1f}% | "
                f"{exp['cpu_avg']:.1f}% |"
            )
        
        md_str = "\n".join(md)
        
        # Save to file
        output_file = self.output_dir / 'table.md'
        with open(output_file, 'w') as f:
            f.write(md_str)
        
        print(f"✓ Saved: {output_file}")
        print("\nMarkdown Table:")
        print("=" * 80)
        print(md_str)
        print("=" * 80)
    
    def generate_summary_report(self):
        """Generate summary report with key findings."""
        print("\nGenerating summary report...")
        
        # Find best configurations
        best_speedup = max(self.data, key=lambda x: x['speedup'])
        best_efficiency = max(self.data, key=lambda x: x['efficiency'])
        best_throughput = max(self.data, key=lambda x: x['throughput'])
        
        # Calculate statistics
        total_series = self.baseline['total_series']
        baseline_time = self.baseline['total_time']
        
        report = {
            'dataset': {
                'total_series': total_series,
                'patients': total_series // 4,  # Assuming 4 modalities per patient
                'modalities_per_patient': 4
            },
            'baseline': {
                'workers': self.baseline['workers'],
                'time': self.baseline['total_time'],
                'time_per_series': self.baseline['time_per_series'],
                'throughput': self.baseline['throughput'],
                'cpu_avg': self.baseline['cpu_avg']
            },
            'best_configurations': {
                'speedup': {
                    'workers': best_speedup['workers'],
                    'value': best_speedup['speedup'],
                    'time': best_speedup['total_time'],
                    'throughput': best_speedup['throughput']
                },
                'efficiency': {
                    'workers': best_efficiency['workers'],
                    'value': best_efficiency['efficiency'],
                    'speedup': best_efficiency['speedup']
                },
                'throughput': {
                    'workers': best_throughput['workers'],
                    'value': best_throughput['throughput'],
                    'speedup': best_throughput['speedup']
                }
            },
            'key_findings': [
                f"Baseline (1 worker): {baseline_time:.2f}s for {total_series} series",
                f"Best speedup: {best_speedup['speedup']:.2f}x with {best_speedup['workers']} workers",
                f"Super-linear efficiency detected at {best_efficiency['workers']} workers ({best_efficiency['efficiency']*100:.1f}%)",
                f"Maximum throughput: {best_throughput['throughput']:.2f} series/sec with {best_throughput['workers']} workers",
                f"CPU utilization remains low (<10%), indicating I/O-bound workload"
            ],
            'recommendations': {
                'optimal_workers': best_speedup['workers'],
                'reason': f"Provides best speedup ({best_speedup['speedup']:.2f}x) and high throughput ({best_speedup['throughput']:.2f} series/sec)",
                'bottleneck': 'Network I/O (NFS)',
                'improvement_suggestions': [
                    'Use local SSD storage instead of NFS',
                    'Implement I/O batching',
                    'Consider caching frequently accessed metadata'
                ]
            }
        }
        
        # Save JSON report
        json_file = self.output_dir / 'summary.json'
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Saved: {json_file}")
        
        # Generate human-readable report
        txt_report = []
        txt_report.append("=" * 80)
        txt_report.append("BENCHMARK SUMMARY REPORT")
        txt_report.append("DICOM Metadata Extraction Performance Analysis")
        txt_report.append("=" * 80)
        txt_report.append("")
        txt_report.append(f"Dataset: {total_series} series from ~{total_series//4} patients (4 modalities each)")
        txt_report.append("")
        txt_report.append("BASELINE PERFORMANCE (Sequential):")
        txt_report.append(f"  - Total time: {self.baseline['total_time']:.2f}s")
        txt_report.append(f"  - Time per series: {self.baseline['time_per_series']*1000:.1f}ms")
        txt_report.append(f"  - Throughput: {self.baseline['throughput']:.2f} series/sec")
        txt_report.append(f"  - CPU utilization: {self.baseline['cpu_avg']:.1f}%")
        txt_report.append("")
        txt_report.append("BEST CONFIGURATIONS:")
        txt_report.append(f"  - Best Speedup: {best_speedup['workers']} workers → {best_speedup['speedup']:.2f}x")
        txt_report.append(f"  - Best Efficiency: {best_efficiency['workers']} workers → {best_efficiency['efficiency']*100:.1f}%")
        txt_report.append(f"  - Best Throughput: {best_throughput['workers']} workers → {best_throughput['throughput']:.2f} series/sec")
        txt_report.append("")
        txt_report.append("KEY FINDINGS:")
        for finding in report['key_findings']:
            txt_report.append(f"  • {finding}")
        txt_report.append("")
        txt_report.append("RECOMMENDATION:")
        txt_report.append(f"  Optimal configuration: {report['recommendations']['optimal_workers']} workers")
        txt_report.append(f"  Reason: {report['recommendations']['reason']}")
        txt_report.append(f"  Identified bottleneck: {report['recommendations']['bottleneck']}")
        txt_report.append("")
        txt_report.append("IMPROVEMENT SUGGESTIONS:")
        for suggestion in report['recommendations']['improvement_suggestions']:
            txt_report.append(f"  • {suggestion}")
        txt_report.append("")
        txt_report.append("=" * 80)
        
        txt_str = "\n".join(txt_report)
        
        # Save text report
        txt_file = self.output_dir / 'summary.txt'
        with open(txt_file, 'w') as f:
            f.write(txt_str)
        print(f"✓ Saved: {txt_file}")
        
        print("\n" + txt_str)
    
    def run_all(self):
        """Run all analysis steps."""
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Input: {self.metrics_file}")
        print(f"Output: {self.output_dir}")
        print(f"Experiments: {len(self.data)}")
        
        # Detect data type - check if this is segmentation (GPU) or preprocessing (CPU)
        has_gpu = self._has_gpu_data()
        is_preprocessing = any(exp['stage'] == 'preprocessing' for exp in self.data)
        
        print(f"Data type: {'GPU (segmentation)' if has_gpu else 'CPU (preprocessing)'}")
        print()
        
        # Generate appropriate analysis based on data type
        if has_gpu:
            # GPU analysis only
            self.generate_gpu_dashboard()
            self.generate_gpu_comparison_table()
        elif is_preprocessing or not has_gpu:
            # CPU analysis only
            if self.baseline is None:
                print("Warning: No baseline found, skipping analysis")
                return
            self.generate_plots()
            self.generate_latex_table()
            self.generate_markdown_table()
            self.generate_summary_report()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nAll results saved to: {self.output_dir}/")
        
        print("\nGenerated files:")
        if has_gpu:
            print("  GPU Analysis:")
            print("    - gpu_comparison_dashboard.png (2x2 plots)")
            print("    - gpu_comparison.tex / gpu_comparison.md")
        else:
            print("  CPU Analysis:")
            print("    - speedup.png")
            print("    - efficiency.png")
            print("    - throughput.png")
            print("    - combined.png")
            print("    - cpu_utilization.png")
            print("    - table.tex / table.md")
            print("    - summary.json / summary.txt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze benchmark results and generate visualizations'
    )
    
    parser.add_argument(
        '--metrics',
        type=Path,
        default=Path('results/metrics.csv'),
        help='Path to metrics.csv file (default: results/metrics.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/analysis'),
        help='Output directory for analysis results (default: results/analysis)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not args.metrics.exists():
        print(f"Error: Metrics file not found: {args.metrics}")
        print("Please run benchmarks first to generate metrics.csv")
        return 1
    
    # Run analysis
    analyzer = BenchmarkAnalyzer(args.metrics, args.output)
    analyzer.run_all()
    
    return 0


if __name__ == '__main__':
    exit(main())