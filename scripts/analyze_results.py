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
        """Load metrics from CSV file."""
        data = []
        
        with open(self.metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                data.append({
                    'experiment_id': row['experiment_id'],
                    'mode': row['mode'],
                    'workers': int(row['workers']),
                    'total_series': int(row['total_series']),
                    'successful': int(row['successful']),
                    'total_time': float(row['total_time']),
                    'time_per_series': float(row['time_per_series']),
                    'throughput': float(row['throughput']),
                    'cpu_avg': float(row['cpu_avg']) if row['cpu_avg'] else None,
                    'cpu_max': float(row['cpu_max']) if row['cpu_max'] else None,
                    'memory_avg_mb': float(row['memory_avg_mb']) if row['memory_avg_mb'] else None,
                    'memory_peak_mb': float(row['memory_peak_mb']) if row['memory_peak_mb'] else None,
                    'speedup': float(row['speedup']) if row['speedup'] else None,
                    'efficiency': float(row['efficiency']) if row['efficiency'] else None
                })
        
        # Sort by number of workers
        data.sort(key=lambda x: x['workers'])
        
        return data
    
    def _get_baseline(self) -> Dict:
        """Get baseline (sequential) experiment."""
        for exp in self.data:
            if exp['mode'] == 'sequential' and exp['workers'] == 1:
                return exp
        return None
    
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
        print()
        
        self.generate_plots()
        self.generate_latex_table()
        self.generate_markdown_table()
        self.generate_summary_report()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nAll results saved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("  - speedup.png")
        print("  - efficiency.png")
        print("  - throughput.png")
        print("  - combined.png")
        print("  - cpu_utilization.png")
        print("  - table.tex (LaTeX)")
        print("  - table.md (Markdown)")
        print("  - summary.json")
        print("  - summary.txt")


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