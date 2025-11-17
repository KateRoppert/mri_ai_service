#!/usr/bin/env python3
"""
Validate BIDS dataset integrity.

Checks:
1. Slice count matches between source and BIDS
2. All 4 modalities present in each session
3. Generates detailed report of incomplete data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class BIDSValidator:
    """Validates BIDS dataset integrity."""
    
    REQUIRED_MODALITIES = {'t1', 't1c', 't2', 't2fl'}
    
    def __init__(self, bids_dir: Path, mapping_file: Path):
        self.bids_dir = bids_dir
        self.mapping_file = mapping_file
        self.mapping_data = None
        
        # Results
        self.slice_mismatches = []
        self.incomplete_sessions = []
        self.missing_directories = []
        
    def load_mapping(self) -> bool:
        """Load dataset mapping file."""
        if not self.mapping_file.exists():
            print(f"❌ Mapping file not found: {self.mapping_file}")
            return False
        
        try:
            with open(self.mapping_file, 'r') as f:
                self.mapping_data = json.load(f)
            print(f"✓ Loaded mapping for {len(self.mapping_data['patients'])} patients")
            return True
        except Exception as e:
            print(f"❌ Failed to load mapping: {e}")
            return False
    
    def count_dicom_files(self, directory: Path) -> int:
        """Count DICOM files in directory (including nested)."""
        if not directory.exists():
            return 0
        return len(list(directory.rglob("*.dcm")))
    
    def validate_slice_counts(self) -> Dict[str, int]:
        """
        Validate that slice counts match between source and BIDS.
        
        Returns:
            Statistics dictionary
        """
        print("\n📊 Validating slice counts...")
        
        total_series = 0
        valid_series = 0
        
        for patient_id, patient_data in self.mapping_data['patients'].items():
            for session_id, session_data in patient_data['sessions'].items():
                for modality, series_info in session_data['series'].items():
                    total_series += 1
                    
                    # Get paths
                    source_path = Path(series_info['original_path'])
                    bids_path = self.bids_dir / patient_id / session_id / 'anat' / modality
                    
                    # Count slices
                    source_count = self.count_dicom_files(source_path)
                    bids_count = self.count_dicom_files(bids_path)
                    
                    # Compare
                    if source_count != bids_count:
                        self.slice_mismatches.append({
                            'patient': patient_id,
                            'original_id': patient_data['original_id'],
                            'session': session_id,
                            'modality': modality,
                            'source_count': source_count,
                            'bids_count': bids_count,
                            'difference': bids_count - source_count
                        })
                    else:
                        valid_series += 1
                    
                    # Check if BIDS directory exists
                    if not bids_path.exists():
                        self.missing_directories.append({
                            'patient': patient_id,
                            'session': session_id,
                            'modality': modality,
                            'expected_path': str(bids_path)
                        })
        
        stats = {
            'total_series': total_series,
            'valid_series': valid_series,
            'mismatched_series': len(self.slice_mismatches),
            'missing_directories': len(self.missing_directories)
        }
        
        print(f"  Total series: {total_series}")
        print(f"  ✓ Valid: {valid_series} ({valid_series/total_series*100:.1f}%)")
        print(f"  ✗ Mismatched: {len(self.slice_mismatches)}")
        print(f"  ✗ Missing directories: {len(self.missing_directories)}")
        
        return stats
    
    def validate_completeness(self) -> Dict:
        """
        Validate that all sessions have all 4 modalities.
        
        Returns:
            Completeness statistics
        """
        print("\n🔍 Validating data completeness...")
        
        complete_patients = 0
        complete_sessions = 0
        total_sessions = 0
        
        for patient_id, patient_data in self.mapping_data['patients'].items():
            patient_incomplete_sessions = []
            
            for session_id, session_data in patient_data['sessions'].items():
                total_sessions += 1
                available_modalities = set(session_data['series'].keys())
                missing_modalities = self.REQUIRED_MODALITIES - available_modalities
                
                if missing_modalities:
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'date': session_data['original_date'],
                        'missing': sorted(list(missing_modalities)),
                        'available': sorted(list(available_modalities))
                    })
                else:
                    complete_sessions += 1
            
            if patient_incomplete_sessions:
                self.incomplete_sessions.append({
                    'patient_id': patient_id,
                    'original_id': patient_data['original_id'],
                    'incomplete_sessions': patient_incomplete_sessions
                })
            else:
                complete_patients += 1
        
        total_patients = len(self.mapping_data['patients'])
        
        stats = {
            'total_patients': total_patients,
            'complete_patients': complete_patients,
            'incomplete_patients': len(self.incomplete_sessions),
            'total_sessions': total_sessions,
            'complete_sessions': complete_sessions,
            'incomplete_sessions': total_sessions - complete_sessions
        }
        
        print(f"  Total patients: {total_patients}")
        print(f"  ✓ Complete patients: {complete_patients} ({complete_patients/total_patients*100:.1f}%)")
        print(f"  ✗ Incomplete patients: {len(self.incomplete_sessions)}")
        print(f"  Total sessions: {total_sessions}")
        print(f"  ✓ Complete sessions: {complete_sessions} ({complete_sessions/total_sessions*100:.1f}%)")
        print(f"  ✗ Incomplete sessions: {total_sessions - complete_sessions}")
        
        return stats
    
    def print_detailed_report(self):
        """Print detailed validation report."""
        print("\n" + "="*80)
        print("=== DETAILED VALIDATION REPORT ===")
        print("="*80)
        
        # Slice count mismatches
        if self.slice_mismatches:
            print("\n⚠️  SLICE COUNT MISMATCHES:")
            print(f"Found {len(self.slice_mismatches)} series with mismatched slice counts:\n")
            
            for mismatch in self.slice_mismatches[:20]:  # Show first 20
                print(f"  {mismatch['patient']} ({mismatch['original_id']})")
                print(f"    Session: {mismatch['session']}, Modality: {mismatch['modality']}")
                print(f"    Source: {mismatch['source_count']} slices")
                print(f"    BIDS:   {mismatch['bids_count']} slices")
                print(f"    Diff:   {mismatch['difference']:+d} slices")
                print()
            
            if len(self.slice_mismatches) > 20:
                print(f"  ... and {len(self.slice_mismatches) - 20} more")
        else:
            print("\n✅ SLICE COUNTS: All series validated successfully!")
        
        # Missing directories
        if self.missing_directories:
            print(f"\n⚠️  MISSING DIRECTORIES:")
            print(f"Found {len(self.missing_directories)} missing BIDS directories:\n")
            
            for missing in self.missing_directories[:10]:
                print(f"  {missing['patient']} / {missing['session']} / {missing['modality']}")
                print(f"    Expected: {missing['expected_path']}")
                print()
            
            if len(self.missing_directories) > 10:
                print(f"  ... and {len(self.missing_directories) - 10} more")
        
        # Incomplete sessions
        if self.incomplete_sessions:
            print(f"\n⚠️  INCOMPLETE DATA:")
            print(f"Found {len(self.incomplete_sessions)} patients with missing modalities:\n")
            
            for patient in self.incomplete_sessions[:10]:  # Show first 10
                print(f"  {patient['patient_id']} ({patient['original_id']})")
                for session in patient['incomplete_sessions']:
                    missing = ', '.join(session['missing'])
                    available = ', '.join(session['available'])
                    print(f"    └─ {session['session_id']} ({session['date']})")
                    print(f"       Missing: {missing}")
                    print(f"       Available: {available}")
                print()
            
            if len(self.incomplete_sessions) > 10:
                print(f"  ... and {len(self.incomplete_sessions) - 10} more")
        else:
            print("\n✅ COMPLETENESS: All patients have all 4 modalities!")
        
        print("\n" + "="*80)
    
    def save_detailed_report(self, output_file: Path):
        """Save detailed report to text file."""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BIDS DATASET VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Slice count mismatches
            if self.slice_mismatches:
                f.write(f"SLICE COUNT MISMATCHES ({len(self.slice_mismatches)} series):\n")
                f.write("-"*80 + "\n\n")
                
                for mismatch in self.slice_mismatches:
                    f.write(f"Patient: {mismatch['patient']} ({mismatch['original_id']})\n")
                    f.write(f"  Session: {mismatch['session']}\n")
                    f.write(f"  Modality: {mismatch['modality']}\n")
                    f.write(f"  Source slices: {mismatch['source_count']}\n")
                    f.write(f"  BIDS slices: {mismatch['bids_count']}\n")
                    f.write(f"  Difference: {mismatch['difference']:+d}\n")
                    f.write("\n")
            else:
                f.write("SLICE COUNTS: All validated successfully!\n\n")
            
            # Missing directories
            if self.missing_directories:
                f.write(f"\nMISSING DIRECTORIES ({len(self.missing_directories)} directories):\n")
                f.write("-"*80 + "\n\n")
                
                for missing in self.missing_directories:
                    f.write(f"Patient: {missing['patient']}\n")
                    f.write(f"  Session: {missing['session']}\n")
                    f.write(f"  Modality: {missing['modality']}\n")
                    f.write(f"  Expected path: {missing['expected_path']}\n")
                    f.write("\n")
            
            # Incomplete sessions
            if self.incomplete_sessions:
                f.write(f"\nINCOMPLETE DATA ({len(self.incomplete_sessions)} patients):\n")
                f.write("-"*80 + "\n\n")
                
                for patient in self.incomplete_sessions:
                    f.write(f"Patient: {patient['patient_id']} ({patient['original_id']})\n")
                    for session in patient['incomplete_sessions']:
                        f.write(f"  Session: {session['session_id']} ({session['date']})\n")
                        f.write(f"    Missing: {', '.join(session['missing'])}\n")
                        f.write(f"    Available: {', '.join(session['available'])}\n")
                    f.write("\n")
            else:
                f.write("\nCOMPLETENESS: All patients have all 4 modalities!\n")
        
        print(f"\n📄 Detailed report saved to: {output_file}")
    
    def print_summary(self, slice_stats: Dict, completeness_stats: Dict):
        """Print summary report."""
        print("\n" + "="*80)
        print("=== VALIDATION SUMMARY ===")
        print("="*80)
        
        print("\n📊 Slice Count Validation:")
        print(f"  Total series checked: {slice_stats['total_series']}")
        
        if slice_stats['mismatched_series'] == 0 and slice_stats['missing_directories'] == 0:
            print(f"  ✅ ALL SERIES VALID ({slice_stats['valid_series']} / {slice_stats['total_series']})")
        else:
            print(f"  ✓ Valid: {slice_stats['valid_series']} ({slice_stats['valid_series']/slice_stats['total_series']*100:.1f}%)")
            if slice_stats['mismatched_series'] > 0:
                print(f"  ✗ Mismatched: {slice_stats['mismatched_series']} ({slice_stats['mismatched_series']/slice_stats['total_series']*100:.1f}%)")
            if slice_stats['missing_directories'] > 0:
                print(f"  ✗ Missing: {slice_stats['missing_directories']} ({slice_stats['missing_directories']/slice_stats['total_series']*100:.1f}%)")
        
        print("\n🔍 Completeness Validation:")
        
        total_patients = completeness_stats['total_patients']
        complete_patients = completeness_stats['complete_patients']
        total_sessions = completeness_stats['total_sessions']
        complete_sessions = completeness_stats['complete_sessions']
        
        if completeness_stats['incomplete_patients'] == 0:
            print(f"  ✅ ALL PATIENTS COMPLETE ({complete_patients} / {total_patients})")
        else:
            print(f"  Patients:")
            print(f"    ✓ Complete: {complete_patients} / {total_patients} ({complete_patients/total_patients*100:.1f}%)")
            print(f"    ✗ Incomplete: {completeness_stats['incomplete_patients']} ({completeness_stats['incomplete_patients']/total_patients*100:.1f}%)")
        
        print(f"  Sessions:")
        print(f"    ✓ Complete: {complete_sessions} / {total_sessions} ({complete_sessions/total_sessions*100:.1f}%)")
        if completeness_stats['incomplete_sessions'] > 0:
            print(f"    ✗ Incomplete: {completeness_stats['incomplete_sessions']} ({completeness_stats['incomplete_sessions']/total_sessions*100:.1f}%)")
        
        # Overall verdict
        all_valid = (slice_stats['mismatched_series'] == 0 and 
                    slice_stats['missing_directories'] == 0 and
                    completeness_stats['incomplete_patients'] == 0)
        
        print("\n" + "="*80)
        if all_valid:
            print("✅ VERDICT: Dataset is FULLY VALID!")
        else:
            print("⚠️  VERDICT: Dataset has ISSUES - see detailed report above")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Validate BIDS dataset integrity'
    )
    parser.add_argument(
        'bids_dir',
        type=Path,
        help='BIDS output directory to validate'
    )
    parser.add_argument(
        '--mapping',
        type=Path,
        default=None,
        help='Path to dataset_mapping.json (default: BIDS_DIR/dataset_mapping.json)'
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=None,
        help='Save detailed report to file (default: BIDS_DIR/validation_report.txt)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.bids_dir.exists():
        print(f"❌ Error: BIDS directory does not exist: {args.bids_dir}")
        sys.exit(1)
    
    # Default paths
    if args.mapping is None:
        args.mapping = args.bids_dir / 'dataset_mapping.json'
    
    if args.report is None:
        args.report = args.bids_dir / 'validation_report.txt'
    
    print("="*80)
    print("BIDS DATASET VALIDATION")
    print("="*80)
    print(f"BIDS directory: {args.bids_dir}")
    print(f"Mapping file: {args.mapping}")
    print(f"Report file: {args.report}")
    
    # Run validation
    validator = BIDSValidator(args.bids_dir, args.mapping)
    
    if not validator.load_mapping():
        sys.exit(1)
    
    # Validate slice counts
    slice_stats = validator.validate_slice_counts()
    
    # Validate completeness
    completeness_stats = validator.validate_completeness()
    
    # Print detailed report
    validator.print_detailed_report()
    
    # Save to file
    validator.save_detailed_report(args.report)
    
    # Print summary
    validator.print_summary(slice_stats, completeness_stats)


if __name__ == '__main__':
    main()