#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dry-run script for testing BIDS scanner without running actual segmentation.

This script scans the BIDS directory and reports what would be processed
without actually sending any data to the segmentation server.
"""
import argparse
import sys
from pathlib import Path
import yaml

# Import necessary classes from the main script
# In real use, these would be imported from the module
# For now, we'll assume they're in the same directory

def load_config(config_path: Path) -> dict:
    """Loads the YAML config file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def scan_bids_directory(input_dir: Path, modality_map: dict, max_subjects: int = None):
    """
    Simulates the BIDSScanner functionality for testing.
    """
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    subject_dirs = sorted([d for d in input_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('sub-')])
    
    print(f"\n{'='*70}")
    print(f"BIDS DIRECTORY SCAN - DRY RUN")
    print(f"{'='*70}")
    print(f"Input directory: {input_dir}")
    print(f"Found {len(subject_dirs)} subject directories")
    print(f"{'='*70}\n")
    
    total_sessions = 0
    valid_sessions = 0
    incomplete_sessions = 0
    
    for subject_dir in subject_dirs:
        if max_subjects and total_sessions >= max_subjects:
            print(f"\n⚠️  Reached max_subjects limit ({max_subjects}). Stopping scan.")
            break
        
        subject_id = subject_dir.name
        print(f"\n📁 {subject_id}")
        
        # Check for sessions
        session_dirs = sorted([d for d in subject_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('ses-')])
        
        if session_dirs:
            for session_dir in session_dirs:
                session_id = session_dir.name
                anat_dir = session_dir / "anat"
                total_sessions += 1
                
                if not anat_dir.exists():
                    print(f"  ⚠️  {session_id}: No 'anat' directory")
                    incomplete_sessions += 1
                    continue
                
                # Check for modality files
                identifier = f"{subject_id}_{session_id}"
                modality_files = {}
                missing = []
                
                for server_key, bids_suffix in modality_map.items():
                    pattern = f"{subject_id}_{session_id}_{bids_suffix}.nii.gz"
                    matches = list(anat_dir.glob(pattern))
                    
                    if matches:
                        modality_files[server_key] = matches[0]
                    else:
                        missing.append(server_key)
                
                if not missing:
                    print(f"  ✓ {session_id}: All modalities present")
                    for mod, path in modality_files.items():
                        print(f"      - {mod}: {path.name}")
                    valid_sessions += 1
                else:
                    print(f"  ⚠️  {session_id}: Missing modalities {missing}")
                    incomplete_sessions += 1
        else:
            # No sessions - check for direct anat directory
            anat_dir = subject_dir / "anat"
            total_sessions += 1
            
            if not anat_dir.exists():
                print(f"  ⚠️  No 'anat' directory")
                incomplete_sessions += 1
                continue
            
            # Check for modality files
            modality_files = {}
            missing = []
            
            for server_key, bids_suffix in modality_map.items():
                pattern = f"{subject_id}_{bids_suffix}.nii.gz"
                matches = list(anat_dir.glob(pattern))
                
                if matches:
                    modality_files[server_key] = matches[0]
                else:
                    missing.append(server_key)
            
            if not missing:
                print(f"  ✓ All modalities present")
                for mod, path in modality_files.items():
                    print(f"      - {mod}: {path.name}")
                valid_sessions += 1
            else:
                print(f"  ⚠️  Missing modalities {missing}")
                incomplete_sessions += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SCAN SUMMARY")
    print(f"{'='*70}")
    print(f"Total sessions found:        {total_sessions}")
    print(f"Valid (complete) sessions:   {valid_sessions}")
    print(f"Incomplete sessions:         {incomplete_sessions}")
    print(f"{'='*70}\n")
    
    if valid_sessions == 0:
        print("⚠️  WARNING: No valid sessions found for processing!")
        return False
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dry-run BIDS scanner to verify directory structure before segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with preprocessed NIfTI files in BIDS format"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "segmentation_config.yaml",
        help="Path to segmentation configuration file"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to scan (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Extract modality map
    try:
        modality_map = config['segmentation']['modality_input_map']
    except KeyError:
        print("Error: 'segmentation.modality_input_map' not found in config")
        sys.exit(1)
    
    # Perform scan
    success = scan_bids_directory(args.input_dir, modality_map, args.max_subjects)
    
    sys.exit(0 if success else 1)