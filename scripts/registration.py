"""
ANTs Registration Script for BIDS-structured NIfTI files.

This script performs two-stage registration:
1. Register T1 images to a template
2. Register other modalities to the registered T1

"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import ants

# Setup logging
logger = logging.getLogger(__name__)


def setup_logging(log_file: Path, console_level: str = "INFO"):
    """Configure logging for the script."""
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


class BIDSFileManager:
    """Manages BIDS file structure and discovery."""
    
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.subjects = self._discover_subjects()
    
    def _discover_subjects(self) -> Dict[str, Dict[str, List[Path]]]:
        """Discover all subjects, sessions, and their files."""
        subjects = {}
        
        # Find all subject directories
        for sub_dir in self.input_dir.glob("sub-*"):
            if not sub_dir.is_dir():
                continue
                
            subject_id = sub_dir.name
            subjects[subject_id] = {}
            
            # Check for sessions
            ses_dirs = list(sub_dir.glob("ses-*"))
            
            if ses_dirs:
                # Multi-session structure
                for ses_dir in ses_dirs:
                    session_id = ses_dir.name
                    anat_dir = ses_dir / "anat"
                    if anat_dir.exists():
                        files = list(anat_dir.glob("*.nii*"))
                        if files:
                            subjects[subject_id][session_id] = files
            else:
                # Single session structure
                anat_dir = sub_dir / "anat"
                if anat_dir.exists():
                    files = list(anat_dir.glob("*.nii*"))
                    if files:
                        subjects[subject_id][""] = files  # Empty string for no session
        
        return subjects
    
    def get_t1_file(self, files: List[Path]) -> Optional[Path]:
        """Find T1w file from a list of files."""
        for file in files:
            filename = file.name.lower()
            # T1w but not contrast enhanced
            if "t1w" in filename and "ce" not in filename and "gad" not in filename:
                return file
        return None
    
    def get_modality(self, file: Path) -> str:
        """Detect modality from filename."""
        filename = file.name.lower()
        
        if "flair" in filename:
            return "FLAIR"
        elif "t2w" in filename:
            return "T2"
        elif "t1w" in filename:
            if "ce" in filename or "gad" in filename:
                return "T1CE"
            else:
                return "T1"
        else:
            return "UNKNOWN"


class RegistrationEngine:
    """Handles ANTs registration operations."""
    
    def __init__(self, template_path: Path, transform_type: str = "SyN"):
        self.template_path = template_path
        self.transform_type = transform_type
        self.template = ants.image_read(str(template_path))
    
    def register_to_template(self, moving_path: Path, output_prefix: Path) -> Dict:
        """Register an image to the template."""
        logger.info(f"Registering {moving_path.name} to template")
        
        moving = ants.image_read(str(moving_path))
        
        # Perform registration
        registration = ants.registration(
            fixed=self.template,
            moving=moving,
            type_of_transform=self.transform_type,
            outprefix=str(output_prefix),
            verbose=False
        )
        
        return {
            "warped_image": registration["warpedmovout"],
            "forward_transforms": registration["fwdtransforms"],
            "inverse_transforms": registration["invtransforms"]
        }
    
    def register_to_target(self, fixed_path: Path, moving_path: Path, 
                          output_prefix: Path) -> Dict:
        """Register an image to another image (e.g., other modalities to T1)."""
        logger.info(f"Registering {moving_path.name} to {fixed_path.name}")
        
        fixed = ants.image_read(str(fixed_path))
        moving = ants.image_read(str(moving_path))
        
        # For intra-subject registration, use rigid or affine
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",  # or "Affine" for more flexibility
            outprefix=str(output_prefix),
            verbose=False
        )
        
        return {
            "warped_image": registration["warpedmovout"],
            "forward_transforms": registration["fwdtransforms"],
            "inverse_transforms": registration["invtransforms"]
        }


class RegistrationPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, input_dir: Path, output_dir: Path, transforms_dir: Path,
                 template_path: Path, transform_type: str = "SyN"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transforms_dir = transforms_dir
        self.file_manager = BIDSFileManager(input_dir)
        self.engine = RegistrationEngine(template_path, transform_type)
        
    def process_session(self, subject_id: str, session_id: str, 
                       files: List[Path]) -> Dict:
        """Process all files in a session."""
        results = {
            "subject": subject_id,
            "session": session_id,
            "status": "success",
            "files_processed": []
        }
        
        # Create output directories
        if session_id:
            output_path = self.output_dir / subject_id / session_id / "anat"
            transform_path = self.transforms_dir / subject_id / session_id / "anat"
        else:
            output_path = self.output_dir / subject_id / "anat"
            transform_path = self.transforms_dir / subject_id / "anat"
        
        output_path.mkdir(parents=True, exist_ok=True)
        transform_path.mkdir(parents=True, exist_ok=True)
        
        # Find T1 file
        t1_file = self.file_manager.get_t1_file(files)
        
        if not t1_file:
            logger.warning(f"No T1w file found for {subject_id}/{session_id}")
            results["status"] = "no_t1"
            return results
        
        # Stage 1: Register T1 to template
        t1_stem = t1_file.stem.replace('.nii', '')
        t1_output_prefix = transform_path / f"{t1_stem}_to_template_"
        
        try:
            t1_reg = self.engine.register_to_template(t1_file, t1_output_prefix)
            
            # Save registered T1
            t1_output_file = output_path / t1_file.name
            ants.image_write(t1_reg["warped_image"], str(t1_output_file))
            
            results["files_processed"].append({
                "input": str(t1_file),
                "output": str(t1_output_file),
                "modality": "T1",
                "registration": "to_template",
                "transforms": t1_reg["forward_transforms"]
            })
            
            # Stage 2: Register other modalities to registered T1
            for file in files:
                if file == t1_file:
                    continue
                
                modality = self.file_manager.get_modality(file)
                file_stem = file.stem.replace('.nii', '')
                output_prefix = transform_path / f"{file_stem}_to_T1_"
                
                try:
                    reg = self.engine.register_to_target(
                        t1_output_file, file, output_prefix
                    )
                    
                    # Save registered file
                    output_file = output_path / file.name
                    ants.image_write(reg["warped_image"], str(output_file))
                    
                    results["files_processed"].append({
                        "input": str(file),
                        "output": str(output_file),
                        "modality": modality,
                        "registration": "to_T1",
                        "transforms": reg["forward_transforms"]
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to register {file.name}: {e}")
                    results["status"] = "partial_failure"
                    
        except Exception as e:
            logger.error(f"Failed to register T1 to template: {e}")
            results["status"] = "t1_registration_failed"
            
        return results
    
    def run(self) -> List[Dict]:
        """Run the registration pipeline."""
        logger.info(f"Starting registration pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Transforms directory: {self.transforms_dir}")
        
        all_results = []
        
        for subject_id, sessions in self.file_manager.subjects.items():
            for session_id, files in sessions.items():
                logger.info(f"Processing {subject_id}/{session_id if session_id else 'no-session'}")
                
                results = self.process_session(subject_id, session_id, files)
                all_results.append(results)
        
        # Summary
        total_subjects = len(self.file_manager.subjects)
        successful = sum(1 for r in all_results if r["status"] == "success")
        
        logger.info(f"Registration complete: {successful}/{total_subjects} subjects processed successfully")
        
        return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ANTs registration for BIDS-structured NIfTI files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_dir", type=Path, help="Input BIDS directory")
    parser.add_argument("output_dir", type=Path, help="Output BIDS directory")
    parser.add_argument("transforms_dir", type=Path, help="Directory for transform files")
    parser.add_argument("--template_path", required=True, type=Path, help="Path to template file")
    parser.add_argument("--transform_type", default="SyN", help="ANTs transform type")
    parser.add_argument("--log_file", type=Path, help="Path to log file")
    parser.add_argument("--summary_file", type=Path, help="Path to save summary JSON")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or args.transforms_dir / "registration.log"
    setup_logging(log_file)
    
    try:
        # Create pipeline and run
        pipeline = RegistrationPipeline(
            args.input_dir,
            args.output_dir,
            args.transforms_dir,
            args.template_path,
            args.transform_type
        )
        
        results = pipeline.run()
        
        # Save summary if requested
        if args.summary_file:
            args.summary_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Exit with appropriate code
        if all(r["status"] == "success" for r in results):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()