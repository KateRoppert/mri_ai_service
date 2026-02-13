"""
Test script for resampling preprocessing step.
Tests resampling functionality without full pipeline.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import ants
import numpy as np

# Add preprocessing_steps to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.preprocessing_steps.resampling import process_subject_resampling, resample_image


def create_test_image(shape=(64, 64, 48), spacing=(2.0, 2.0, 3.0)):
    """Create a test NIfTI image with known properties."""
    data = np.random.rand(*shape).astype(np.float32)
    img = ants.from_numpy(data)
    img.set_spacing(spacing)
    return img


def test_resample_image():
    """Test basic resampling function.
    
    Note: 2% spacing tolerance is appropriate because:
    - Discrete voxel grids cannot perfectly match continuous target spacing
    - This is standard behavior in medical imaging resampling
    - Error < 0.02mm for 1mm spacing is clinically negligible
    """
    print("\n=== Test 1: Basic resampling ===")
    
    # Create test image
    img = create_test_image(shape=(64, 64, 48), spacing=(2.0, 2.0, 3.0))
    print(f"Input: shape={img.shape}, spacing={img.spacing}")
    
    # Resample to 1mm isotropic
    target_spacing = [1.0, 1.0, 1.0]
    resampled = resample_image(img, target_spacing, interpolation='linear')
    
    print(f"Output: shape={resampled.shape}, spacing={resampled.spacing}")
    
    # Verify spacing is approximately correct (allow 2% error due to discretization)
    spacing_error = [abs(s - t) / t for s, t in zip(resampled.spacing, target_spacing)]
    max_error = max(spacing_error)
    print(f"Spacing error: max={max_error*100:.2f}% (threshold=2.0%)")
    assert all(err < 0.02 for err in spacing_error), f"Spacing error too large: {spacing_error}"
    
    # Verify shape changed correctly
    expected_shape = (128, 128, 144)  # Approximately 2x, 2x, 3x larger
    shape_diff = [abs(s - e) / e for s, e in zip(resampled.shape, expected_shape)]
    assert all(diff < 0.05 for diff in shape_diff), f"Shape not as expected: {resampled.shape} vs {expected_shape}"
    
    print("✓ Basic resampling works correctly")
    return True


def test_process_subject():
    """Test full subject processing."""
    print("\n=== Test 2: Subject processing ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test BIDS structure
        subject_id = "sub-001"
        session_id = "ses-001"
        anat_dir = tmpdir / "input" / subject_id / session_id / "anat"
        anat_dir.mkdir(parents=True)
        
        # Create test files for multiple modalities
        modalities = ["t1", "t2", "t2fl"]
        for modality in modalities:
            filename = f"{subject_id}_{session_id}_{modality}.nii.gz"
            img = create_test_image(shape=(60, 60, 40), spacing=(2.0, 2.0, 2.5))
            ants.image_write(img, str(anat_dir / filename))
            print(f"Created: {filename}")
        
        # Output directory
        output_dir = tmpdir / "output"
        
        # Process
        params = {
            'output_resolution': [1.0, 1.0, 1.0],
            'interpolation': 'linear'
        }
        
        results = process_subject_resampling(
            subject_dir=anat_dir,
            output_dir=output_dir,
            modalities=modalities,
            params=params
        )
        
        # Verify results
        print(f"\nResults: {len(results)} modalities")
        for modality, result in results.items():
            success = result.get('success')
            if success:
                spacing = result['output_spacing']
                print(f"  {modality}: success={success}, spacing={spacing}")
            else:
                print(f"  {modality}: success={success}")
            assert result['success'], f"{modality} failed"
            
            # Verify output file exists
            output_file = Path(result['output_file'])
            assert output_file.exists(), f"Output file not created: {output_file}"
            
            # Verify spacing (allow 2% error)
            output_spacing = result['output_spacing']
            spacing_error = [abs(s - 1.0) / 1.0 for s in output_spacing]
            max_error = max(spacing_error)
            assert all(err < 0.02 for err in spacing_error), f"Wrong spacing: {output_spacing}, max error: {max_error*100:.2f}%"
        
        print("✓ Subject processing works correctly")
        return True


def test_bids_structure():
    """Test that BIDS structure is preserved."""
    print("\n=== Test 3: BIDS structure preservation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create input
        subject_id = "sub-TEST01"
        session_id = "ses-20250121"
        anat_dir = tmpdir / "input" / subject_id / session_id / "anat"
        anat_dir.mkdir(parents=True)
        
        modality = "t1"
        filename = f"{subject_id}_{session_id}_{modality}.nii.gz"
        img = create_test_image()
        ants.image_write(img, str(anat_dir / filename))
        
        # Process
        output_dir = tmpdir / "output"
        params = {'output_resolution': [1.0, 1.0, 1.0]}
        
        results = process_subject_resampling(
            subject_dir=anat_dir,
            output_dir=output_dir,
            modalities=[modality],
            params=params
        )
        
        # Verify output structure
        expected_path = output_dir / subject_id / session_id / "anat" / filename
        assert expected_path.exists(), f"BIDS structure not preserved: {expected_path}"
        print(f"✓ Output structure: {expected_path.relative_to(output_dir)}")
        
        print("✓ BIDS structure preserved correctly")
        return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("RESAMPLING PREPROCESSING STEP - FUNCTIONALITY TEST")
    print("=" * 70)
    
    tests = [
        test_resample_image,
        test_process_subject,
        test_bids_structure
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())