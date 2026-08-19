import glob
import subprocess

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch


def predict_for_api(input_folder, output_folder, tta, folds, model_folder,
                     checkpoint='checkpoint_final.pth'):
    """
    Run nnUNet v2 inference against an already-resolved model folder.

    model_folder must be the full path to the trainer/plans/configuration
    directory, e.g.
    ".../Dataset111_Meta/nnUNetTrainerSegResNet__nnUNetPlans__3d_fullres"
    (built by service_server.py from the active server_config.yaml preset).
    """
    if not isdir(output_folder):
        maybe_mkdir_p(output_folder)

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=not tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=False,
                                verbose_preprocessing=False)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        folds,
        checkpoint_name=checkpoint
    )
    predictor.predict_from_files(input_folder, output_folder, save_probabilities=False,
                                overwrite=True,
                                num_processes_preprocessing=3,
                                num_processes_segmentation_export=3,
                                folder_with_segs_from_prev_stage=None,
                                num_parts=1,
                                part_id=0)

    for nii in glob.glob(f"{output_folder}/*.nii"):
        subprocess.run(['gzip', '-fq', nii], check=True)

    return ""
