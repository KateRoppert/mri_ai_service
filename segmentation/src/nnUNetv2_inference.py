import glob
import subprocess

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder
import torch

def predict_for_api(input_folder, output_folder, tta, folds, task_name):
    PLANS = 'nnUNetPlans'
    CONFIG = '3d_fullres'
    CHECKPOINT = 'checkpoint_final.pth'

    if task_name == 'ISBI':
        dataset_id = 1
        trainer = 'nnUNetTrainer_DiceCE'
    elif task_name == 'Shifts':
        dataset_id = 2
        trainer = 'nnUNetTrainer'
    elif task_name == 'META':
        dataset_id = 3
        trainer = 'nnUNetTrainer_DiceCE'

    model_folder = get_output_folder(dataset_id, trainer, PLANS, CONFIG)

    if not isdir(output_folder):
        maybe_mkdir_p(output_folder)

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=not tta,
                                perform_everything_on_gpu=True,
                                device=device,
                                verbose=False,
                                verbose_preprocessing=False)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        folds,
        checkpoint_name=CHECKPOINT
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