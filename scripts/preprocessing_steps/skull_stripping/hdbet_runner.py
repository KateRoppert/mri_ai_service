"""
HD-BET on one file without the CLI's worker fan-out.

Run by HdBetStripper with the Python interpreter that has HD-BET installed
(FSL's Python in the image), never imported by the pipeline itself.

The `hd-bet` CLI (HD-BET 2.0.1) hardcodes 4 preprocessing and 8 export
workers plus a Pool(4) for applying the mask, each importing torch: ~13
processes and 5.1-5.2 GB for a single volume. The same predictor with one
worker of each kind takes ~2 GB, in the same time, with a bit-identical mask
and stripped image (measured 2026-10-12, SibBMS ses-005, atlas-space and
native 1 mm t1). Outputs match the CLI with --save_bet_mask: the stripped
image at -o and the mask at `<o without .nii.gz>_bet.nii.gz`.
"""
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-device", default="cuda")
    parser.add_argument("--disable_tta", action="store_true")
    args = parser.parse_args(argv)
    if not args.o.endswith(".nii.gz"):
        parser.error("-o must end with .nii.gz")

    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import apply_bet, get_hdbet_predictor

    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=not args.disable_tta,
                                    device=torch.device(args.device))
    mask = args.o[:-len(".nii.gz")] + "_bet.nii.gz"
    predictor.predict_from_files(
        [[args.i]], [mask],
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=1, num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0,
    )
    # nnU-Net leaves these next to its output; the CLI removes them too.
    for name in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
        try:
            os.remove(os.path.join(os.path.dirname(os.path.abspath(mask)), name))
        except OSError:
            pass
    apply_bet(args.i, mask, args.o)
    return 0


# The guard is required: nnU-Net's workers are spawned and re-import this file.
if __name__ == "__main__":
    sys.exit(main())
