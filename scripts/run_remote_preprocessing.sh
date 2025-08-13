#!/bin/bash

# Remote paths
REMOTE_HOST="roppert@bigdata.nsu.ru"
REMOTE_INPUT="/media/storage/roppert/bids_nifti"
REMOTE_OUTPUT="/media/storage/roppert/MS_preprocessed/preprocessed"
REMOTE_TRANSFORMS="/media/storage/roppert/MS_preprocessed/transforms"

# Local mount points
LOCAL_BASE="$HOME/mnt/bigdata"
LOCAL_INPUT="$LOCAL_BASE/input"
LOCAL_OUTPUT="$LOCAL_BASE/output"
LOCAL_TRANSFORMS="$LOCAL_BASE/transforms"

# Create mount points
mkdir -p "$LOCAL_BASE"/{input,output,transforms}

# Mount remote directories
echo "Mounting remote directories..."
sshfs -o compression=yes,kernel_cache,reconnect "$REMOTE_HOST:$REMOTE_INPUT" "$LOCAL_INPUT"
sshfs -o compression=yes,kernel_cache,reconnect "$REMOTE_HOST:$REMOTE_OUTPUT" "$LOCAL_OUTPUT"
sshfs -o compression=yes,kernel_cache,reconnect "$REMOTE_HOST:$REMOTE_TRANSFORMS" "$LOCAL_TRANSFORMS"

# Run the preprocessing script
echo "Running preprocessing..."
python preprocessing.py \
    "$LOCAL_INPUT" \
    "$LOCAL_OUTPUT" \
    "$LOCAL_TRANSFORMS" \
    --template_path "/home/roppert/work/mri_ai_service/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii" \
    --config "/home/roppert/work/mri_ai_service/config/config.yaml" \
    "$@"  # Pass any additional arguments

# Unmount when done
echo "Unmounting remote directories..."
fusermount -u "$LOCAL_INPUT" 2>/dev/null || umount "$LOCAL_INPUT"
fusermount -u "$LOCAL_OUTPUT" 2>/dev/null || umount "$LOCAL_OUTPUT"
fusermount -u "$LOCAL_TRANSFORMS" 2>/dev/null || umount "$LOCAL_TRANSFORMS"

echo "Done!"