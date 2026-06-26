#!/usr/bin/env bash
# One-time: extract FSL standard-space atlases needed for MS McDonald zone
# classification from the project's Docker image (which bundles FSL).
# Run once; output is committed to data/templates/ms_zones_raw/.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/templates/ms_zones_raw"
mkdir -p "$OUT_DIR"

IMAGE="kateroppert/mri-ai-service:latest"
FSL_ATLASES="/usr/local/fsl/data/atlases"

echo "Extracting atlases from $IMAGE into $OUT_DIR ..."

# NOTE: the image's default ENTRYPOINT always launches the Flask webapp
# (see `docker inspect kateroppert/mri-ai-service:latest`), ignoring any
# command passed after the image name. --entrypoint bash overrides that
# so we can run our extraction command instead.
docker run --rm --entrypoint bash -v "$OUT_DIR:/out" "$IMAGE" -c "
  cp '$FSL_ATLASES/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz' /out/ &&
  cp '$FSL_ATLASES/HarvardOxford-Subcortical.xml' /out/ &&
  cp '$FSL_ATLASES/Cerebellum/Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz' /out/ &&
  cp '$FSL_ATLASES/Cerebellum_MNIflirt.xml' /out/
"

echo "Done. Files in $OUT_DIR:"
ls -la "$OUT_DIR"
