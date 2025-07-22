#!/usr/bin/env bash


participant="${1}"
if [[ ! -z "${participant}" ]]; then
  echo "[ERROR] Participant ID required."
  exit 1
fi

# Directories ------------------------------------------------------------------
 
DATA_HOME="/paths/to/dataset"
DATA_DIR="${DATA_HOME}/qsiprep/${participant}/anat"
OUT_DIR="/path/to/output"
if [[ ! -d "${OUT_DIR}" ]]; then mkdir -p "${OUT_DIR}"; fi
FILES_DIR="/paths/to/files"

# Create Glass Brain -----------------------------------------------------------

fname="${participant}_space-ACPC_desc-glass_mask.nii.gz"
if [[ ! -f "${OUT_DIR}/${fname}" ]]; then
  apptainer run \
    --bind "${DATA_DIR}:/data" \
    --bind "${OUT_DIR}:/out" \
    --bind "${FILES_DIR}:/files" \
    "${APPTAINER_DIR}/antspyx.sif" \
    /files/create_glass_brain.py \
    --aseg_image "/data/${participant}_space-ACPC_desc-aseg_dseg.nii.gz" \
    --output_fname "/out/${fname}" 
fi