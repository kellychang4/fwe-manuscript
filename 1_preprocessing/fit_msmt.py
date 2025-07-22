import argparse
import numpy as np
import os.path as op
import nibabel as nib

from dipy.data import get_sphere
from dipy.core.gradients import gradient_table, unique_bvals_magnitude
from dipy.reconst.mcsd import (
  MultiShellDeconvModel,
  mask_for_response_msmt,
  multi_shell_fiber_response,
  response_from_mask_msmt,
)

FODF_SPHERE = get_sphere("symmetric362")


def extract_lower_bval(dwi_data_file, bval_file, bvec_file, bval_thr = None):
  dwi_image = nib.load(dwi_data_file).get_fdata()
  gtab = gradient_table(bvals = bval_file, bvecs = bvec_file)
  unique_bvals = unique_bvals_magnitude(gtab.bvals)
  if bval_thr is None: # if not provided, use the smallest non-zero b-value
    bval_thr = np.min(unique_bvals[unique_bvals > 0])
  indx = gtab.bvals <= bval_thr # lower b-value shell data index
  dwi_lower  = dwi_image[..., indx]
  gtab_lower = gradient_table(bvals = gtab.bvals[indx], bvecs = gtab.bvecs[indx])
  return dwi_lower, gtab_lower


def estimate_msmt_response(dwi_data_file, bval_file, bvec_file,
                           sh_order_max = 8):
  # extract the lower b-value shell data for mask estimation
  dwi_lower, gtab_lower = extract_lower_bval(dwi_data_file, bval_file, bvec_file)

  # mask the data for response estimation
  mask_wm, mask_gm, mask_csf = mask_for_response_msmt(
    gtab       = gtab_lower, 
    data       = dwi_lower, 
    roi_radii  = 10, 
    wm_fa_thr  = 0.7, 
    gm_fa_thr  = 0.3,
    csf_fa_thr = 0.15, 
    gm_md_thr  = 0.001, 
    csf_md_thr = 0.0032,
  )

  # load the dwi data, mask, and gradient table
  dwi_values = nib.load(dwi_data_file).get_fdata()
  gtab = gradient_table(bvals = bval_file, bvecs = bvec_file)

  # estimate the response functions
  response_wm, response_gm, response_csf = response_from_mask_msmt(
    gtab, dwi_values, mask_wm, mask_gm, mask_csf
  )

  # combine response functions into a 3 compartment response function
  response_msmt = multi_shell_fiber_response(
    sh_order_max = sh_order_max,
    bvals  = unique_bvals_magnitude(gtab.bvals),
    wm_rf  = response_wm,
    gm_rf  = response_gm,
    csf_rf = response_csf,
  )    
  return response_msmt


def main(dwi_data_file, bval_file, bvec_file, mask_file, output_prefix,
         sh_order_max = 8):
  # load the dwi data, mask, and gradient table
  dwi_image  = nib.load(dwi_data_file)
  mask_image = nib.load(mask_file)
  gtab = gradient_table(bvals = bval_file, bvecs = bvec_file)
  
  # estimate three tissue response functions 
  response_msmt = estimate_msmt_response(
    dwi_data_file, bval_file, bvec_file, sh_order_max)
  
  # declare the multi-shell multi-tissue CSD model
  model = MultiShellDeconvModel(gtab, response_msmt)

  # fit the model 
  model_fit = model.fit(dwi_image.get_fdata(), mask_image.get_fdata())

  # extract model parameters
  model_params = {
    "shcoeff": model_fit.all_shm_coeff,
    "fodf": model_fit.odf(FODF_SPHERE),
    "CSF": { "shcoeff": model_fit.all_shm_coeff[..., 0] },
     "GM": { "shcoeff": model_fit.all_shm_coeff[..., 1] },
     "WM": { "shcoeff": model_fit.shm_coeff },
     "VF": model_fit.volume_fractions
  }

  # save the model parameters
  for key, value in model_params.items():
    if isinstance(value, dict): # if value is a dictionary
      [(k, value)] = value.items() # extract the inner value
      save_name = f"{output_prefix}_model-MSMT_label-{key}_param-{k}_dwimap.nii.gz"
    else: # else, value is an image
      save_name = f"{output_prefix}_model-MSMT_param-{key}_dwimap.nii.gz"
    image = nib.Nifti1Image(value, affine = dwi_image.affine)
    nib.save(image, save_name)
    print(f"Saved: {op.basename(save_name)}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dwi_data_file", type = str)
  parser.add_argument("--bval_file", type = str)
  parser.add_argument("--bvec_file", type = str)
  parser.add_argument("--mask_file", type = str)
  parser.add_argument("--output_prefix", type = str)
  parser.add_argument("--sh_order_max", type = int, default = 8)
  args = parser.parse_args()

  main(
    dwi_data_file = args.dwi_data_file,
    bval_file     = args.bval_file,
    bvec_file     = args.bvec_file, 
    mask_file     = args.mask_file,
    output_prefix = args.output_prefix,
    sh_order_max  = args.sh_order_max
  )