import argparse
import os.path as op
import nibabel as nib

import dipy.reconst.dki_micro as dki_micro
from dipy.core.gradients import gradient_table

# declare the model relevant kwargs
MK_KWARGS = {
  "min_kurtosis": -1, 
  "max_kurtosis": 3
}


def main(dwi_data_file, bval_file, bvec_file, mask_file, output_prefix):
  # load the diffusion, bval, bvecs, and mask file
  dwi_image  = nib.load(dwi_data_file)
  mask_image = nib.load(mask_file)
  gtab = gradient_table(bval_file, bvec_file)

  # declare the model
  model = dki_micro.KurtosisMicrostructureModel(gtab)
  
  # fit the model
  model_fit = model.fit(
    data = dwi_image.get_fdata(),
    mask = mask_image.get_fdata()
  )

  # extract model parameters
  model_params = {
    "modelparams": model_fit.model_params,
    "AWF": model_fit.awf,
    "FA": model_fit.fa,
    "MD": model_fit.md,
    "MK": model_fit.mk(**MK_KWARGS)
  }

  # save the model parameters
  for key, value in model_params.items():
    image = nib.Nifti1Image(value, affine = dwi_image.affine)
    save_name = f"{output_prefix}_model-DKI_param-{key}_dwimap.nii.gz"
    nib.save(image, save_name)
    print(f"Saved: {op.basename(save_name)}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dwi_data_file", type = str)
  parser.add_argument("--bval_file", type = str)
  parser.add_argument("--bvec_file", type = str)
  parser.add_argument("--mask_file", type = str)
  parser.add_argument("--output_prefix", type = str)
  args = parser.parse_args()
  
  main(
    dwi_data_file = args.dwi_data_file,
    bval_file     = args.bval_file, 
    bvec_file     = args.bvec_file, 
    mask_file     = args.mask_file, 
    output_prefix = args.output_prefix
  )