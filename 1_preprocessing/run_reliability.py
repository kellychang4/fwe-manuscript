import os 
import re
import argparse
import numpy as np
import os.path as op
import nibabel as nib

from dipy.core.gradients import (
  gradient_table, 
  unique_bvals_magnitude
)
from dipy.data import get_sphere
from dipy.reconst.csdeconv import (
  ConstrainedSphericalDeconvModel, 
  auto_response_ssst, 
)
FODF_SPHERE = get_sphere("symmetric362")


def split_diffusion_data(dwi_data_file, fwe_data_file, bval_file, bvec_file, 
                         output_dir):
  # load the diffusion and free water eliminated data, b-values, and b-vectors
  dwi_image = nib.load(dwi_data_file) 
  fwe_image = nib.load(fwe_data_file)
  gtab = gradient_table(bvals = bval_file, bvecs = bvec_file)

  # define the file base name template for output files
  bname = re.sub("_desc-preproc_dwi.nii.gz", "", op.basename(dwi_data_file))

  # define the b-value indices for each unique b-value
  unique_bvals = unique_bvals_magnitude(gtab.bvals)
  bval_indices = {x: np.where(gtab.bvals == x)[0] for x in unique_bvals}

  # split pseudorandomly split the b-value indices into two halves
  splits = [{}, {}]; # initialize 
  for key, value in bval_indices.items(): 
    shuffle_index = np.random.permutation(value) 
    n_half = shuffle_index.size // 2 # integer divide in half
    splits[0] = {**splits[0], **{key: shuffle_index[:n_half]}}
    splits[1] = {**splits[1], **{key: shuffle_index[n_half:]}}

  # split and save the split diffusion and free-water eliminated data
  for i, split in enumerate(splits): # for each split
    # collapse the split indices into a single array
    indices_split = np.concatenate([x for x in split.values()])

    # split the b-values and b-vectors
    bval_split = gtab.bvals[indices_split]
    bvec_split = gtab.bvecs[indices_split].T
    
    # split the diffusion data
    dwi_split = nib.Nifti1Image(
      dwi_image.get_fdata()[..., indices_split], 
      affine = dwi_image.affine
    )

    # split the free-water eliminated data
    fwe_split = nib.Nifti1Image(
      fwe_image.get_fdata()[..., indices_split], 
      affine = fwe_image.affine
    )
    
    # define (and create) the split directory
    paths_split = op.join(output_dir, f"split-{i + 1}", "data")
    os.makedirs(paths_split, exist_ok = True)

    # define the base name for the original diffusion split files
    fname_base = f"{bname}_desc-original_split-{i + 1}_dwi"

    # save the split b-values
    save_name = f"{fname_base}.bval"
    np.savetxt(op.join(paths_split, save_name), bval_split)
    print(f"Saved: {save_name}")

    # save the split b-vectors
    save_name = f"{fname_base}.bvec"
    np.savetxt(op.join(paths_split, save_name), bvec_split)
    print(f"Saved: {save_name}")

    # save the split diffusion data
    save_name = f"{fname_base}.nii.gz"
    nib.save(dwi_split, op.join(paths_split, save_name))
    print(f"Saved: {save_name}")

    # define and save the split free-water eliminated image
    save_name = f"{bname}_desc-fwe_split-{i + 1}_dwi.nii.gz"
    nib.save(fwe_split, op.join(paths_split, save_name))
    print(f"Saved: {save_name}")


def estimate_csd_model(dwi_data_file, fwe_data_file, bval_file, bvec_file, mask_file, output_dir):
  # load the diffusion and free water eliminated data, mask, b-values, and b-vectors
  dwi_image  = nib.load(dwi_data_file) 
  fwe_image  = nib.load(fwe_data_file)
  mask_image = nib.load(mask_file).get_fdata()
  gtab = gradient_table(bvals = bval_file, bvecs = bvec_file)

  # calculate the response function from b0 and lowest non-zero b-value diffusion data
  low_index = gtab.bvals <= np.min(gtab.bvals[gtab.bvals > 0])
  response_gtab = gradient_table(
    bvals = gtab.bvals[low_index], bvecs = gtab.bvecs[low_index])
  response_data = dwi_image.get_fdata()[..., low_index]
  response, _ = auto_response_ssst(
    response_gtab, response_data, roi_radii = 10, fa_thr = 0.7)

  # define the CSD model with the gradient table
  dwi_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max = 8)
  fwe_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max = 8)

  # fit the CSD model on the current split diffusion data
  dwi_fit = dwi_model.fit(dwi_image.get_fdata(), mask = mask_image)
  fwe_fit = fwe_model.fit(fwe_image.get_fdata(), mask = mask_image)

  # define base file name for output files
  dwi_bname = re.sub("_dwi.nii.gz", "", op.basename(dwi_data_file))
  fwe_bname = re.sub("_dwi.nii.gz", "", op.basename(fwe_data_file))

  # save the original CSD spherical harmonics
  save_name = f"{dwi_bname}_model-CSD_param-shcoeff_dwimap.nii.gz"
  dwi_coeff = nib.Nifti1Image(dwi_fit.shm_coeff, affine = dwi_image.affine)
  nib.save(dwi_coeff, op.join(output_dir, save_name))
  print(f"Saved: {save_name}")

  # save the original fCSD ODFs
  save_name = f"{dwi_bname}_model-CSD_param-fodf_dwimap.nii.gz"
  dwi_odf = nib.Nifti1Image(dwi_fit.odf(FODF_SPHERE), affine = dwi_image.affine)
  nib.save(dwi_odf, op.join(output_dir, save_name))
  print(f"Saved: {save_name}")

  # save the free water eliminated CSD spherical harmonics
  save_name = f"{fwe_bname}_model-CSD_param-shcoeff_dwimap.nii.gz"
  fwe_coeff = nib.Nifti1Image(fwe_fit.shm_coeff, affine = fwe_image.affine)
  nib.save(fwe_coeff, op.join(output_dir, save_name))
  print(f"Saved: {save_name}")

  # save the free water eliminated CSD fODFs
  save_name = f"{fwe_bname}_model-CSD_param-fodf_dwimap.nii.gz"
  fwe_odf = nib.Nifti1Image(fwe_fit.odf(FODF_SPHERE), affine = fwe_image.affine)
  nib.save(fwe_odf, op.join(output_dir, save_name))
  print(f"Saved: {save_name}")


def calculate_corr2_map(odf_file1, odf_file2, mask_file, output_file):
  # load the first and second ODFs files
  odf_image1 = nib.load(odf_file1).get_fdata()
  odf_image2 = nib.load(odf_file2).get_fdata()

  # load and get the shape of the mask image
  mask_image = nib.load(mask_file)
  mask_value = mask_image.get_fdata()
  anat_shape = mask_image.shape

  # calculate the correlation^2 map
  corr2_value = np.zeros(anat_shape) # initialize
  for index in np.ndindex(anat_shape): # for each voxel
    if mask_value[index]: # if the voxel is in the mask
      corr2_value[index] = np.corrcoef(odf_image1[index], odf_image2[index])[0,1]
  corr2_value = corr2_value ** 2 # pearson's r squared, normalized

  # save the correlation^2 map
  corr2_image = nib.Nifti1Image(corr2_value, affine = mask_image.affine)
  nib.save(corr2_image, output_file)
  print(f"Saved: {op.basename(output_file)}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparser = parser.add_subparsers(dest = "command")

  # split diffusion dataset argument parser
  split_dwi = subparser.add_parser("split_dwi")
  split_dwi.add_argument("--dwi_data_file", type = str)
  split_dwi.add_argument("--fwe_data_file", type = str)
  split_dwi.add_argument("--bval_file", type = str)
  split_dwi.add_argument("--bvec_file", type = str)
  split_dwi.add_argument("--output_dir", type = str)

  # estimate and save csd model argument parser
  estimate_csd = subparser.add_parser("estimate_csd")
  estimate_csd.add_argument("--dwi_data_file", type = str)
  estimate_csd.add_argument("--fwe_data_file", type = str)
  estimate_csd.add_argument("--bval_file", type = str)
  estimate_csd.add_argument("--bvec_file", type = str)
  estimate_csd.add_argument("--mask_file", type = str)
  estimate_csd.add_argument("--output_dir", type = str)

  # estimate and save corr2 map argument parser
  calc_corr2 = subparser.add_parser("calc_corr2")
  calc_corr2.add_argument("--odf_file1", type = str)
  calc_corr2.add_argument("--odf_file2", type = str)
  calc_corr2.add_argument("--mask_file", type = str)
  calc_corr2.add_argument("--output_file", type = str)

  args = parser.parse_args()

  # call the appropriate function based on the command
  match args.command:
    case "split_dwi": 
      split_diffusion_data(
        dwi_data_file = args.dwi_data_file,
        fwe_data_file = args.fwe_data_file, 
        bval_file     = args.bval_file, 
        bvec_file     = args.bvec_file,
        output_dir    = args.output_dir
      )
    case "estimate_csd": 
      estimate_csd_model(
        dwi_data_file = args.dwi_data_file, 
        fwe_data_file = args.fwe_data_file, 
        bval_file     = args.bval_file, 
        bvec_file     = args.bvec_file,
        mask_file     = args.mask_file,
        output_dir    = args.output_dir
      )
    case "calc_corr2": 
      calculate_corr2_map(
        odf_file1   = args.odf_file1, 
        odf_file2   = args.odf_file2,
        mask_file   = args.mask_file,
        output_file = args.output_file
      )
