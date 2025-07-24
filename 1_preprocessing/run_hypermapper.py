import ants
import skimage
import argparse
import numpy as np
import os.path as op
import nibabel as nib
import skimage.morphology as morph


def _preprocess_volume(image_file, mask_file):
  image  = ants.image_read(image_file)
  mask   = ants.image_read(mask_file)
  image = ants.n4_bias_field_correction(image, mask)
  return ants.mask_image(image, mask)


def rigid_coregistration(moving_image, moving_mask, target_image, target_mask, 
                         output_file):
  # load and preprocess moving image and mask
  moving_masked = _preprocess_volume(moving_image, moving_mask)

  # load and preprocess target image and mask
  target_masked = _preprocess_volume(target_image, target_mask)

  # coregister the moving to the target image (rigid)
  mtx = ants.registration(
    fixed  = target_masked, 
    moving = moving_masked, 
    type_of_transform = "Rigid"
  )

  # apply transformation matrix to moving image
  coreg_image = ants.apply_transforms(
    fixed  = target_masked, 
    moving = moving_masked, 
    transformlist = mtx["fwdtransforms"]
  )
  
  # write coregistered moving image
  ants.image_write(coreg_image, output_file)
  print(f"Saved: {output_file}")


def transform_image(image_file, transform_file, output_file, 
                    interpolation = "linear"):
  # read image and transformation file
  image     = ants.image_read(image_file)
  transform = ants.read_transform(transform_file)

  # apply transformation and save transformed image
  image = transform.apply_to_image(image, interpolation = interpolation)
  ants.image_write(image, output_file)
  print(f"Saved: {output_file}")


def probseg2dseg(probseg_file, output_file, probseg_threshold):     
  # load white matter hyperintensity segmentation probabilities
  probseg = ants.image_read(probseg_file)

  # threshold probabilities and convert to labels
  probseg_thr    = probseg.numpy() > probseg_threshold
  probseg_labels = skimage.measure.label(probseg_thr)
  probseg_labels = probseg_labels.astype(np.float32)

  # convert discrete labels to ants
  dseg = ants.from_numpy(
    data      = probseg_labels,
    origin    = probseg.origin,
    spacing   = probseg.spacing, 
    direction = probseg.direction
  )

  # write discrete segmentation file
  ants.image_write(dseg, output_file)
  print(f"Saved: {output_file}")


def _create_mask(aseg_image, labels):
  # extract label values from image
  image = aseg_image.get_fdata().copy() 

  # create mask as combination of labels
  mask = np.zeros_like(image)
  for label in labels: # for each value
    mask = np.logical_or(mask, image == label)

  # return mask (as float)
  return mask * 1.0


def clean_white_matter(aseg_file, output_file, radius = 1):
  # load segmentation file
  aseg = nib.load(aseg_file)
  
  # create white matter mask
  wm        = _create_mask(aseg, [2, 41])
  ventricle = _create_mask(aseg, [4, 43])

  # dilate ventricle mask
  ventricle_dilated = morph.isotropic_dilation(ventricle, radius = radius)

  # erode ventricle mask
  ventricle_eroded = morph.isotropic_erosion(ventricle, radius = radius)  

  # remove ventricle mask from white matter
  wm_cleaned = wm.copy(); wm_cleaned[ventricle_dilated] = 0
 
  # save cleaned white matter as nifti file
  wm_cleaned = nib.Nifti1Image(wm_cleaned, affine = aseg.affine) 
  nib.save(wm_cleaned, output_file)
  print(f"Saved: {output_file}")

  # save eroded ventricle mask as nifti file 
  output_file = op.join(op.dirname(output_file), "ventricle_clean.nii.gz")
  ventricle_eroded = nib.Nifti1Image(
    ventricle_eroded * 1.0, affine = aseg.affine)
  nib.save(ventricle_eroded, output_file)
  print(f"Saved: {output_file}")


def clean_white_matter_hyperintensity(
    wmh_dseg_file, wm_file, ventricle_file, output_file):
  # load white matter hyperintensity segmentation file
  wmh_dseg = nib.load(wmh_dseg_file)
  
  # load white matter mask file
  wm = nib.load(wm_file).get_fdata() 

  # load ventricle mask file
  ventricle = nib.load(ventricle_file).get_fdata()
  
  # restrict wmh to white matter mask
  indx = np.logical_not(np.logical_and(wmh_dseg.get_fdata() > 0, wm == 1))
  wmh_cleaned = wmh_dseg.get_fdata().copy(); wmh_cleaned[indx] = 0

  # identify normal-appearing white matter, assign -1 
  indx = np.logical_and(wmh_cleaned == 0, wm == 1)
  wmh_cleaned[indx] = -1 # normal-appear white matter

  # identify ventricles, assign -2
  wmh_cleaned[ventricle == 1] = -2 # ventricles

  # save cleaned wmh as nifti file
  wmh_cleaned = nib.Nifti1Image(wmh_cleaned, affine = wmh_dseg.affine)
  nib.save(wmh_cleaned, output_file)
  print(f"Saved: {output_file}")


def normalize_within_mask(input_file, mask_file, output_file):     
  # read input and mask image files
  image = ants.image_read(input_file)
  mask  = ants.image_read(mask_file) 

  # extract image values
  image_values = image.numpy()
  
  # calculate image mean and sd within mask values
  values = image_values[mask.numpy() == 1] 
  m = np.nanmean(values); s = np.nanstd(values)
  
  # compute normalized flair and convert to ants
  image_norm = (image_values - m) / s # normalize image
  image_norm = ants.from_numpy(
    data      = image_norm, 
    origin    = image.origin, 
    spacing   = image.spacing, 
    direction = image.direction
  )

  # write normalized image file
  ants.image_write(image_norm, output_file)
  print(f"Saved: {output_file}")


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  subparser = parser.add_subparsers(dest = "command")

  # coregister_images command
  coregister_images = subparser.add_parser("coregister_images")
  coregister_images.add_argument("--moving_image", type = str)
  coregister_images.add_argument("--moving_mask", type = str)
  coregister_images.add_argument("--target_image", type = str)
  coregister_images.add_argument("--target_mask", type = str)
  coregister_images.add_argument("--output_file", type = str)

  # apply_transform command
  apply_transform = subparser.add_parser("apply_transform")
  apply_transform.add_argument("--image_file", type = str)
  apply_transform.add_argument("--transform_file", type = str)
  apply_transform.add_argument("--output_file", type = str)
  apply_transform.add_argument("--interpolation", type = str, default = "linear")
  
  # probseg_to_dseg command
  probseg_to_dseg = subparser.add_parser("probseg_to_dseg")
  probseg_to_dseg.add_argument("--probseg_file", type = str)
  probseg_to_dseg.add_argument("--output_file", type = str)
  probseg_to_dseg.add_argument("--probseg_threshold", type = float, default = 0.5)
  
  # clean_wm command
  clean_wm = subparser.add_parser("clean_wm")
  clean_wm.add_argument("--aseg_file", type = str)
  clean_wm.add_argument("--output_file", type = str)
  clean_wm.add_argument("--radius", type = float, default = 1)
  
  # clean_wmh command
  clean_wmh = subparser.add_parser("clean_wmh")
  clean_wmh.add_argument("--wmh_dseg_file", type = str)
  clean_wmh.add_argument("--wm_file", type = str)
  clean_wmh.add_argument("--ventricle_file", type = str)
  clean_wmh.add_argument("--output_file", type = str)
  
  # mask_norm command
  mask_norm = subparser.add_parser("mask_norm")
  mask_norm.add_argument("--input_file", type = str)
  mask_norm.add_argument("--mask_file", type = str)
  mask_norm.add_argument("--output_file", type = str)

  args = parser.parse_args()

  match args.command:
    case "coregister_images": 
      rigid_coregistration(
        moving_image = args.moving_image,
        moving_mask  = args.moving_mask, 
        target_image = args.target_image, 
        target_mask  = args.target_mask, 
        output_file  = args.output_file
      )
    case "apply_transform":
      transform_image(
        image_file      = args.image_file,
        transform_file  = args.transform_file, 
        output_file     = args.output_file, 
        interpolation   = args.interpolation
      )
    case "probseg_to_dseg":
      probseg_to_dseg(
        probseg_file      = args.probseg_file, 
        output_file       = args.output_file,
        probseg_threshold = args.probseg_threshold
      )
    case "clean_wm":
      clean_white_matter(
        aseg_file   = args.aseg_file,
        output_file = args.output_file,
        radius      = args.radius
      )
    case "clean_wmh": 
      clean_white_matter_hyperintensity(
        wmh_dseg_file  = args.wmh_dseg_file,
        wm_file        = args.wm_file,  
        ventricle_file = args.ventricle_file,  
        output_file    = args.output_file
      )
    case "mask_norm": 
      normalize_within_mask(
        input_file  = args.input_file,
        mask_file   = args.mask_file,
        output_file = args.output_file
      )
  