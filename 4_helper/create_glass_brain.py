import ants
import argparse
import os.path as op


def main(aseg_file, output_fname):
  # Read aseg image
  aseg_image = ants.image_read(aseg_file)
  
  # Create "Glass Brain" Mask from aseg Image
  upsample_resolution = [aseg_image.spacing[0] / 2,] * 3 # upsample resolution by 2
  mask_image = ants.resample_image(aseg_image, upsample_resolution) # upsample image
  mask_image = ants.smooth_image(mask_image, sigma = 3) # smooth image with gaussian
  mask_image = ants.threshold_image(mask_image, low_thresh = 0.25) # threshold image
  mask_image = ants.morphology(mask_image, operation = "close", radius = 5) # close holes
  
  # Save mask image
  ants.image_write(mask_image, output_fname)
  print(f"Saved: {output_fname}")


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument("--aseg_file", type = str)
  parser.add_argument("--output_fname", type = str)
  args = parser.parse_args()
  
  main(
    aseg_file    = args.aseg_file,
    output_fname = args.output_fname
  )