import argparse

import AFQ.api.bundle_dict as abd
from AFQ.definitions.image import ImageFile
from AFQ.api.participant import ParticipantAFQ


def main(dwi_data_file, bval_file, bvec_file, mask_file, fodf_file, output_dir,
         stop_mask_file = None, stop_threshold = 0.2):
  # Define bundles dictionary
  bundles = abd.default18_bd() + abd.callosal_bd()

  # Define brain main image file
  brain_mask_definition = ImageFile(path = mask_file)

  # Define tracking parameters
  tracking_params = {
    "n_seeds": 2, # n_seeds per dim, [2 2 2] = 8 seeds per voxel
    "odf_model": ImageFile(path = fodf_file), 
    "stop_threshold": stop_threshold,
    "trx": True
  }

  # If stop mask provided...
  if stop_mask_file: # add stop mask to tracking_params
    tracking_params["stop_mask"] = ImageFile(path = stop_mask_file)
    del tracking_params["stop_threshold"] # delete stop threshold

  # define ParticipantAFQ object
  myafq = ParticipantAFQ(
    dwi_data_file         = dwi_data_file, 
    bval_file             = bval_file,
    bvec_file             = bvec_file,
    output_dir            = output_dir,
    bundle_info           = bundles,
    brain_mask_definition = brain_mask_definition, 
    tracking_params       = tracking_params
  )
  myafq.cmd_outputs(cmd = "rm", dependent_on = "recog")

  # call export_all, start tractography
  myafq.export_all(xforms = False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dwi_data_file", type = str)
  parser.add_argument("--bval_file", type = str)
  parser.add_argument("--bvec_file", type = str)
  parser.add_argument("--mask_file", type = str)
  parser.add_argument("--fodf_file", type = str)
  parser.add_argument("--output_dir", type = str)
  parser.add_argument("--stop_mask_file", type = str, default = None)
  parser.add_argument("--stop_threshold", type = float, default = 0.2)
  args = parser.parse_args()
  
  main(
    dwi_data_file  = args.dwi_data_file,
    bval_file      = args.bval_file, 
    bvec_file      = args.bvec_file, 
    mask_file      = args.mask_file, 
    fodf_file      = args.fodf_file,
    output_dir     = args.output_dir, 
    stop_mask_file = args.stop_mask_file, 
    stop_threshold = args.stop_threshold
  )