import ants
import argparse
import numpy as np
import nibabel as nib


def create_qsiprep_wm_mask(dseg_file, reference_file, output_file):
  # read the dseg image and reference image
  dseg_image = ants.image_read(dseg_file)
  ref_image  = ants.image_read(reference_file)

  # resample the white matter mask to the reference
  wm_image = ants.resample_image_to_target(
    dseg_image, ref_image, interp_type = "genericLabel")
  wm_image = wm_image == 3 # binarize the white matter mask
  
  # save the white matter mask
  ants.image_write(wm_image, output_file)
  print(f"Saved as: {output_file}")


def run_fwe(dwi_data_file, bvals_files, mask_file, fwf_file, 
            output_file, Diso = 3.0e-3):
  
  # load diffusion, b-values, mask, and free water fraction images
  dwi_image  = nib.load(dwi_data_file)
  bvals      = np.loadtxt(bvals_files)
  mask_image = nib.load(mask_file).get_fdata()
  fwf        = nib.load(fwf_file).get_fdata()

  # extract b0 signal from dwi image
  S0 = dwi_image.get_fdata()[..., bvals == 0].mean(axis = -1)

  # compute free water signal
  fw_decay  = np.exp(-bvals * Diso) # free-water exponential decay
  fw_signal = (S0 * fwf).reshape(-1, 1) * fw_decay
  fw_signal = fw_signal.reshape(fwf.shape + (dwi_image.shape[-1], ))
  fw_signal[mask_image == 0] = 0 # set fw signal = 0 outside of brain
  
  # compute free water eliminated signal
  fwe_signal = dwi_image.get_fdata() - fw_signal

  # save free water eliminated signal
  fwe_image = nib.Nifti1Image(fwe_signal, affine = dwi_image.affine)
  nib.save(fwe_image, output_file)
  print(f"Saved as: {output_file}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparser = parser.add_subparsers(dest = "command")
  
  # qsiprep white matter mask argument parser
  qsiprep_wmmask = subparser.add_parser("qsiprep_wmmask")
  qsiprep_wmmask.add_argument("--dseg_file", type = str)
  qsiprep_wmmask.add_argument("--reference_file", type = str)
  qsiprep_wmmask.add_argument("--output_file", type = str)

  # free water elimination argument parser
  fwe = subparser.add_parser("fwe")
  fwe.add_argument("--dwi_data_file", type = str)
  fwe.add_argument("--bvals_file", type = str)
  fwe.add_argument("--mask_file", type = str)
  fwe.add_argument("--fwf_file", type = str)
  fwe.add_argument("--output_file", type = str)

  args = parser.parse_args()

  match args.command:
    case "qsiprep_wmmask": 
      create_qsiprep_wm_mask(
        dseg_file      = args.dseg_file,
        reference_file = args.reference_file, 
        output_file    = args.output_file
      )
    case "fwe": 
      run_fwe(
        dwi_data_file = args.dwi_data_file,
        bvals_file    = args.bvals_file,
        mask_file     = args.mask_file,
        fwf_file      = args.fwf_file,
        output_file   = args.output_file
      )

  