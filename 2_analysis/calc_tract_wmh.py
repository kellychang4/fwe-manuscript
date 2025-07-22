import os
import re
import glob
import ants
import itertools
import concurrent
import pandas as pd
import os.path as op
import nibabel as nib
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from dipy.io.streamline import load_tractogram
from AFQ.utils.volume import density_map, dice_coeff

from utils import PARTICIPANTS, METHOD_LIST


def _get_method(x):
  return op.basename(op.dirname(op.dirname(op.dirname(x))))


def calculate_tract_wmh(trx_list, mask_list, output_bname, chunk_dir, participant):
  # extract dataset, method, and tract name information from filename
  dataset_list = set(["multi-shell", "single-shell"])
  method_list  = set(METHOD_LIST)
  trkname_list = set([re.sub(".+_desc-(\w+)_.+", "\\1", op.basename(x)) 
                      for x in trx_list])
  conds = itertools.product(dataset_list, method_list, trkname_list)

  wmh_fname = [x for x in mask_list if participant in x]
  if wmh_fname: # if white matter hyerintensity mask exists
    wmh_image = ants.image_read(wmh_fname[0])

    df = [] # initialize dataframe
    for dataset, method, trkname in conds: # for each condition
      # locate original and fwe tract based on current condition
      curr_trx = [x for x in trx_list if participant in x and dataset in x 
                  and _get_method(x) == method and trkname in x]

      if curr_trx: # if tract and white matter hyerintensity mask exists
        # extract tractogram and wmh mask files
        curr_trx, wmh_fname = curr_trx[0], wmh_fname[0]

        # locate diffusion references file
        reference_fname = glob.glob(
          op.join(op.dirname(op.dirname(curr_trx)), "*_desc-APM_dwi.nii.gz"))[0]
        
        # resample wmh mask to diffusion reference space
        reference  = ants.image_read(reference_fname)
        wmh_values = ants.resample_image_to_target(wmh_image, reference)
        wmh_values = (wmh_values.numpy() > 0) * 1.0 # only wmh mask

        # load trx files from split-1 and split-2 (bbox_valid_check = False)
        reference = nib.load(reference_fname) # load reference image as nibabel object
        curr_trx = load_tractogram(curr_trx, reference = reference, bbox_valid_check = False) 
        curr_trx.to_vox() # conform streamlines to same space
        curr_trx.remove_invalid_streamlines() # remove invalid streamlines before calculating density maps

        # calculate trx density maps
        density_image = density_map(curr_trx)

        # calculate weighted dice coefficient
        coeff = dice_coeff(density_image, wmh_values, weighted = True)    

        # append dice coefficient to dataframe
        df.append({
          "participant": participant, "dataset": dataset, "method": method, 
          "tract": trkname, "dice": coeff
        })  

    # convert dataframe to pandas dataframe
    df = pd.DataFrame(df)

    # save dataframe to csv file
    save_name = f"{output_bname}_{participant}.csv"
    df.to_csv(op.join(chunk_dir, save_name), index = False)


def main(trx_list, mask_list, output_fname, max_workers = 10):
  # define chunk basename and directory
  output_dir   = op.dirname(output_fname)
  output_bname = op.basename(output_fname).split(".")[0]
  chunk_dir    = op.join(output_dir, output_bname)
  os.makedirs(chunk_dir, exist_ok = True)

  # locate existing participant files and remove from participant list
  chunk_list = glob.glob(op.join(chunk_dir, f"{output_bname}_*.csv"))
  completed_list = [re.sub(f"{output_bname}_(.+).csv", "\\1", op.basename(x)) 
                    for x in chunk_list]
  participant_list = list(set(PARTICIPANTS) - set(completed_list))

  # prepare calculate_tract_wmh function with partial arguments
  thread_func = partial(
    calculate_tract_wmh, trx_list, mask_list, output_bname, chunk_dir)
  
  with ThreadPoolExecutor(max_workers = max_workers) as executor:
    futures = [] # initialize futures list
    for participant in participant_list: # for each processing chunk 
      future = executor.submit(thread_func, participant)
      futures.append(future)

    for future in concurrent.futures.as_completed(futures):
      _ = future.result() # call result to ensure clean exit

  # concatenate all chunked dataframes
  chunk_list = glob.glob(op.join(chunk_dir, f"{output_bname}_*.csv"))
  df = pd.concat([pd.read_csv(x) for x in chunk_list])

  # save concatenated dataframe to csv file
  df.to_csv(op.join(output_dir, f"{output_bname}.csv"), index = False)
  print(f"Saved: {output_bname}.csv")


if __name__ == "__main__":
  paths_multi  = op.join("/path", "multi-shell", "derivatives")
  paths_single = op.join("/path", "single-shell", "derivatives")
  paths_wmh    = op.join(paths_multi, "hypermapper")
  output_dir   = op.join("/path", "to", "output")

  trx_pattern = op.join("afq-*", "sub-*", "bundles", "*.trx")
  trx_list = [
    *glob.glob(op.join(paths_multi,  trx_pattern)), 
    *glob.glob(op.join(paths_single, trx_pattern)), 
  ]

  mask_list = glob.glob(
    op.join(paths_wmh, "sub-*", "*_label-WMH_desc-clean_dseg.nii.gz"))
  
  main(trx_list, mask_list, op.join(output_dir, "tract_wmh.csv"))