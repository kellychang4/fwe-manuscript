import os
import re
import ants
import glob
import concurrent
import numpy as np
import pandas as pd
import os.path as op
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from utils import PARTICIPANTS


def calculate_odf_corr2(odf_list, mask_list, output_bname, chunk_dir, participant):
  df = [] # initialize 
  for dataset in ["multi-shell", "single-shell", "hcp-retest"]: # for each dataset
    # locate original, fwe, and mask files
    curr_odfs  = [x for x in odf_list if participant in x and dataset in x]
    fwe_fname  = [x for x in curr_odfs if re.search("desc-(ss|hcp)?fwe", x)]
    msmt_fname = [x for x in curr_odfs if re.search("desc-MSMT", x)]
    orig_fname = [x for x in curr_odfs 
                  if re.search("desc-(ss|hcp)?(original|single)", x) or
                  not (x in fwe_fname or x in msmt_fname)]
    mask_fname = [x for x in mask_list if participant in x]

    # if all files are found
    if orig_fname and fwe_fname and msmt_fname and mask_fname:
      # load original, fwe, MSMT, and mask files
      orig = ants.image_read(orig_fname[0])
      fwe  = ants.image_read(fwe_fname[0])
      msmt = ants.image_read(msmt_fname[0])

      # calculate percent differences for fwe methods
      avg_fwe  = (fwe.numpy() + orig.numpy()) / 2
      diff_fwe = (fwe.numpy() - orig.numpy()) / avg_fwe * 100

      # calculate percent differences for msmt methods
      avg_msmt  = (msmt.numpy() + orig.numpy()) / 2
      diff_msmt = (msmt.numpy() - orig.numpy()) / avg_msmt * 100

      # load and resample mask
      mask = ants.image_read(mask_fname[0])
      mask = ants.resample_image_to_target(mask, orig, interp = "nearestNeighbor")
      mask = np.round(mask.numpy())

      # append results for fwe method
      if dataset.startswith("hcp-"): # only NAWM for HCP dataset
        df.append({
          "participant": participant, "dataset": dataset, "method": "FWE",
          "nawm": np.nanmean(diff_fwe[mask == 1]),
        })

        df.append({
          "participant": participant, "dataset": dataset, "method": "MSMT",
          "nawm": np.nanmean(diff_msmt[mask == 1]),
        })
      else: # for multi-shell and single-shell datasets (WMH and NAWM)
        df.append({
          "participant": participant, "dataset": dataset, "method": "FWE",
          "nawm": np.nanmean(diff_fwe[mask == -1]),
          "wmh":  np.nanmean(diff_fwe[mask > 0])
        })

        df.append({
          "participant": participant, "dataset": dataset, "method": "MSMT",
          "nawm": np.nanmean(diff_msmt[mask == -1]),
          "wmh":  np.nanmean(diff_msmt[mask > 0])
        })

  # convert dataframe to pandas dataframe
  df = pd.DataFrame(df)

  # save dataframe to csv file
  save_name = f"{output_bname}_{participant}.csv"
  df.to_csv(op.join(chunk_dir, save_name), index = False)


def main(odf_list, mask_list, output_fname, max_workers = 10):
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
  
  # prepare calculate_odf_corr2 function with partial arguments
  thread_func = partial(
    calculate_odf_corr2, odf_list, mask_list, output_bname, chunk_dir)
  
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
  paths_hcp    = op.join("/path", "hcp-retest", "derivatives")
  output_dir   = op.join("/path", "to", "output")

  corr2_pattern = op.join("fodf_reliability", "sub-*", "*_param-corr2_dwimap.nii.gz")
  odf_list = [
    *glob.glob(op.join(paths_multi,  corr2_pattern)), 
    *glob.glob(op.join(paths_single, corr2_pattern)),
    *glob.glob(op.join(paths_hcp,    corr2_pattern)), 
  ]

  paths_mask1 = op.join(paths_multi, "hypermapper")
  paths_mask2 = op.join("/path", "hcp-retest", "data", "sub-*", "ses-01")
  mask_list = [
    *glob.glob(op.join(paths_mask1, "**", "*_label-WMH_desc-clean_dseg.nii.gz")),
    *glob.glob(op.join(paths_mask2, "*_label-WM_desc-clean_mask.nii.gz")),
  ]
  
  main(odf_list, mask_list, op.join(output_dir, "fodf_corr2.csv"))
