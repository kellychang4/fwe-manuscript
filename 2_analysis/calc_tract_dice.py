import os
import re
import glob
import itertools
import concurrent
import pandas as pd
import os.path as op
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from dipy.io.streamline import load_tractogram
from AFQ.utils.volume import density_map, dice_coeff

from utils import PARTICIPANTS, METHOD_LIST


def _get_method(x):
  method = op.basename(op.dirname(op.dirname(x)))
  if method.startswith("ses-"): # if session folder instead of method folder
    method = op.basename(op.dirname(op.dirname(op.dirname(op.dirname(x))))) 
  return method


def calculate_tract_dice(trx_list, output_bname, chunk_dir, participant):
  # extract dataset, method, and tract name information from filename
  dataset_list = set(["multi-shell", "single-shell", "hcp-retest"])
  method_list  = set(METHOD_LIST)
  trkname_list = set([re.sub(".+_desc-(\w+)_.+", "\\1", op.basename(x)) 
                      for x in trx_list])
  conds = itertools.product(dataset_list, method_list, trkname_list)

  df = [] # initialize dataframe
  for dataset, method, trkname in conds: # for each condition
    # locate original and fwe tract based on current condition
    curr_trxs = [x for x in trx_list if participant in x and dataset in x 
                 and _get_method(x) == method and trkname in x]

    if len(curr_trxs) == 2: # if both files exist
      # locate diffusion references files
      refs = [glob.glob(op.join(op.dirname(op.dirname(x)), 
              "*_desc-APM_dwi.nii.gz"))[0] for x in curr_trxs]

      # load trx files from split-1 and split-2 (bbox_valid_check = False)
      curr_trxs = [load_tractogram(x, reference = y, bbox_valid_check = False) 
                    for x, y in zip(curr_trxs, refs)]
      
      # conform streamlines to same space
      curr_trxs[0].to_vox()
      curr_trxs[1].to_vox()
      
      # remove invalid streamlines before calculating density maps
      curr_trxs[0].remove_invalid_streamlines()
      curr_trxs[1].remove_invalid_streamlines()

      # calculate trx density maps
      maps = [density_map(x) for x in curr_trxs]

      # calculate weighted dice coefficient
      coeff = dice_coeff(maps[0], maps[1], weighted = True)    

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


def main(trx_list, output_fname, max_workers = 10):
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

  # prepare calculate_tract_dice function with partial arguments
  thread_func = partial(calculate_tract_dice, trx_list, output_bname, chunk_dir)
  
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

  trx_pattern = op.join("reliability", "sub-*", "split-*", "afq-*", "bundles", "*.trx")
  trx_list = [
    *glob.glob(op.join(paths_multi,  trx_pattern)), 
    *glob.glob(op.join(paths_single, trx_pattern)), 
    *glob.glob(op.join(paths_hcp, "afq-*", "sub-*", "ses-*", "bundles", "*.trx"))
  ]
  
  main(trx_list, op.join(output_dir, "tract_dice.csv"))