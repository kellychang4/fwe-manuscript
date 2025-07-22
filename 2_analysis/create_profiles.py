import os
import re
import glob
import concurrent
import pandas as pd
import os.path as op
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from utils import PARTICIPANTS, METRIC_DICT


def collate_profiles(csv_list, chunk_bname, chunk_dir, participant):
  # define current csv list  
  curr_list = [x for x in csv_list if participant in x]

  if curr_list: # if files found
    df = [] # initialize empty list to store dataframes
    for fname in curr_list: # for each participant file
      # extract dataset, method, and metric information
      dataset = fname.split(os.sep)[3] # extract dataset name
      method = op.basename(op.dirname(fname)); split = "" # initialize method and split variables
      if not method.startswith("afq-"): # if profiles-reliability organization
        split = method # reassign split variable
        method = op.basename(op.dirname(op.dirname(fname)))
      metric = [k for k, x in METRIC_DICT.items() if re.search(x, op.basename(fname))][0]

      # read csv file and append file information
      curr_df = pd.read_csv(fname)
      curr_df["participant"] = participant
      curr_df["dataset"]     = dataset
      curr_df["method"]      = method
      curr_df["split"]       = split
      curr_df["metric"]      = metric

      # append dataframe to list
      df.append(curr_df)

    # concatenate list of dataframes
    df = pd.concat(df) 

    # select and rename columns of dataframe
    df = (df[["participant", "dataset", "sampling", "method", "split", 
              "metric", "tract", "nodeID", "value"]]
            .rename(columns = {"nodeID": "node"}))

    # save dataframe to csv file
    save_name = f"{chunk_bname}_{participant}.csv"
    df.to_csv(op.join(chunk_dir, save_name), index = False)

def main(csv_list, output_fname, max_workers = 10):
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

  # prepare collate_profiles function with partial arguments
  thread_func = partial(collate_profiles, csv_list, output_bname, chunk_dir)
  
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
  df.to_csv(output_fname, index = False)
  print(f"Saved: {op.basename(output_fname)}")


if __name__ == "__main__":
  paths_multi  = op.join("/path", "multi-shell", "derivatives")
  paths_single = op.join("/path", "single-shell", "derivatives")
  paths_hcp    = op.join("/path", "hcp-retest", "derivatives")
  output_dir   = op.join("/path", "to", "output")

  profiles_pattern = op.join("profiles", "sub-*", "afq-*", "*_dwimap.csv")
  profiles_list = [
    *glob.glob(op.join(paths_multi,  profiles_pattern)),  
    *glob.glob(op.join(paths_single, profiles_pattern))
  ]
  main(profiles_list, op.join(output_dir, "profiles.csv"))

  reilprofiles_pattern = op.join("profiles-reliability", "sub-*", "afq-*", "split-*", "*.csv")
  reliprofiles_list = [
    *glob.glob(op.join(paths_multi,  reilprofiles_pattern)), 
    *glob.glob(op.join(paths_single, reilprofiles_pattern)),
    *glob.glob(op.join(paths_hcp, "profiles-reliability", "sub-*", "afq-*", "ses-*", "*.csv"))
  ]
  main(reliprofiles_list, op.join(output_dir, "reliprofiles.csv"))