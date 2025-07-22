import os
import re
import glob
import concurrent
import pandas as pd
import os.path as op
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from utils import PARTICIPANTS


def collate_tract_statistics(csv_list, output_bname, chunk_dir, participant):
  # define current csv list for participant
  curr_list = [x for x in csv_list if participant in x]

  df = [] # initialize empty list to store dataframes
  for fname in curr_list: # for each csv file chunk
    # extract dataset, method, and metric information
    dataset = fname.split(os.sep)[3] # extract dataset name
    method = op.basename(op.dirname(op.dirname(fname))); split = "" # initialize method and split variables
    if method.startswith("sub-"): # if hcp-retest organization
      split = op.basename(op.dirname(fname)) # determine split variable
      method = op.basename(op.dirname(op.dirname(op.dirname(fname))))

    # read csv file and append file information
    curr_df = pd.read_csv(fname).rename(columns = {"Unnamed: 0": "tract"})
    curr_df["participant"] = participant
    curr_df["dataset"]     = dataset
    curr_df["method"]      = method
    curr_df["split"]       = split
    
    # append dataframe to list
    df.append(curr_df)

  # concatenate list of dataframes
  df = pd.concat(df) 

  # select and rename columns of dataframe
  x_cols = ["participant", "dataset", "method", "split", "tract"]
  y_col  = list(set(df.columns) - set(x_cols))
  df = df[[*x_cols, *y_col]]

  # save dataframe to csv file
  save_name = f"{output_bname}_{participant}.csv"
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

  # prepare collate_tract_statistics function with partial arguments
  thread_func = partial(
    collate_tract_statistics, csv_list, output_bname, chunk_dir)
  
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

  file_pattern = "*_recogmethod-AFQ_desc-slCount_dwi.csv"
  streamline_list = [
    *glob.glob(op.join(paths_multi, "afq-*", "sub-*", file_pattern)), 
    *glob.glob(op.join(paths_single, "afq-*", "sub-*", file_pattern)), 
    *glob.glob(op.join(paths_hcp, "afq-*", "sub-*", "ses-*", file_pattern))
  ]
  main(streamline_list, op.join(output_dir, "n_streamlines.csv"))  

  file_pattern = "*_recogmethod-AFQ_desc-medianBundleLengths_dwi.csv"
  mdnlength_list = [
    *glob.glob(op.join(paths_multi, "afq-*", "sub-*", file_pattern)), 
    *glob.glob(op.join(paths_single, "afq-*", "sub-*", file_pattern)), 
    *glob.glob(op.join(paths_hcp, "afq-*", "sub-*", "ses-*", file_pattern))
  ]
  main(mdnlength_list, op.join(output_dir, "mdn_lengths.csv"))  