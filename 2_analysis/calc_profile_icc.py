import re
import os
import glob
import itertools
import concurrent
import pandas as pd
import os.path as op
import pingouin as pg
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from utils import PARTICIPANTS, METHOD_LIST, METRIC_DICT


def calculate_profile_icc(csv_list, output_bname, chunk_dir, participant):
  # extract participant and dataset information from filename
  dataset_list = set(["multi-shell", "single-shell", "hcp-retest"])
  method_list  = set(METHOD_LIST)
  metric_list  = set(METRIC_DICT.values())
  conds = itertools.product(dataset_list, method_list, metric_list)

  df = [] # initialize dataframe
  for dataset, method, metric in conds: # for each condition
    # locate original and fwe tract based on current condition
    curr_csv = [x for x in csv_list if participant in x and dataset in x
                and op.basename(op.dirname(op.dirname(x))) == method
                and re.search(metric, op.basename(x))]

    if len(curr_csv) == 2: # if both files exist across split/session
      # load and prepare the csv files into a pandas dataframe
      csv_1 = pd.read_csv([x for x in curr_csv if "split-1" in x or "ses-01" in x][0])
      csv_2 = pd.read_csv([x for x in curr_csv if "split-2" in x or "ses-02" in x][0])
      csv_1["tract_node"] = [f"{x}_{y}" for x, y in zip(csv_1["tract"], csv_1["nodeID"])]
      csv_2["tract_node"] = [f"{x}_{y}" for x, y in zip(csv_2["tract"], csv_2["nodeID"])]

      # edit tract information from condition labels
      metric = [k for k, x in METRIC_DICT.items() if metric == x][0]

      # idenfity all unique tracts
      tracts = list(set(csv_1["tract"]) & set(csv_2["tract"]))

      for trk in tracts: # for each tract
        # combine the two csv files into a single dataframe
        values_1 = csv_1[csv_1["tract"] == trk][["tract_node", "value"]]
        values_2 = csv_2[csv_2["tract"] == trk][["tract_node", "value"]]
        values_1["split"] = "split-1"; values_2["split"] = "split-2"
        df_icc = pd.concat([values_1, values_2], axis = 0).reset_index(drop = True)

        # calculate the ICC of the tract for the subject
        results = pg.intraclass_corr(
          data = df_icc, targets = "tract_node", raters = "split", ratings = "value")
        icc_value = results[results["Description"] == "Average random raters"]["ICC"].values[0]
        
        # append the ICC value to the dataframe
        df.append({ 
          "participant": participant, "dataset": dataset, "method": method, 
          "metric": metric, "tract": trk, "icc": icc_value 
        })
        
  # convert dataframe to pandas dataframe
  df = pd.DataFrame(df)

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

  # prepare calculate_profile_icc function with partial arguments
  thread_func = partial(calculate_profile_icc, csv_list, output_bname, chunk_dir)
  
  with ThreadPoolExecutor(max_workers = max_workers) as executor:
    futures = [] # initialize futures list
    for i, participant in enumerate(participant_list): # for each processing chunk 
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

  csv_pattern = op.join(
    "profiles-reliability", "sub-*", "afq-*", "s*-*", "*_dwimap.csv")
  csv_list = [
    *glob.glob(op.join(paths_multi,  csv_pattern)), 
    *glob.glob(op.join(paths_single, csv_pattern)), 
    *glob.glob(op.join(paths_hcp,    csv_pattern)) 
  ]
  
  main(csv_list, op.join(output_dir, "profile_icc.csv"))