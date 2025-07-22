import glob
import pandas as pd
import os.path as op


def main(qc_list, output_dir):
  df = [] # initialize empty list
  for fname in qc_list: # for each csv file
    curr_df = pd.read_csv(fname)
    
    df.append({
      "participant": curr_df["subject_id"].values[0],
      "neicorr": curr_df["raw_neighbor_corr"].values[0]
    })

  # convert list of dictionaries to dataframe
  df = pd.DataFrame(df)

  # save quality control dataframe
  df.to_csv(op.join(output_dir, "quality_control.csv"), index = False)
  print("Saved: quality_control.csv")  

  # calculate and filter low neighbors correlations
  lb = df["neicorr"].mean() - (2 * df["neicorr"].std())
  df = df[df["neicorr"] > lb] # remove low qc

  # remove participants with failed QSIprep outputs
  with open(op.join(output_dir, "QSIPREP_FAILED.txt")) as f:
    QSIPREP_FAILED = f.read().splitlines()
  df = df[~df["participant"].isin(QSIPREP_FAILED)]

  # write quality controlled participants 
  df["participant"].to_csv(op.join(output_dir, "PARTICIPANTS.txt"), 
                           header = None, index = None, sep = " ")
  print("Saved: PARTICIPANTS.txt\n")


if __name__ == "__main__":
  paths_qsiprep = op.join("/path", "multi-shell", "derivatives", "qsiprep")
  output_dir    = op.join("/path", "to", "output")

  qc_list = glob.glob(op.join(paths_qsiprep, "sub-*", "dwi", "*_desc-image_qc.csv"))

  main(qc_list, output_dir)