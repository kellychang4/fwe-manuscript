import re
import pandas as pd
import os.path as op

from utils import PARTICIPANTS


def main(scores_csv, output_dir):
  # read and edit scores csv participants identifier format
  df = pd.read_csv(scores_csv) # load score csv
  participant = [f"sub-{re.sub('_', 'x', x)}" for x in df["act_id"]]
  df.insert(0, "participant", participant) # insert participant column first
  df = df.drop("act_id", axis = 1) # drop act_id column

  # extract fazekas scores columns
  faz_cols = [x for x in df.columns if "leuk_faz" in x]
  df = df[["participant", *faz_cols]] # select participant and fazekas columns

  # rename columns to shorter names
  rename_dict = {x: re.sub("leuk_faz_", "", x) for x in df.columns}
  df = df.rename(columns = rename_dict)

  # save edited scores csv file
  df = df[df["participant"].isin(PARTICIPANTS)] # filter to passing participants
  df.to_csv(op.join(output_dir, "fazekas_scores.csv"), index = False, na_rep = "nan")
  print("Saved: fazekas_scores.csv\n")
  

if __name__ == "__main__":
  scores_csv = op.join("/path", "to", "ACTImaging_Fazekas.csv")
  output_dir = op.join("/path", "to", "output")

  main(scores_csv, output_dir)