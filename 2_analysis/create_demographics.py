import pandas as pd
import os.path as op

from utils import PARTICIPANTS


def main(csv_fname, output_dir):
  df_full = pd.read_csv(csv_fname) # read demographics csv file
  df = df_full[df_full["participant"].isin(PARTICIPANTS)] # filter to passing
  df.to_csv(op.join(output_dir, "demographics.csv"), index = False, na_rep = "nan")
  print("Saved: demographics.csv\n")
  

if __name__ == "__main__":
  demographics_csv = op.join("/path", "to", "ACTImaging_Demographics.csv")
  output_dir = op.join("/path", "to", "output")

  main(demographics_csv, output_dir)