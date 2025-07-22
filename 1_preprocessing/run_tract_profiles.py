import re
import glob
import argparse
import numpy as np
import pandas as pd
import os.path as op
import nibabel as nib

import dipy.stats.analysis as dsa
from dipy.io.streamline import load_tractogram


def main(vol_fname, trk_dir, trk_ref, output_csv):
  # Read input and tractogram reference images
  vol_image = nib.load(vol_fname)
  ref_image = nib.load(trk_ref)

  # Create tract profiles from subject trk files
  trk_df = list() # initialize
  trk_files = [
    *glob.glob(op.join(trk_dir, "*.trx")), 
    *glob.glob(op.join(trk_dir, "*.trk"))
  ]
  for fname in trk_files: # for each trk files
    # Extract and adjust bundle name for dataframe
    bundle_name = re.sub(".+_desc-(\\w+)_.+", "\\1", op.basename(fname))
    if re.search("[LR]$", bundle_name): # adjust bundle name 
        bundle_name = re.sub("(\w)$", "_\\1", bundle_name) 
    print(f"Profiling Tract: {bundle_name}")

    # Load trk file with reference image
    trk = load_tractogram(fname, reference = ref_image)
    trk.to_vox() # covert tract to voxel space

    # Sample tract profile values from input image
    try: 
      weights = dsa.gaussian_weights(trk.streamlines)
      profile = dsa.afq_profile(
          data    = vol_image.get_fdata(), 
          bundle  = trk.streamlines, 
          affine  = np.eye(4),
          weights = weights
      ) 

      # Collect tract values into dataframe
      trk_info = { 
        "tract": bundle_name, 
        "nodeID": np.arange(len(profile)), 
        "value": profile
      }
      trk_info = pd.DataFrame.from_dict(trk_info)
      trk_df.append(trk_info) # append tract information to df
    except:
      print("  -> Skipped")

  # Concatenate tractogram dfs and write profiles as csv
  df = pd.concat(trk_df).reset_index().drop("index", axis = 1)
  df.to_csv(output_csv, index = False) # write csv!
  print(f"Saved: {op.basename(output_csv)}\n\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--vol_fname", type = str)
  parser.add_argument("--trk_dir", type = str)
  parser.add_argument("--trk_ref", type = str)
  parser.add_argument("--output_csv", type = str)
  args = parser.parse_args()

  main(
    vol_fname  = args.vol_fname, 
    trk_dir    = args.trk_dir, 
    trk_ref    = args.trk_ref,
    output_csv = args.output_csv
  )