from glob import glob
from utils_tbox.utils_tbox import read_pklz, write_pklz, decompress_obj
import argparse
import sys
import os 

parser=argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
if __name__ == "__main__":
    args=parser.parse_args()
    s = args.i#"*.pklz.fold0"
    root_dir = "."
    all_finished_fname = glob(os.path.join(root_dir, "lightning_logs/itgpt/**", s),recursive=True)
    print(all_finished_fname)
    
    for fname in all_finished_fname:
        results = read_pklz(fname)
        results["logits"] = results["logits"].cpu()
        results["yclass"] = results["yclass"].cpu()
        sys.exit(0)
        write_pklz(fname.replace(".fold0", ".cpu.fold0"), results)
