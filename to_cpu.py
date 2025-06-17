from glob import glob
from utils_tbox.utils_tbox import read_pklz, write_pklz, decompress_obj
import argparse
import sys


parser=argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
if __name__ == "__main__":
    args=parser.parse_args()
    s = args.i#"*.pklz.fold0"
    all_finished_fname = glob(os.path.join(root_dir, "lightning_logs/itgpt/**", s),recursive=True)
    print(all_finished_fname)
    sys.exit(0)
    for fname in all_finished_fname:
        results = read_pklz(fname)
        results["logits"] = results["logits"].cpu()
        results["yclass"] = results["yclass"].cpu()
        write_pklz(fname.replace(".fold0", ".cpu.fold0"), results)
