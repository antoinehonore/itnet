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
    #sys.exit(0)
    for fname in all_finished_fname:
        print(fname)
        results = read_pklz(fname)
        if isinstance(results, dict):
            results["logits"] = results["logits"].cpu()
            results["yclass"] = results["yclass"].cpu()
            #results["ltrainer"] = results["ltrainer"].cpu()
            results[i]["ltrainer"] = None
        else:
            for i in range(len(results)):
                results[i]["logits"] = results[i]["logits"].cpu()
                results[i]["yclass"] = results[i]["yclass"].cpu()
                results[i]["ltrainer"] = None
                #results[i]["ltrainer"].model.itgpt.normalized_batch = None
                #results[i]["ltrainer"] = results[i]["ltrainer"].cpu()
                #print()
        write_pklz(fname+".cpu", results)
    sys.exit(0)