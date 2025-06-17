from glob import glob
from utils_tbox.utils_tbox import read_pklz, write_pklz, decompress_obj

all_finished_fname = glob(os.path.join(root_dir, "lightning_logs/itgpt/**","*.pklz.fold0"),recursive=True)
print(all_finished_fname)

for fname in all_finished_fname:
    results = read_pklz(fname)
    results["logits"] = results["logits"].cpu()
    results["yclass"] = results["yclass"].cpu()
    write_pklz(fname.replace(".fold0", ".cpu.fold0"), results)
