import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class TheDataset(Dataset):
    def __init__(self, data, device="cpu"):
        self.patids = list(data.keys())
        self.data = data  ###  {cutter_no: {m:data[cutter_no]["calX"][m].to(device=device)for m in data[cutter_no]["calX"].keys()} for cutter_no in data.keys()}
        self.mu = None
        self.sigma = None
        self.device = device
        y = torch.cat([d["targets"] for d in self.data.values()]).squeeze(-1).numpy()#.#tolist()
        self.class_weights = torch.from_numpy(compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)).to(dtype=torch.float32) #/len(self.data.values())
    
    def __len__(self):
        return len(self.patids)

    def __getitem__(self, i):
        thedata = self.data[self.patids[i]]
        thedata["class_weights"] = self.class_weights  ###[thedata["targets"].int()]
        return thedata

