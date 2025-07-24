import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datetime import datetime
import pandas as pd


groups_stems = ["heart_rate", "respiratory_rate", "Body Temperature", 
                "Body weight", "Diastolic blood pressure","Heart rate",
                "O/E - muscle mass", "Skin Temperature", "Systolic blood pressure","Total body water"]

def get_variable_groups(train_dataset):
        
    variable_groups = {k: [v for v in train_dataset.feature_names if k in v] for k in groups_stems}
    others = {v:[v] for v in train_dataset.feature_names if not v in list(set( sum(list(variable_groups.values()),[]) ))}
    if len(others) > 0:
        variable_groups = {**variable_groups, **others}
    return variable_groups


class TheDataset(Dataset):
    def __init__(self, data, device="cpu"):
        patids = list(set(list(data.keys())))
        self.patid_to_id = {k:i for i,k in enumerate(patids)}
        self.data = {self.patid_to_id[k]:v for k,v in data.items()}  ###  {cutter_no: {m:data[cutter_no]["calX"][m].to(device=device)for m in data[cutter_no]["calX"].keys()} for cutter_no in data.keys()}
        self.ids = list(self.data.keys())
        self.mu = None
        self.sigma = None
        self.device = device
        y = torch.cat([d["targets_int"] for d in self.data.values()]).squeeze(-1).numpy()#.#tolist()
        self.class_weights = torch.from_numpy(compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)).to(dtype=torch.float32) #/len(self.data.values())
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        thedata = self.data[self.ids[idx]]
        thedata["class_weights"] = self.class_weights  ###[thedata["targets"].int()]
        thedata["id"] = self.ids[idx]
        thedata["class_weights"] = self.class_weights  ###[thedata["targets"].int()]
        return thedata


def get_tihm_modalities(df, timeline, target, groups, ref_date=None):
    """Group data variables according to "groups"""
    DTYPE = torch.float32
    assert(timeline.shape[0] == df.shape[0])
    if ref_date is None:
        ref_date = datetime(year=2019, month=1, day=1)
    calX = {}
    idx = {}
    timelines = {}
    i_targets = np.isnan(target)==0
    targets = torch.from_numpy(target[i_targets]).to(dtype=DTYPE)
    inference_timeline = torch.from_numpy((pd.to_datetime(timeline[i_targets])-ref_date).total_seconds().values/3600/24).to(dtype=DTYPE)
    
    for modality_name, vv in groups.items():
        theidx = (df[vv].isna().sum(1) == 0).values
        idx[modality_name]        = torch.from_numpy(theidx).to(dtype=DTYPE)
        calX[modality_name]       = torch.from_numpy(df[vv][theidx].values).to(dtype=DTYPE)
        timelines[modality_name]  = torch.from_numpy((pd.to_datetime(timeline[theidx])-ref_date).total_seconds().values/3600/24).to(dtype=DTYPE)
        if timelines[modality_name].shape[0]>0:
            #diff =  torch.diff(timelines[modality_name], prepend=torch.tensor([timelines[modality_name][0].item()])).to(dtype=DTYPE)
            calX[modality_name] = torch.cat([calX[modality_name], timelines[modality_name].unsqueeze(-1)],axis=1).to(dtype=DTYPE)
        else:# No data at all ?
            calX[modality_name] = torch.zeros((1,len(groups[modality_name])+1))
            timelines[modality_name]= torch.zeros((1))+inference_timeline[0]
    calX["reference"]  = inference_timeline
    return {"data":calX, "targets_int":targets.int()}

def to_dict(train_dataset, test_dataset):
    """Group data per patient"""
    train_patients = {}
    for patid in np.unique(train_dataset.train_patient_id):
        train_patients[patid] = {}
        idx = train_dataset.train_patient_id==patid
        train_patients[patid]["timeline"] = train_dataset.train_date[idx]
        train_patients[patid]["data"] = train_dataset.train_data[idx]
        train_patients[patid]["target"] = train_dataset.train_target[idx]

    test_patients = {}
    for patid in np.unique(test_dataset.test_patient_id):
        test_patients[patid] = {}
        idx = test_dataset.test_patient_id==patid
        test_patients[patid]["timeline"] = test_dataset.test_date[idx]
        test_patients[patid]["data"] = test_dataset.test_data[idx]
        test_patients[patid]["target"] = test_dataset.test_target[idx]
    return train_patients, test_patients

def get_tihm_data(patients,groups,columns, ref_date=None):
    if ref_date is None:
        ref_date = min([min(v["timeline"]) for v in patients.values()])
        ref_date = datetime(year=ref_date.year,month=ref_date.month,day=ref_date.day)
        
    data = {}
    for i, (patid, v) in enumerate(patients.items()):
        if i == -1:
            fig, ax = plt.subplots()
            ax.imshow(v["target"], aspect="auto")
            fig, ax = plt.subplots()
            ax.imshow(v["data"], aspect="auto")
        
        timeline = v["timeline"]
        thedata = v["data"]
        target = v["target"][:, 1] >= 1
        df = pd.DataFrame(data=thedata, columns=columns)
        data[patid] = get_tihm_modalities(df, timeline, target, groups,ref_date=ref_date)
    a_patid = list(data.keys())[0]

    return data, ref_date
