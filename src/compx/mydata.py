import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src.datastruct import TSdata

#slope, intercept = np.polyfit(dfmu_train[c].dropna().index, dfmu_train[c].dropna(), 1)

class TheDataset(Dataset):
    def __init__(self, data):
        self.ids = list(data.keys())
        self.data = data  ###  {cutter_no: {m:data[cutter_no]["calX"][m].to(device=device)for m in data[cutter_no]["calX"].keys()} for cutter_no in data.keys()}
        #self.slopes = {v_id: {k:compute_slope(v) for k,v in self.data[v_id]["data"].items() if (k!="specs") and (k!="reference")} for v_id in self.data.keys()}

    def get_class_weights(self):
        self.n_classes = self.data[self.ids[0]]["targets_OH"].shape[1]

        y = torch.cat([d["targets_int"] for d in self.data.values()]).squeeze(-1).numpy()#.#tolist()
        self.class_weights = torch.from_numpy(compute_class_weight(class_weight="balanced", classes=np.arange(self.n_classes), y=y)).to(dtype=torch.float32) #/len(self.data.values())
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        thedata = self.data[self.ids[idx]]
        thedata["class_weights"] = self.class_weights  ###[thedata["targets"].int()]
        thedata["id"] = self.ids[idx]
        thedata["data"]["reference"] = thedata["data"]["reference"].reshape(-1,1)
        return thedata

def get_ts_edge_index(N, device="cpu"):
    tmp = torch.arange(N-1,device=device,dtype=torch.long).reshape(1,-1)
    edge_index = torch.cat([tmp,tmp+1])
    return edge_index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot_confusion_matrix
import os 
import torch

from utils_tbox.utils_tbox import read_pklz, write_pklz

num_variables = [["171_0"], ["666_0"], ["427_0"], 
                    ["837_0"], ["309_0"], ["835_0"], 
                    ["370_0"], ["100_0"]
                ]

cat_variables = {varname: ["_".join([varname,str(i)]) for i in range(n_bins)] 
                    for varname, n_bins in zip(["167", "272", "291", "158", "459", "397"], 
                                            [10, 10, 11, 10, 20, 36]
                                            )
                }

# We want to create labels for the training data based on the time to event data
# Labels in validation set are denoted by 0, 1, 2, 3, 4 where they are related to readouts within a time window of: (more than 48), (48 to 24), (24 to 12), (12 to 6), and (6 to 0) time_step before the failure, respectively. 
# If we don't have a failure reported, and the time_step left is less 48 we don't know when the failure will happen, so we will label it as -1. 

def get_class_label_int(row):
    #classes denoted by 0, 1, 2, 3, 4 where they are related to readouts within a time window of: (more than 48), (48 to 24), (24 to 12), (12 to 6), and (6 to 0) time_step before the failure, respectively
    if row['time_to_potential_event'] > 48:
        return 0 #No failure within 48 time steps
    elif row['time_to_potential_event'] > 24 and row['in_study_repair'] == 1:
        return 1 #Failure within 48 to 24 time steps
    elif row['time_to_potential_event'] > 12 and row['in_study_repair'] == 1:
        return 2 #Failure within 24 to 12 time steps
    elif row['time_to_potential_event'] > 6 and row['in_study_repair'] == 1:
        return 3 #Failure within 12 to 6 time steps
    elif row['time_to_potential_event'] > 0 and row['in_study_repair'] == 1:
        return 4 #Failure within 6 to 0 time steps
    else:
        return 5 #No failure reported, but within 48 time steps from the end of the study, don't know if it will fail or not

def get_class_label_OH(iclass, n_classes=6):
    tmp = torch.zeros(n_classes, dtype=torch.float)
    if iclass<n_classes:
        tmp[iclass] = 1
    return tmp.reshape(1,-1)

def get_class_label_multilabel(row):
    if row['time_to_potential_event'] > 48:
        tmp[0] = 1

    if row['in_study_repair'] == 1: # Failure reported
        if row['time_to_potential_event'] <= 48: #In less than 48 timesteps
            tmp[1] = 1
        if row['time_to_potential_event'] <= 24: #In less than 24 timesteps
            tmp[2] = 1
        if row['time_to_potential_event'] <= 12: #In less than 12 timesteps
            tmp[3] = 1
        if row['time_to_potential_event'] <= 6: #In less than 6 timesteps
            tmp[4] = 1
    #
    elif row['time_to_potential_event'] <= 48: # Might fail after end of study.
        tmp[5] = 1
    #
    return tmp.reshape(1,-1)

    #classes denoted by 0, 1, 2, 3, 4 where they are related to readouts within a time window of: (more than 48), (48 to 24), (24 to 12), (12 to 6), and (6 to 0) time_step before the failure, respectively
    
def add_class_labels(tte, readouts):
    # Join the readouts and the time to event data
    df = pd.merge(readouts, tte, on = 'vehicle_id', how='left').copy()
    # Calculate the time to a failure event
    df['time_to_potential_event'] = df['length_of_study_time_step'] - df['time_step']
    #df['class_label_multilabel'] = df.apply(get_class_label_multilabel, axis=1)
    df['class_label_int'] = df.apply(get_class_label_int, axis=1)
    df['class_label_OH'] = df['class_label_int'].apply(get_class_label_OH)
    return df

def remove_missing_rows(dd):
    #drop_idx = (dd.isna().sum(1).values == dd.shape[1])
    dd = dd.dropna(how="all",axis=0)    #dd = dd[keep_idx]
    assert not (dd.isna().any().any())
    return dd

#No labels for test data yet
def dataframe2X(dd, append_diff=True):
    dd = remove_missing_rows(dd)
    t = dd.index.values
    X = dd.values
    if X.shape[0] > 0:
        #l = [X]
        #l = []
        #dx = np.diff(X, axis=0, prepend=X[0,0])
        #assert (dx[0,0]==0)

        #dt = np.diff(t, prepend=0).reshape(-1,1)
        #dt[0,0] = 1
        #dt2 = np.diff(t, prepend=t[0]).reshape(-1,1)

        #l += [X]
        l = [X, t.reshape(-1,1)]
        X = np.concat(l, axis=1)
        
    else:
        X = np.zeros((1, X.shape[1]+1))#+ (1+append_diff)))

    return torch.from_numpy(X).to(torch.float)#,t)

def append_dummy_timeline(dd, append_diff=True):
    if append_diff:
        out = torch.from_numpy(np.concat([dd.values, np.array([[0]]), np.array([[0]])],axis=1)).to(torch.float)
    else:
        out = torch.from_numpy(np.concat([dd.values,np.array([[0]])], axis=1)).to(torch.float)
    return out

def get_vid_data(v_id, dd, specs):
    numerics = {num_varname[0]: dataframe2X(dd.copy()[num_varname])  for num_varname in num_variables}
    categorical = {cat_varname:    dataframe2X(dd.copy()[cat_varnames]) for cat_varname,cat_varnames in cat_variables.items()}
    specs = {"specs": append_dummy_timeline(specs.copy()) }
    return {**numerics, **categorical, **specs}

def readouts2dict(readouts, tte, specs, root_dir=".",labels=None,dataset="training"):
    metadata =      ['vehicle_id', 'time_step']
    targets =       ["in_study_repair", "time_to_potential_event", "class_label"]

    specs.set_index("vehicle_id",inplace=True)
    if dataset=="validation":
        assert (tte is None)
        assert(not (labels is None))
        labels_dict = labels.set_index("vehicle_id")["class_label"].to_dict()
        df = readouts

    elif dataset == "training":
        df = add_class_labels(tte, readouts)
        #df = df[df["class_label_int"]!= 5]
        df.drop(columns=["length_of_study_time_step"], inplace=True)
    
    elif dataset == "testing":
        df = readouts

    the_dict = {
        v_id: g.drop(columns="vehicle_id").set_index("time_step")
        for v_id, g in df.groupby("vehicle_id")
    }
    the_dict2 = {v_id:
                    {"data": get_vid_data(v_id,dd,specs[specs.index == v_id])
                    }
                for v_id, dd in the_dict.items()}
    
    if dataset == "training":
        for v_id, dd in the_dict.items():
            the_dict2[v_id]["data"]["reference"] = torch.from_numpy(dd["class_label_int"].index.values).to(torch.float)
            the_dict2[v_id]["targets_OH"]           = torch.cat(dd["class_label_OH"].values.tolist())
            the_dict2[v_id]["targets_int"]          = torch.from_numpy(dd["class_label_int"].values).to(torch.float)
    
    elif dataset == "validation":
        for v_id, dd in the_dict.items():
            the_dict2[v_id]["data"]["reference"] = torch.from_numpy(dd.index.values[[-1]]).to(torch.float)
            the_dict2[v_id]["targets_int"]          = torch.from_numpy(np.array([labels_dict[v_id]])).to(torch.float)

    elif dataset == "testing":
        for v_id, dd in the_dict.items():
            the_dict2[v_id]["data"]["reference"] = torch.from_numpy(dd.index.values[[-1]]).to(torch.float)
    return the_dict2

def get_data(DPATH):
    root_dir = DPATH#"."
    #Read the raw data
    # Train data

    fname = os.path.join(root_dir, "data.pklz")
    if not os.path.exists(fname):
        tteTrain = pd.read_csv(os.path.join(root_dir, 'train_tte.csv'))
        specificationsTrain = pd.read_csv(os.path.join(root_dir, 'train_specifications.csv'))

        # Read data
        readoutsTrain = pd.read_csv(os.path.join(root_dir, 'train_operational_readouts.csv'))
        readoutsValidation = pd.read_csv(os.path.join(root_dir, 'validation_operational_readouts.csv'))
        readoutsTest = pd.read_csv(os.path.join(root_dir, 'test_operational_readouts.csv'))

        # Normalization stats
        mu = 0;#readoutsTrain.set_index(['vehicle_id', 'time_step']).mean(0)
        sigma = 1#readoutsTrain.set_index(['vehicle_id', 'time_step']).std(0)

        # Validation data
        labelsValidation = pd.read_csv(os.path.join(root_dir, 'validation_labels.csv'))
        specificationsValidation = pd.read_csv(os.path.join(root_dir, 'validation_specifications.csv'))
        specificationsTest = pd.read_csv(os.path.join(root_dir, 'test_specifications.csv'))

        # Normalize
        readoutsTrain = ((readoutsTrain.set_index(['vehicle_id', 'time_step'])-mu)/sigma).reset_index()
        readoutsValidation = ((readoutsValidation.set_index(['vehicle_id', 'time_step'])-mu)/sigma).reset_index()
        readoutsTest = ((readoutsTest.set_index(['vehicle_id', 'time_step'])-mu)/sigma).reset_index()
        
        specificationsTrain["dataset"] = "train"
        specificationsValidation["dataset"] = "validation"
        specificationsTest["dataset"] = "test"
        
        specs = pd.concat([specificationsTrain, specificationsValidation, specificationsTest])
        specs_varnames = specs.set_index(["vehicle_id","dataset"]).columns
        all_specs_varnames = ["_".join([varname,s]) for varname in specs_varnames for s in specs[varname].unique().tolist()]
        all_specs_varnames = sorted(all_specs_varnames)

        specs = pd.get_dummies(specs.set_index(["vehicle_id","dataset"]))#.reset_index()
        specs = specs[all_specs_varnames]  ###.shape
        specs.reset_index(inplace=True)
        specificationsTrain,specificationsValidation,specificationsTest = [specs[specs["dataset"]==s] for s in ["train", 'validation', "test"]]
        specificationsTrain.drop(columns=["dataset"],inplace=True)
        specificationsValidation.drop(columns=["dataset"],inplace=True)
        specificationsTest.drop(columns=["dataset"],inplace=True)

        train_dict = readouts2dict(readoutsTrain,tteTrain,specificationsTrain,root_dir=root_dir, dataset="training")
        test_dict = readouts2dict(readoutsTest, None, specificationsTest, root_dir=root_dir, dataset="testing")
        val_dict = readouts2dict(readoutsValidation, None, specificationsValidation, root_dir=root_dir, labels=labelsValidation, dataset="validation")
        
        write_pklz(fname, [train_dict, val_dict, test_dict])
    else:
        train_dict, val_dict, test_dict = read_pklz(fname)

    return train_dict, val_dict, test_dict
    