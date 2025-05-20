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

        
    def get_class_weights(self):
        y = torch.cat([d["targets2"] for d in self.data.values()]).squeeze(-1).numpy()#.#tolist()
        n_classes = self.data[list(self.data.keys())[0]]["targets"].shape[1]
        self.class_weights = torch.from_numpy(compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y)).to(dtype=torch.float32) #/len(self.data.values())
    
    def __len__(self):
        return len(self.patids)

    def __getitem__(self, i):
        thedata = self.data[self.patids[i]]
        thedata["class_weights"] = self.class_weights  ###[thedata["targets"].int()]
        return thedata

#Includes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import os 
import torch

from utils_tbox.utils_tbox import read_pklz, write_pklz




# We want to create labels for the training data based on the time to event data
# Labels in validation set are denoted by 0, 1, 2, 3, 4 where they are related to readouts within a time window of: (more than 48), (48 to 24), (24 to 12), (12 to 6), and (6 to 0) time_step before the failure, respectively. 
# If we don't have a failure reported, and the time_step left is less 48 we don't know when the failure will happen, so we will label it as -1. 

def get_class_label2(row):
    
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
    
def get_class_label(row):
    tmp = torch.zeros(6, dtype=torch.float)
    
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
    
    elif row['time_to_potential_event'] <= 48: # Might fail after end of study.
        tmp[5] = 1

    return tmp.reshape(1,-1)

    #classes denoted by 0, 1, 2, 3, 4 where they are related to readouts within a time window of: (more than 48), (48 to 24), (24 to 12), (12 to 6), and (6 to 0) time_step before the failure, respectively
    #if row['time_to_potential_event'] > 48:
    #    return 0 #No failure within 48 time steps
    #elif row['time_to_potential_event'] > 24 and row['in_study_repair'] == 1:
    #    return 1 #Failure within 48 to 24 time steps
    #elif row['time_to_potential_event'] > 12 and row['in_study_repair'] == 1:
    #    return 2 #Failure within 24 to 12 time steps
    #elif row['time_to_potential_event'] > 6 and row['in_study_repair'] == 1:
    #    return 3 #Failure within 12 to 6 time steps
    #elif row['time_to_potential_event'] > 0 and row['in_study_repair'] == 1:
    #    return 4 #Failure within 6 to 0 time steps
    #else:
    #    return -1 #No failure reported, but within 48 time steps from the end of the study, don't know if it will fail or not
    
def add_class_labels(tte, readouts):
    # Join the readouts and the time to event data
    df = pd.merge(readouts, tte, on = 'vehicle_id', how='left').copy()
    # Calculate the time to a failure event
    df['time_to_potential_event'] = df['length_of_study_time_step'] - df['time_step']
    df['class_label'] = df.apply(get_class_label, axis=1)
    df['class_label2'] = df.apply(get_class_label2, axis=1)

    return df

def remove_missing_rows(dd):
    keep_idx = ~(dd.isna().sum(1).values == dd.shape[1])
    dd = dd[keep_idx]
    assert not (dd.isna().any().any())
    return dd

#No labels for test data yet
def dataframe2X(dd, append_diff=True):
    dd = remove_missing_rows(dd)
    t = dd.index.values
    X = dd.values
    if X.shape[0] > 0:
        l = [X]
        if append_diff:
            l+=[np.diff(t,prepend=t[0]).reshape(-1,1)]
        l += [t.reshape(-1,1)]
        X = np.concat(l, axis=1)
    else:
        X = np.zeros((1, X.shape[1]+ (1+append_diff)))
        #X[:] = np.nan
        #print("")
    return torch.from_numpy(X).to(torch.float)

def append_dummy_timeline(dd):
    return torch.from_numpy(np.concat([dd.values, np.array([[0]]), np.array([[0]])],axis=1)).to(torch.float)
    
def readouts2dict(readouts, tte, specs, root_dir=".",labels=None,dataset="training"):
    all_vehicles =  readouts["vehicle_id"].unique()
    metadata =      ['vehicle_id', 'time_step']
    targets =       ["in_study_repair", "time_to_potential_event", "class_label"]
    num_variables = [["171_0"], ["666_0"], ["427_0"], 
                        ["837_0"], ["309_0"], ["835_0"], 
                        ["370_0"], ["100_0"]
                    ]
    
    specs_varnames = specs.set_index("vehicle_id").columns
    all_specs_varnames = ["_".join([varname,s]) for varname in specs_varnames for s in specs[varname].unique().tolist()]
    all_specs_varnames = sorted(all_specs_varnames)

    specs = pd.get_dummies(specs.set_index("vehicle_id"))#.reset_index()
    specs = specs[all_specs_varnames]  ###.shape

    cat_variables = {varname: ["_".join([varname,str(i)]) for i in range(n_bins)] 
                        for varname, n_bins in zip(["167", "272", "291", "158", "459", "397"], 
                                                [10, 10, 11, 10, 20, 36]
                                                )
                    }

    if dataset=="validation":
        assert (tte is None)
        assert(not (labels is None))
        labels_dict = labels.set_index("vehicle_id")["class_label"].to_dict()
        df = readouts

    elif dataset == "training":
        df = add_class_labels(tte, readouts)
        df.drop(columns=["length_of_study_time_step"], inplace=True)

    elif dataset == "testing":
        df = readouts

        #raise Exception("NYI")

    the_dict = {v_id:
                    df[df["vehicle_id"]==v_id].drop(columns=["vehicle_id"]).set_index("time_step")
                    for v_id in all_vehicles
                    }
    
    the_dict2 = {v_id:
                    {"data":{
                        **{num_varname[0]: dataframe2X(dd[num_varname])  for num_varname in num_variables},
                        **{cat_varname:    dataframe2X(dd[cat_varnames]) for cat_varname,cat_varnames in cat_variables.items()},
                        **{"specs":        append_dummy_timeline(specs[specs.index == v_id])}
                        },
                    }
                for v_id, dd in the_dict.items()}
    
    if dataset == "training":
        for v_id, dd in the_dict.items():
            the_dict2[v_id]["data"]["reference"] = torch.from_numpy(dd["class_label"].index.values).to(torch.float)
            the_dict2[v_id]["targets"]           = torch.cat(dd["class_label"].values.tolist())
            the_dict2[v_id]["targets2"]          = torch.from_numpy(dd["class_label2"].values).to(torch.float)
    
    elif dataset == "validation":
        for v_id, dd in the_dict.items():
            the_dict2[v_id]["data"]["reference"] = torch.from_numpy(dd.index.values[[-1]]).to(torch.float)
            the_dict2[v_id]["targets2"]          = torch.from_numpy(np.array([labels_dict[v_id]])).to(torch.float)
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
        readoutsTrain = pd.read_csv(os.path.join(root_dir, 'train_operational_readouts.csv'))

        mu = readoutsTrain.set_index(['vehicle_id', 'time_step']).mean(0)
        sigma = readoutsTrain.set_index(['vehicle_id', 'time_step']).std(0)
        readoutsTrain = ((readoutsTrain.set_index(['vehicle_id', 'time_step'])-mu)/sigma).reset_index()

        # Validation data
        labelsValidation = pd.read_csv(os.path.join(root_dir, 'validation_labels.csv'))
        specificationsValidation = pd.read_csv(os.path.join(root_dir, 'validation_specifications.csv'))
        readoutsValidation = pd.read_csv(os.path.join(root_dir, 'validation_operational_readouts.csv'))
        readoutsValidation = ((readoutsValidation.set_index(['vehicle_id', 'time_step'])-mu)/sigma).reset_index()

        # Test data
        specificationsTest = pd.read_csv(os.path.join(root_dir, 'test_specifications.csv'))
        readoutsTest = pd.read_csv(os.path.join(root_dir, 'test_operational_readouts.csv'))
        readoutsTest = ((readoutsTest.set_index(['vehicle_id', 'time_step'])-mu)/sigma).reset_index()

        test_dict = readouts2dict(readoutsTest, None, specificationsTest, root_dir=root_dir, dataset="testing")
        
        val_dict = readouts2dict(readoutsValidation, None, specificationsValidation, root_dir=root_dir, labels=labelsValidation, dataset="validation")

        train_dict = readouts2dict(readoutsTrain,tteTrain,specificationsTrain,root_dir=root_dir, dataset="training")

        write_pklz(fname, [train_dict, val_dict, test_dict])
    else:
        train_dict,val_dict,test_dict = read_pklz(fname)

    return train_dict,val_dict,test_dict
    