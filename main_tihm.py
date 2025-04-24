import pandas as pd
import torch
import numpy as np 

from src.fusionattn import FusionAttn
import torch.nn.functional as F
import torch.nn as nn
from utils_tbox.utils_tbox import read_pklz, write_pklz
from src.tihm.mydata import TheDataset
from src.tihm.trainer import lTrainer

import os 

import socket
import json
import argparse
import matplotlib
import socket
if socket.gethostname()=="cmm0958":
    matplotlib.use('tkagg') 
else:
    matplotlib.use('agg')

import matplotlib.pyplot as plt
from torch.utils.data import Subset
from src.tihm.data_loader import TIHMDataset
from datetime import datetime
from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut,LeavePGroupsOut, GroupKFold, KFold,TimeSeriesSplit
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger

import lightning as L
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from sklearn.preprocessing import LabelEncoder


groups_stems = ["heart_rate", "respiratory_rate", "Body Temperature", 
                "Body weight", "Diastolic blood pressure","Heart rate",
                "O/E - muscle mass", "Skin Temperature", "Systolic blood pressure","Total body water"]



def get_modalities(df, timeline, target, groups, ref_date=None):
    """Group data variables according to "groups"""
    DTYPE = torch.float32
    assert(timeline.shape[0] == df.shape[0])
    if ref_date is None:
        ref_date = datetime(year=2019, month=1, day=1)
    calX = {}
    idx = {}
    timelines = {}
    i_targets = np.isnan(target).sum(1)==0
    targets = torch.from_numpy(target[i_targets]).to(dtype=DTYPE)
    inference_timeline = torch.from_numpy((pd.to_datetime(timeline[i_targets])-ref_date).total_seconds().values/3600/24).to(dtype=DTYPE)
    
    for modality_name, vv in groups.items():
        theidx = (df[vv].isna().sum(1) == 0).values
        idx[modality_name]        = torch.from_numpy(theidx).to(dtype=DTYPE)
        calX[modality_name]       = torch.from_numpy(df[vv][theidx].values).to(dtype=DTYPE)
        timelines[modality_name]  = torch.from_numpy((pd.to_datetime(timeline[theidx])-ref_date).total_seconds().values/3600/24).to(dtype=DTYPE)
        if timelines[modality_name].shape[0]>0:
            diff =  torch.diff(timelines[modality_name], prepend=torch.tensor([timelines[modality_name][0].item()])).to(dtype=DTYPE)
            calX[modality_name] = torch.cat([calX[modality_name], diff.unsqueeze(-1), timelines[modality_name].unsqueeze(-1)],axis=1).to(dtype=DTYPE)
        else:# No data at all ?
            calX[modality_name] = torch.zeros((1,len(groups[modality_name])+2))
            timelines[modality_name]= torch.zeros((1))+inference_timeline[0]

    return {"timelines": timelines, "calX":calX, "idx":idx,
            "inference_timeline":inference_timeline,"targets":targets}

def to_dict(train_dataset,test_dataset):
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

def get_data(patients,groups,columns, ref_date=None):
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
        target = v["target"][:, [1]] >= 1
        df = pd.DataFrame(data=thedata, columns=columns)
        data[patid] = get_modalities(df, timeline, target, groups,ref_date=ref_date)
    a_patid = list(data.keys())[0]
    data_dimensions = {m: data[a_patid]["calX"][m].shape[1] for m in data[a_patid]["calX"].keys()}

    return data, data_dimensions, ref_date

class Predictor(torch.nn.Module):
    def __init__(self, hparams):
        super(Predictor, self).__init__()
        self.hparams = hparams
        self.fusion_model = FusionAttn(hparams)
    
    def forward(self, batch):
        #thefeatures = self.feature_extractor(batch)
        #thefeatures["reference"] = batch["inference_timeline"].unsqueeze(-1)
        thefeatures = {}
        thefeatures["reference"] = batch["inference_timeline"].T.unsqueeze(0).unsqueeze(0)
        thefeatures = {**thefeatures,**{m: v.unsqueeze(1) for m,v in batch["calX"].items()}}
        yhat = self.fusion_model(thefeatures)
        #yhat = torch.nn.functional.sigmoid(yhat)
        return yhat

def get_modality_dimensions(data_dimensions, model_params):
    modalities_dimension = {}
    modalities_dimension["q"] = 1

    if model_params["qk_type"]=="time": # Only the time column will be used to compute the attention weights
        modalities_dimension["k"] = {m: 1   for m in data_dimensions.keys()}
        # The model removes the time information from the data when computing the values, because the time info is used only to compute the attention weights 
        modalities_dimension["v"] = {m: d - 2 for m,d in data_dimensions.items()}

    elif "data" in model_params["qk_type"]: # Attention weight computed from the whole data
        modalities_dimension["k"] = {m: d for m, d in data_dimensions.items()}
        modalities_dimension["v"] = {m: d for m, d in data_dimensions.items()}
    return modalities_dimension

def get_variable_groups(train_dataset):
        
    variable_groups = {k: [v for v in train_dataset.feature_names if k in v] for k in groups_stems}
    others = {v:[v] for v in train_dataset.feature_names if not v in list(set( sum(list(variable_groups.values()),[]) ))}
    if len(others) > 0:
        variable_groups = {**variable_groups, **others}
    return variable_groups


def patient_timesplit(patid, d, n_splits=5):
    istart = max([0, d["inference_timeline"].shape[0]-7*5])
    ### indices = {patid+"_{}".format(i+1): indexes for i,indexes in enumerate(TimeSeriesSplit(n_splits).split(d["inference_timeline"]))}
    
    indices = {patid+"_{}".format(i+1): (np.concatenate([np.arange(istart), istart+indexes[0]]), istart+indexes[1]) for i,indexes in enumerate(TimeSeriesSplit(n_splits).split(d["inference_timeline"][istart:]))}
    out_tr = {k:  dict(d) for k in indices.keys()}
    out_val = {k:  dict(d) for k in indices.keys()}
    for k in indices.keys():
        out_tr[k]["inference_timeline"] = out_tr[k]["inference_timeline"][indices[k][0]]
        out_tr[k]["targets"] = out_tr[k]["targets"][indices[k][0]]
        out_val[k]["inference_timeline"] = out_val[k]["inference_timeline"][indices[k][1]]#tr_inference_timeline[i]
        out_val[k]["targets"] = out_val[k]["targets"][indices[k][1]]

    return out_tr, out_val


def get_tr_val_index_lists(data):
    all_training_data = {}
    all_validation_data = {}
    for patid, d in data.items():
        if d["inference_timeline"].shape[0]>5:
            patient_training, patient_validation = patient_timesplit(patid, d)
            all_training_data = {**all_training_data,**patient_training}
            all_validation_data = {**all_validation_data,**patient_validation}

    training_dataset = TheDataset(all_training_data)
    validation_dataset = TheDataset(all_validation_data)

    tr_fold_indices = [[i for i,k in enumerate(all_training_data.keys()) if k.endswith(str(ifold)) ] for ifold in range(1,6)]
    val_fold_indices = [[i for i,k in enumerate(all_validation_data.keys()) if k.endswith(str(ifold)) ] for ifold in range(1,6)]
    return training_dataset, validation_dataset, zip(tr_fold_indices, val_fold_indices)

def init_tau(data):
    """A value for $\\tau$ in the attention kernels"""
    example_patient_id = list(data.keys())[0]
    return torch.diff(data[example_patient_id]["timelines"]["Back Door"]).max().item()*5

def main(args):
    cfg_fname = args.i
    output_fname = args.o
    os.makedirs(output_fname.replace(".pklz",""),exist_ok=True)
    exp_name_ = os.path.join(os.path.basename(os.path.dirname(cfg_fname)), 
                            os.path.basename(cfg_fname).replace(".json",""))
    exp_name = exp_name_
    torch.set_num_threads(4)
    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    torch.set_float32_matmul_precision('medium')
    plot = args.plot
    save = args.save
    show = args.show
    debug = False

    with open(cfg_fname,"r") as fp:
        hparams = json.load(fp)["params"]
    
    n_epochs = hparams["training"]["n_epochs"]
    
    model_params = hparams["model"]


    # loading dataset
    # Please change the path with the path of your dataset
    DPATH = 'data/tihm/Dataset/'
    TEST_START = "2019-06-23"
    n_days = 0
    batch_size = 1
    impute = {"imputer": None}

    if hparams["data"]["impute"] == "linear":
        impute = {}

    train_dataset = TIHMDataset(
        root=DPATH, train=True, normalise="global", n_days=n_days, **impute, TEST_START=TEST_START#"2019-06-23"
    )

    test_dataset = TIHMDataset(
        root=DPATH, train=False, normalise="global", n_days=n_days,**impute, TEST_START=TEST_START#"2019-06-23"
    )
    
    train_patients, test_patients = to_dict(train_dataset, test_dataset)

    variable_groups = get_variable_groups(train_dataset)

    data, data_dimensions, ref_date = get_data(train_patients, variable_groups, train_dataset.feature_names)
    test_data, _, _ = get_data(test_patients, variable_groups, test_dataset.feature_names, ref_date=ref_date)

    model_params["modalities_dimension"] = get_modality_dimensions(data_dimensions, model_params)

    dataset = TheDataset(data)
    
    test_dataset = TheDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams["data"]["batch_size"], shuffle=False)

    model_params["init_tau"] = init_tau(data)
    
    training_dataset, validation_dataset, tr_val_index_lists = get_tr_val_index_lists(dataset.data)

    all_fold_results = []

    for fold_idx, (fold_train_index, fold_val_index) in enumerate(tr_val_index_lists): #enumerate(GroupKFold(n_splits=5).split(dataset, groups=groups)):
        training_set = Subset(training_dataset, fold_train_index)
        val_set = Subset(validation_dataset, fold_val_index)
        
        train_dataloader = DataLoader(training_set, batch_size=hparams["data"]["batch_size"], shuffle=True)
        val_dataloader =   DataLoader(val_set, batch_size=hparams["data"]["batch_size"], shuffle=False)
        
        log_dir = "lightning_logs"

        logger = TensorBoardLogger(log_dir, name=exp_name, default_hp_metric=False)
        os.makedirs(os.path.dirname(logger.log_dir), exist_ok=True)

        ltrainer = lTrainer(model=Predictor(hparams["model"]), hparams=hparams)
        
        log_every_n_steps = len(train_dataloader)
        check_val_every_n_epoch = 1
        trainer = L.Trainer(max_epochs=n_epochs, logger=logger, log_every_n_steps=log_every_n_steps, # max_steps=len(training_set)*n_epochs,
                            check_val_every_n_epoch=check_val_every_n_epoch,
                            enable_progress_bar=False, enable_checkpointing=False)
        
        trainer.fit(ltrainer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        last_checkpoint = os.path.join(logger.log_dir, "checkpoints", "last.ckpt")
        trainer.save_checkpoint(last_checkpoint)
        
        outputfname = os.path.join(log_dir, exp_name, "results.pklz.fold{}".format(fold_idx))
        
        results_train =  trainer.validate(ltrainer, dataloaders=train_dataloader)
        results_val =    trainer.validate(ltrainer, dataloaders=val_dataloader)
        #results_test =   trainer.validate(ltrainer, dataloaders=test_dataloader)
        results_test = []

        results = [results_train, results_val, results_test]

        results.append(last_checkpoint)
        
        write_pklz(outputfname, results)
        all_fold_results.append(results)
        
        if debug:
            break
    
    outputfname = os.path.join(log_dir, exp_name, "results.pklz")
    write_pklz(outputfname, all_fold_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",type=str,required=True,help="Input config file")
    parser.add_argument("-o",type=str,required=True,help="output results file")
    parser.add_argument("-v", help="Verbosity level", default=0, type=int)
    parser.add_argument("-j", help="Number of tasks", default=1, type=int)
    parser.add_argument("--plot", action="store_true", help="Plot figures", default=False)
    parser.add_argument("--save", action="store_true", help="save figures", default=False)
    parser.add_argument("--show", action="store_true", help="Show figures", default=False)
    parser.add_argument("--profile", action="store_true", help="profile epoch", default=False)

    args = parser.parse_args()
    main(args)
    