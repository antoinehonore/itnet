import pandas as pd
import torch
import numpy as np 

from src.itnet import TSdata
from src.itgpt import ITGPT, Predictor

import torch.nn.functional as F
import torch.nn as nn
from utils_tbox.utils_tbox import read_pklz, write_pklz
from src.compx.mydata import TheDataset, get_data
from src.ITGPT.trainer import lTrainer
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
from datetime import datetime
from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut,LeavePGroupsOut, GroupKFold, KFold,TimeSeriesSplit
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger

import lightning as L
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from sklearn.preprocessing import LabelEncoder

def get_modality_dimensions(data_dimensions, model_params):
    modalities_dimension = {}
    modalities_dimension["q"] = 1

    if model_params["qk_type"]=="time": # Only the time column will be used to compute the attention weights
        modalities_dimension["k"] = {m: 1   for m in data_dimensions.keys()}
        # The model removes the time information from the data when computing the values, because the time info is used only to compute the attention weights 
        modalities_dimension["v"] = {m: d - 2 for m,d in data_dimensions.items()}

    elif "data" in model_params["qk_type"]: # Attention weight computed from the whole data
        # d_in_q, d_in_kv, d_qk, d_out
        modalities_dimension = {mname: dict(in_q=1, in_kv=d_in, out_qk=model_params["d_qk"], out_v=model_params["d_out"]) for mname,d_in in data_dimensions.items()}
    return modalities_dimension

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

def get_tr_val_index_lists(dataset, k=5):
    patids = np.array(dataset.patids)
    if k>0:
        tr_val_index_lists = KFold(k).split(patids)
    else:
        tr_val_index_lists = [[np.arange(len(patids)),np.zeros(0)]]

    return tr_val_index_lists

def init_tau(data):
    """A value for $\\tau$ in the attention kernels"""
    example_patient_id = list(data.keys())[0]
    return torch.diff(data[example_patient_id]["timelines"]["Back Door"]).max().item()*5

def get_profiler(profiler):
    if not (profiler is None): 
        #num_training_steps = 20
        from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
        if profiler=="simple":
            profiler = SimpleProfiler(filename="{}_profiler_results.txt".format(profiler))

        elif profiler=="advanced":
            profiler = AdvancedProfiler(filename="{}_profiler_results.txt".format(profiler))
    return profiler#, num_training_steps

def main(args):
    cfg_fname = args.i
    output_fname = args.o
    os.makedirs(output_fname.replace(".pklz",""),exist_ok=True)
    exp_name_ = os.path.join(os.path.basename(os.path.dirname(cfg_fname)), 
                            os.path.basename(cfg_fname).replace(".json",""))
    log_dir = "lightning_logs"
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
    
    data_dimensions = {}
    import warnings, socket
    warnings.filterwarnings("ignore")

    # loading dataset
    # Please change the path with the path of your dataset
    DPATH = 'data/compx/'
    if  (not (args.small)):# and (os.path.exists(DPATH + "/datasmall.pklz"))):
        data,valdata,testdata = get_data(DPATH)
        dataset = TheDataset(data)
        dataset.get_class_weights()
        class_weights = dataset.class_weights
        datasmall = {k: data[k] for k in list(data.keys())[-len(data.keys())//20:]}

        write_pklz("data/compx/datasmall.pklz",[class_weights, datasmall, valdata, testdata])
    else:
        class_weights, data, valdata, testdata = read_pklz(DPATH + "/datasmall.pklz")

    dataset = TheDataset(data)
    dataset.class_weights = class_weights
    
    batch_size = 1

    model_params["init_tau"] = 1  ###  init_tau(data)
    a_patid = list(data.keys())[0]
    data_dimensions = {m: data[a_patid]["data"][m].shape[1] for m in data[a_patid]["data"].keys() if m != "reference"}
    model_params["modalities_dimension"] = get_modality_dimensions(data_dimensions, model_params)

    tr_val_index_lists = get_tr_val_index_lists(dataset, k=hparams["training"]["kfold"])
    all_fold_results = []
    test_set =  TheDataset(testdata)
    test_set.class_weights = class_weights
    test_dataloader = DataLoader(test_set, batch_size=hparams["data"]["batch_size"], shuffle=False)

    val_set =  TheDataset(valdata)
    val_set.class_weights = class_weights
    val_dataloader =   DataLoader(val_set, batch_size=hparams["data"]["batch_size"], shuffle=False)

    for fold_idx, (fold_train_index, fold_val_index) in enumerate(tr_val_index_lists): ###  enumerate(GroupKFold(n_splits=5).split(dataset, groups=groups)):
        training_set = Subset(dataset, fold_train_index)
        val_set_internal = Subset(dataset, fold_val_index)
        
        train_dataloader = DataLoader(training_set, batch_size=hparams["data"]["batch_size"], shuffle=True, num_workers=args.j)
        val_internal_dataloader = DataLoader(val_set_internal, batch_size=hparams["data"]["batch_size"], shuffle=False)
        exp_name = exp_name_ + "/fold{}".format(fold_idx)

        logger = TensorBoardLogger(log_dir, name=exp_name, default_hp_metric=False)
        os.makedirs(os.path.dirname(logger.log_dir), exist_ok=True)
        model = Predictor(hparams["model"])
        if args.compile:
            model = torch.compile(model)

        ltrainer = lTrainer(model=model, hparams=hparams)
        
        log_every_n_steps = len(train_dataloader)//100
        check_val_every_n_epoch = 1
        profiler = get_profiler(args.profiler)
        limit_train_batches = None if not args.small else 10
        limit_test_batches = limit_train_batches
        limit_val_batches = limit_train_batches

        if not (profiler is None):
            n_epochs = 9
            check_val_every_n_epoch = 10
            log_every_n_steps = 2
            limit_train_batches = 1000
            limit_test_batches = 10
            limit_val_batches = 10
        
        extra_dtraining_kwargs = {"precision": args.precision, 
                                  "use_distributed_sampler":False,
                                  "num_sanity_val_steps":0}
        
        limits = dict(limit_test_batches=limit_test_batches, limit_train_batches=limit_train_batches,limit_val_batches=limit_val_batches)

        trainer = L.Trainer(max_epochs=n_epochs, logger=logger, log_every_n_steps=log_every_n_steps, 
                            check_val_every_n_epoch=check_val_every_n_epoch,
                            enable_progress_bar=args.v>1,
                            enable_checkpointing=False, profiler=profiler,
                            **extra_dtraining_kwargs, **limits)
        
        trainer.fit(ltrainer, train_dataloaders=train_dataloader, val_dataloaders=[val_internal_dataloader,val_dataloader])

        last_checkpoint = os.path.join(logger.log_dir, "checkpoints", "last.ckpt")
        trainer.save_checkpoint(last_checkpoint)
        
        outputfname = os.path.join(log_dir, exp_name, "results.pklz.fold{}".format(fold_idx))
        
        results = {}
        results["train"] =  trainer.validate(ltrainer, dataloaders=[train_dataloader])
        results["val_internal"] =    trainer.test(ltrainer, dataloaders=[val_internal_dataloader])
        results["val"] = trainer.test(ltrainer, dataloaders=[val_dataloader])
        results["test"] = trainer.test(ltrainer, dataloaders=test_dataloader)
        
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
    parser.add_argument('--profiler', type=str, default=None, help="simple or advanced")
    parser.add_argument('--small', action="store_true", default=False, help="Run on all patients by default")
    parser.add_argument('--compile', action="store_true", default=False, help="Do not compile model by default")
    parser.add_argument('--precision', type=str, default="32", help="Float encoding")

    args = parser.parse_args()
    main(args)
    