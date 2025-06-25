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
    vids = np.array(dataset.vids)
    if k>0:
        tr_val_index_lists = KFold(k, shuffle=True).split(vids)
    else:
        tr_val_index_lists = [[np.arange(len(vids)),np.zeros(0)]]

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

def inference_code(fname_ckpt,val_set):
    loaded = torch.load(fname_ckpt)
    hparams = loaded["hyper_parameters"]
    val_dataloader = DataLoader(val_set, batch_size=hparams["data"]["batch_size"], shuffle=False, num_workers=0)
    
    model = Predictor(hparams["model"])
    ltrainer=lTrainer(model=model, hparams=hparams)
    plt.close("all")
    ltrainer.load_state_dict(loaded["state_dict"])
    limit_test_batches = None#10
    L.Trainer(limit_test_batches=limit_test_batches).test(ltrainer, dataloaders=val_dataloader)
    results = ltrainer.test_scores
    yclass = torch.cat(results['yclass'])
    logits = torch.cat(results['logits'])
    fig, ax = plt.subplots()
    plot_confusion_matrix(ax, yclass, logits.argmax(-1), num_classes=logits.shape[1])

def check_model():
    if False:
        n1=2
        n2=2
        d=16
        Q = torch.randn(1,1,1+1,d)
        K = torch.randn(1,1,n1+n2,d)
        V = torch.randn(1,1,n1+n2,1)
        q_t = torch.tensor([[2, 2]])
        q_idx = torch.tensor([[0, 1]])
        kv_t = torch.tensor([[0,1,0,1]])
        kv_idx = torch.tensor([[0,0,1,1]])
        model.eval()
        f = model.itgpt.model[0].encodeMMA.uni_modal_attention["171_0"].causal_scaled_dot_product_attention
        y = f(Q, K, V, q_t, kv_t, q_idx=q_idx, kv_idx=kv_idx)

        y1 = f(Q[...,:1,:],K[...,:n1,:],V[...,:n1,:],q_t[:,:1], kv_t[:,:n1], q_idx=q_idx[:,:1], kv_idx=kv_idx[:,:n1])
        y2 = f(Q[...,1:,:],K[...,n1:,:],V[...,n1:,:],q_t[:,1:], kv_t[:,n1:], q_idx=q_idx[:,1:], kv_idx=kv_idx[:,n1:])
        print(y, y1, y2)

        batch1 = training_set[0]
        batch2 = training_set[1]
        _, out1 = model(my_collate([batch1]))
        _, out2 = model(my_collate([batch2]))
        _, out12 = model(my_collate([batch1, batch2]))
        print((out12[0, :out1.shape[1],:] - out1[0]).norm())
    
def my_collate(batch):
    # Separate data and labels
    all_modalities = list(batch[0]["data"].keys())
    data = {m: cat_modalities(m,[item['data'][m] for item in batch]) for m in all_modalities}

    #labels = cat_modalities("reference", [item['targets_int'] for item in batch])
    vids = torch.tensor([item['vid'] for item in batch])
    # Pad the sequences (assuming 'data' is a sequence)
    #padded_data = pad_sequence(data, batch_first=True) # Important: batch_first!

    # Convert labels to tensor (if needed)
    labels = torch.cat([item['targets_int'] for item in batch]).unsqueeze(0)

    return {'data': data, 'label': labels, "class_weights":batch[0]["class_weights"], "vid":vids}

def cat_modalities(m,batches):
    idx = torch.cat([torch.ones(t.shape[0])*i for i,t in enumerate(batches)])#.reshape(-1,1)
    data = torch.cat(batches)
    if m == "reference":
        data = data.reshape(-1,1)
    data = TSdata(data[...,:-1].unsqueeze(0).unsqueeze(0), data[...,-1].unsqueeze(0), idx)
    return data


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

        write_pklz("data/compx/datasmall.pklz", [class_weights, datasmall, valdata, testdata])
    else:
        class_weights, data, valdata, testdata = read_pklz(DPATH + "/datasmall.pklz")

    tr_vids = list(data.keys())
    val_vids = list(valdata.keys())

    assert(len(list(set(tr_vids+val_vids))) == (len(tr_vids)+len(val_vids)))
    data = {**data,**valdata}

    dataset = TheDataset(data)
    dataset.class_weights = class_weights
    
    model_params["init_tau"] = 1  ###  init_tau(data)
    a_patid = list(data.keys())[0]
    data_dimensions = {m: data[a_patid]["data"][m].shape[1]-1 for m in data[a_patid]["data"].keys() if m != "reference"}
    model_params["modalities_dimension"] = get_modality_dimensions(data_dimensions, model_params)
    
    loaders_kwargs = dict(num_workers=args.j, pin_memory=args.j>0, persistent_workers=args.j>0)

    tr_val_index_lists = get_tr_val_index_lists(dataset, k=hparams["training"]["kfold"])
    all_fold_results = []
    test_set =  TheDataset(testdata)
    test_set.class_weights = class_weights
    test_dataloader = DataLoader(test_set, batch_size=hparams["data"]["batch_size"], shuffle=False,**loaders_kwargs)

    outputfname = os.path.join(log_dir, exp_name_, "results.pklz")

    for fold_idx, (fold_train_index, fold_val_index) in enumerate(tr_val_index_lists): ###  enumerate(GroupKFold(n_splits=5).split(dataset, groups=groups)):
        training_set = Subset(dataset, fold_train_index)
        val_set_internal = Subset(dataset, fold_val_index)
        ###my_collate = None
        train_dataloader = DataLoader(training_set, batch_size=hparams["data"]["batch_size"], shuffle=True, collate_fn=my_collate,**loaders_kwargs)
        val_internal_dataloader = DataLoader(val_set_internal, batch_size=hparams["data"]["batch_size"], shuffle=False, collate_fn=my_collate, **loaders_kwargs)
        exp_name = exp_name_ + "/fold{}".format(fold_idx)

        logger = TensorBoardLogger(log_dir, name=exp_name, default_hp_metric=False)
        
        os.makedirs(os.path.dirname(logger.log_dir), exist_ok=True)
        model = Predictor(hparams["model"])
        
        def initialize_all_parameters_to_zero(model: nn.Module):
            for param in model.parameters():
                nn.init.normal_(param)
                #nn.init.xavier_uniform_(param)
                #nn.init.constant_(param, 0)

        initialize_all_parameters_to_zero(model)

        if args.compile:
            model = torch.compile(model)
        print(model)
        ltrainer = lTrainer(model=model, hparams=hparams)
        
        all_tr_vids = torch.tensor([batch["vid"] for batch in training_set])
        all_tr_vids = all_tr_vids[torch.randperm(all_tr_vids.shape[0])]
        n_labels = int(hparams["training"]["use_p_label"] * all_tr_vids.shape[0])
        ltrainer.use_labels_vids = all_tr_vids[:n_labels]

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
        accumulate_grad_batches =  max([1, hparams["training"]["grad_step_every"]//hparams["data"]["batch_size"]])
        trainer = L.Trainer(max_epochs=n_epochs, logger=logger if not args.fast else None, 
                            log_every_n_steps=log_every_n_steps  if not args.fast else None, 
                            check_val_every_n_epoch=check_val_every_n_epoch,
                            enable_progress_bar=args.v>1  if not args.fast else False,
                            enable_checkpointing=False, profiler=profiler, accumulate_grad_batches=accumulate_grad_batches,
                            **extra_dtraining_kwargs, **limits, barebones=args.fast)
        

        trainer.fit(ltrainer, train_dataloaders=train_dataloader, val_dataloaders=[val_internal_dataloader])

        last_checkpoint = os.path.join(logger.log_dir, "checkpoints", "last.ckpt")
        trainer.save_checkpoint(last_checkpoint)
        
        results = dict(fold_train_index=fold_train_index, fold_val_index=fold_val_index,last_checkpoint=last_checkpoint)
        trainer.test(ltrainer, dataloaders=val_internal_dataloader)
        results["yclass"] = torch.cat(ltrainer.test_scores['yclass']).cpu()
        results["logits"] = torch.cat(ltrainer.test_scores['logits']).cpu()
        #ltrainer.test_scores = []
        #ltrainer.model.itgpt.normalized_batch = None
        #results["ltrainer"] = ltrainer.cpu()
        #write_pklz(outputfname_fold, results)
        all_fold_results.append(results)
        
        if debug:
            break
        
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
    parser.add_argument('--fast', action="store_true", default=False, help="Barebones training, no validation")

    args = parser.parse_args()
    main(args)
    