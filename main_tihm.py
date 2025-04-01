import pandas as pd
import torch
import numpy as np 
import torchmetrics

from torchmetrics.classification import BinaryStatScores

from src.fusionattn import FusionAttn
import torch.nn.functional as F
import torch.nn as nn
from src.feature_extractor import FeatureExtractor
from utils_tbox.utils_tbox import read_pklz, write_pklz

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

from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut,LeavePGroupsOut, GroupKFold
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch.loggers import TensorBoardLogger


import lightning as L
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from torchmetrics import ConfusionMatrix
from sklearn.preprocessing import LabelEncoder

"""
Column 1: Force (N) in X dimension
Column 2: Force (N) in Y dimension
Column 3: Force (N) in Z dimension
Column 4: Vibration (g) in X dimension
Column 5: Vibration (g) in Y dimension
Column 6: Vibration (g) in Z dimension
Column 7: AE-RMS (V) 
"""

from sklearn.utils.class_weight import compute_class_weight

def plot_spectrogram(ax1, x, kernel_size):
    fs = 50000/kernel_size
    T_x = 1 / fs
    
    #x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) # the signal
    
    g_std = 8  # standard deviation for Gaussian window in samples
    
    w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    
    SFT = ShortTimeFFT(w, hop=10, fs=fs, mfft=1024, scale_to='magnitude')
    
    
    
    #for i,thename in enumerate(columns):
    N = x.shape[0]
    
    t_x = np.arange(N) * T_x  # time indexes for signal
    
    f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
    
    Sx = SFT.stft(x)  # perform the STFT
   
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    
    ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, " +
    
                  rf"$\sigma_t={g_std*SFT.T}\,$s)")
    
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
    
                   rf"$\Delta t = {SFT.delta_t:g}\,$s)",
    
            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
    
                   rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
    
            xlim=(t_lo, t_hi))
    
    
    im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                     extent=SFT.extent(N), cmap='viridis')
    
    ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
    
    #fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
    
    
    # Shade areas where window slices stick out to the side:
    
    for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
    
                     (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    
        ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    
    for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
    
        ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    
    ax1.legend()
    return ax1


def plot_confusion_matrix(ax, y_true, y_pred, class_names=None, normalize=False, cmap="Blues"):
    """
    Plots a confusion matrix using matplotlib.
    
    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.
        class_names (list, optional): List of class names.
        normalize (bool, optional): Whether to normalize the confusion matrix.
        cmap (str, optional): Color map for the heatmap.
    """
    # Convert labels to tensor if they are not
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Ensure class names are available
    if class_names is None:
        class_names=["0","1"]
    
    # Compute confusion matrix
    num_classes = len(class_names)
    confmat = ConfusionMatrix(task="binary", num_classes=num_classes)
    cm = confmat(y_pred, y_true).cpu().numpy()
    
    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot using matplotlib
    cax = ax.matshow(cm, cmap=cmap)

    # Set labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    
    # Annotate each cell with its value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}", ha='center', va='center', color='black')
    
    return ax

def get_modalities(df, timeline, target, groups, ref_date=None):
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
        calX[modality_name]       = {"data":torch.from_numpy(df[vv][theidx].values).to(dtype=DTYPE)}
        timelines[modality_name]  = torch.from_numpy((pd.to_datetime(timeline[theidx])-ref_date).total_seconds().values/3600/24).to(dtype=DTYPE)
        if timelines[modality_name].shape[0]>0:
            diff =  torch.diff(timelines[modality_name], prepend=torch.tensor([timelines[modality_name][0].item()])).to(dtype=DTYPE)
            calX[modality_name]["data"] = torch.cat([calX[modality_name]["data"], diff.unsqueeze(-1), timelines[modality_name].unsqueeze(-1)],axis=1).to(dtype=DTYPE)
        else:# No data at all ?
            calX[modality_name]["data"] = torch.zeros((1,len(groups[modality_name])+2))
            timelines[modality_name]= torch.zeros((1))+inference_timeline[0]

    return {"timelines": timelines, "calX":calX, "idx":idx,
            "inference_timeline":inference_timeline,"targets":targets}

def to_dict(train_dataset,test_dataset):
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

def get_data(patients,ref_date,groups,columns):
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
    return data

class Predictor(torch.nn.Module):
    def __init__(self, hparams):
        super(Predictor, self).__init__()
        self.hparams = hparams
        self.fusion_model = FusionAttn(hparams)
    
    def forward(self, batch):
        #thefeatures = self.feature_extractor(batch)
        #thefeatures["reference"] = batch["inference_timeline"].unsqueeze(-1)
        thefeatures = {}
        thefeatures["reference"] = {"data": batch["inference_timeline"].T.unsqueeze(0).unsqueeze(0)}
        thefeatures = {**thefeatures,**{m: {"data":v["data"].unsqueeze(1)} for m,v in batch["calX"].items()}}
        yhat = self.fusion_model(thefeatures)
        yhat = torch.nn.functional.sigmoid(yhat)
        return yhat

def get_mu_sigma(batches,theindexes):
    mu=0. 
    sigma=0.
    for i in theindexes:
        y = batches[i]["Y"]
        mu += y.mean(0)
        sigma += y.std(0)
    return mu/len(theindexes), sigma/len(theindexes)

def normalize(batches,theindexes,mu,sigma):
    for i in theindexes:
        batches[i]["Y"]=(batches[i]["Y"]-mu)/sigma

class TheDataset(Dataset):
    def __init__(self, data, device="cpu"):
        self.patids = list(data.keys())
        self.data = data#{cutter_no: {m:data[cutter_no]["calX"][m].to(device=device)for m in data[cutter_no]["calX"].keys()} for cutter_no in data.keys()}
        self.mu=None
        self.sigma=None
        self.device = device
        y = torch.cat([d["targets"] for d in self.data.values()]).squeeze(-1).numpy()#.#tolist()
        self.class_weights = torch.from_numpy(compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)).to(dtype=torch.float32) #/len(self.data.values())
    
    def __len__(self):
        return len(self.patids)

    def __getitem__(self, i):
        thedata = self.data[self.patids[i]]
        thedata["class_weights"] = self.class_weights#[thedata["targets"].int()]

        return thedata

class lTrainer(L.LightningModule):
    def __init__(self, model=None, hparams=None):
        super(lTrainer, self).__init__()
        self.model = model
        self.save_hyperparameters(hparams)

        self.val_scores = {"y":[],"yhat":[]}
        self.train_scores = {"y":[],"yhat":[]}
        self.loss_fun_name = hparams["training"]["loss"] 

        if self.loss_fun_name == "CE":
            self.loss_fun = torch.nn.functional.cross_entropy
        
        elif self.loss_fun_name == "MSE":
            self.loss_fun = torch.nn.functional.mse_loss

        self.train_recon_figure     = plt.subplots(figsize=(10,6))
        self.val_recon_figure       = plt.subplots(figsize=(10,6))
        self.val_senspec_figure     = plt.subplots(figsize=(5,3))
        self.train_senspec_figure   = plt.subplots(figsize=(5,3))

        self.spectrogram_figure     = [plt.subplots(figsize=(10,6)) for _ in range(4)]
        self.val_attn_matrix        = {k:plt.subplots(figsize=(10,6)) for k in model.fusion_model.estimate_fusion.attn_matrices.keys()}
        self.automatic_optimization = False
        self.the_training_step  = 0

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
        {"mse/val": torch.nan, "mse/train": torch.nan})

        self.val_scores = {"y":   [],   "yhat": []}
        self.train_scores = {"y": [],   "yhat": []}
    
    def compute_loss(self,batch):
        y = batch["targets"]
        yhat = self.model(batch)

        y_n = y.squeeze(0).long()
        self.train_scores["y"].append(y.squeeze(0))
        self.train_scores["yhat"].append(yhat.detach().squeeze(0))
        if self.loss_fun_name == "CE":
            yhat = torch.cat([1-yhat,yhat], axis=2).permute(1,2,0)#torch.cat([1-yhat,yhat], axis=2).transpose(0,2)
        sample_weights = batch["class_weights"][0][y_n]
        loss = (self.loss_fun(yhat.squeeze(0), y_n, reduction="none")*sample_weights).mean()#.squeeze(-1).T.long())
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        opt = self.optimizers()
        loss = 0.0
        self.the_training_step += 1
        log_dict = {}
        loss = self.compute_loss(batch)
        self.manual_backward(loss)

        if self.the_training_step % self.hparams["training"]["grad_step_every"]:
            opt.step()
            opt.zero_grad()
        self.log("{}/train".format(self.loss_fun_name), loss,on_epoch=True,batch_size=1, on_step=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y = batch["targets"]
        y_n = y

        yhat = self.model(batch)
        if batch_idx == 0 and (self.logger is not None):
            for modality_name,(fig,ax) in self.val_attn_matrix.items():
                A = self.model.fusion_model.estimate_fusion.attn_matrices[modality_name]
                if A is not None:
                    ax.cla()
                    ax.imshow(A[0,0].T, aspect="equal")
                    ax.set_title("{}: $\\tau={:.03f}$".format(modality_name, self.model.fusion_model.estimate_fusion.history_temperature[modality_name].item()))
                    ax.set_ylabel("Measurements timeline")
                    ax.invert_yaxis()
                    ax.set_xlabel("Predictions timeline")
                    self.logger.experiment.add_figure("attn_figure/{}/val".format(modality_name), fig, self.the_training_step)

        self.val_scores["y"].append(y.squeeze(0))
        self.val_scores["yhat"].append(yhat.detach().squeeze(0))

    def on_train_epoch_end(self):
        y = torch.cat(self.train_scores["y"]).squeeze(-1)
        yhat = torch.cat(self.train_scores["yhat"]).squeeze(-1)
        [tp, fp, tn, fn, sup] = torchmetrics.functional.classification.binary_stat_scores(yhat,y)
        f1score = torchmetrics.functional.f1_score(yhat, y, task="binary")
        sensitivity = tp/(tp+fn)
        specificity =  tn/(tn+fp)

        ax = self.train_senspec_figure[1]
        ax.cla()
        ax.bar([0, 1], [sensitivity,specificity], label=["Sensitivity", "Specificity"], color=["darkblue","darkred"], alpha=0.5)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.set_xlim([-2, 3])
        
        if self.logger is not None:
            self.logger.experiment.add_figure("senspec/train", self.train_senspec_figure[0], self.the_training_step)

        self.log_dict({"f1score/train": f1score,"sensitivity/train":sensitivity, "specificity/train":specificity},on_epoch=True,on_step=False,batch_size=1)
        self.train_scores = {"y": [], "yhat": []}

        i = 0
        ax = self.train_recon_figure[1]
        ax.cla()
        plot_confusion_matrix(ax, y, yhat)
        if self.logger is not None:
            self.logger.experiment.add_figure("recon_figure/train", self.train_recon_figure[0], self.the_training_step)

    def on_validation_epoch_end(self):
        y = torch.cat(self.val_scores["y"]).squeeze(-1)
        yhat = torch.cat(self.val_scores["yhat"]).squeeze(-1)
        
        [tp, fp, tn, fn, sup] = torchmetrics.functional.classification.binary_stat_scores(yhat,y)
        f1score = torchmetrics.functional.f1_score(yhat,y,task="binary")
        sensitivity = tp/(tp+fn)
        specificity =  tn/(tn+fp)

        ax = self.val_senspec_figure[1]
        ax.cla()
        ax.bar([0,1],[sensitivity,specificity], label=["Sensitivity", "Specificity"], color=["darkblue","darkred"],alpha=0.5)
        ax.legend()
        ax.set_ylim([0,1])
        ax.set_xlim([-2,3])

        if self.logger is not None:
            self.logger.experiment.add_figure("senspec/val", self.val_senspec_figure[0], self.the_training_step)
            self.log_dict({"f1score/val": f1score, "sensitivity/val":sensitivity, "specificity/val":specificity},on_epoch=True,on_step=False,batch_size=1)#, "spec/val":specificity, "sen/val":sensitivity})#, "mse/val": loss_val})
        
        self.val_scores = {"y": [], "yhat": []}

        #i=0
        #ax = self.val_recon_figure[1]
        #ax.cla()
        #plot_confusion_matrix(ax,y,yhat)
        #if self.logger is not None:
        #    self.logger.experiment.add_figure("recon_figure/val", self.val_recon_figure[0], self.the_training_step)

    def configure_optimizers(self):
        optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], 
                lr=self.hparams["training"]['lr'])
        return optim

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
    np.random.seed(2*seed)
    
    torch.set_float32_matmul_precision('medium')
    plot = args.plot
    save = args.save
    show = args.show
    debug = False

    #torch.autograd.set_detect_anomaly(True)

    with open(cfg_fname,"r") as fp:
        hparams = json.load(fp)["params"]
    
    lr = "linear"   ###=1e-3
    n_epochs = hparams["training"]["n_epochs"]

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
        root=DPATH, train=True, normalise="global", n_days=n_days,**impute, TEST_START=TEST_START#"2019-06-23"
    )

    test_dataset = TIHMDataset(
        root=DPATH, train=False, normalise="global", n_days=n_days,**impute, TEST_START=TEST_START#"2019-06-23"
    )
    
    train_patients, test_patients = to_dict(train_dataset,test_dataset)
    groups_stems = ["heart_rate", "respiratory_rate", "Body Temperature", 
                    "Body weight", "Diastolic blood pressure","Heart rate",
                    "O/E - muscle mass", "Skin Temperature", "Systolic blood pressure","Total body water"]

    variable_groups = {k: [v for v in train_dataset.feature_names if k in v] for k in groups_stems}
    others = {v:[v] for v in train_dataset.feature_names if not v in list(set( sum(list(variable_groups.values()),[]) ))}
    if len(others) > 0:
        variable_groups = {**variable_groups, **others}
    
    ref_date = min([min(v["timeline"]) for v in train_patients.values()])
    ref_date = datetime(year=ref_date.year,month=ref_date.month,day=ref_date.day)
    
    data = get_data(train_patients, ref_date, variable_groups, train_dataset.feature_names)
    test_data = get_data(test_patients, ref_date, variable_groups, test_dataset.feature_names)

    a_patid = list(data.keys())[0]
    
    model_params = hparams["model"]
    data_dimensions = {m:   data[a_patid]["calX"][m]["data"].shape[1] for m in data[a_patid]["calX"].keys()}

    modalities_dimension = {}
    modalities_dimension["q"] = 1

    if model_params["qk_type"]=="time": # Only the time column will be used to compute the attention weights
        modalities_dimension["k"] = {m: 1   for m in data_dimensions.keys()}
        # The model removes the time information from the data when computing the values, because the time info is used only to compute the attention weights 
        modalities_dimension["v"] = {m: d - 2 for m,d in data_dimensions.items()}

    elif "data" in model_params["qk_type"]: # Attention weight computed from the whole data
        modalities_dimension["k"] = {m: d for m, d in data_dimensions.items()}
        modalities_dimension["v"] = {m: d for m, d in data_dimensions.items()}

    dataset = TheDataset(data)
    
    test_dataset = TheDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams["data"]["batch_size"], shuffle=False)

    groups = dataset.patids
    all_fold_results = []

    model_params["modalities_dimension"] = modalities_dimension
    #{"reference":1, **{m: data_dimensions[m] if model_params["qk_type"]=="data" else data_dimensions[m]-2 for m in  data_dimensions.keys()}}
    
    model_params["init_tau"] = torch.diff(data[a_patid]["timelines"]["Back Door"]).max().item()*5
    
    for fold_idx, (train_index, val_index) in enumerate(GroupKFold(n_splits=5).split(dataset,groups=groups)):
        training_set = Subset(dataset, train_index)
        val_set = Subset(dataset, val_index)
        
        train_dataloader = DataLoader(training_set, batch_size=hparams["data"]["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=hparams["data"]["batch_size"], shuffle=False)
        
        log_dir = "lightning_logs"

        logger = TensorBoardLogger(log_dir, name=exp_name, default_hp_metric=False)
        os.makedirs(os.path.dirname(logger.log_dir), exist_ok=True)

        model = Predictor(model_params)  ###.to(device)
        ltrainer = lTrainer(model=model, hparams=hparams)
        
        log_every_n_steps = len(train_dataloader)
        check_val_every_n_epoch = 1
        trainer = L.Trainer(max_epochs=n_epochs,logger=logger,log_every_n_steps=log_every_n_steps,  
                            check_val_every_n_epoch=check_val_every_n_epoch,
                            enable_progress_bar=False, enable_checkpointing=False)
        
        trainer.fit(ltrainer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        last_checkpoint = os.path.join(logger.log_dir, "checkpoints", "last.ckpt")
        trainer.save_checkpoint(last_checkpoint)
        
        outputfname = os.path.join(log_dir, os.path.dirname(exp_name), "results.pklz.fold{}".format(fold_idx))
        
        results_train =  trainer.validate(ltrainer, dataloaders=train_dataloader)
        results_val =    trainer.validate(ltrainer, dataloaders=val_dataloader)
        results_test =   trainer.validate(ltrainer, dataloaders=test_dataloader)

        results = [results_train, results_val, results_test]

        results.append(last_checkpoint)
        
        write_pklz(outputfname, results)
        all_fold_results.append(results)
        
        if debug:
            break

    outputfname = os.path.join(log_dir, os.path.dirname(exp_name), "results.pklz")
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
    