import pandas as pd
import torch
import numpy as np 

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

from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut

from torch.utils.data import Dataset, DataLoader

from lightning.pytorch.loggers import TensorBoardLogger


import lightning as L
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

"""
Column 1: Force (N) in X dimension
Column 2: Force (N) in Y dimension
Column 3: Force (N) in Z dimension
Column 4: Vibration (g) in X dimension
Column 5: Vibration (g) in Y dimension
Column 6: Vibration (g) in Z dimension
Column 7: AE-RMS (V) 
"""

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


class Predictor(torch.nn.Module):
    def __init__(self, hparams):
        super(Predictor, self).__init__()
        self.hparams=hparams

        self.feature_extractor = FeatureExtractor(hparams["dimensions"], h_dim=hparams["h_dim"], kernel_size=hparams["kernel_size"])

        self.fusion_model = FusionAttn(hparams)
    
    def forward(self, batch):
        thefeatures = self.feature_extractor(batch)
        thefeatures["reference"] = batch["inference_timeline"].unsqueeze(-1)
        thefeatures = {m: {"data":v.unsqueeze(1)} for m,v in thefeatures.items()}
        
        yhat, _ = self.fusion_model(thefeatures)
        return yhat

def get_data(hparams, device="cpu"):

    columns = ["F_x","F_y","F_z","V_x", "V_y", "V_z", "AE"]
    
    data = {}
    fs = 50000
    T_min={}
    for cutter_no in hparams["cutter_list"]:
        data_fname = "data/phm2010/c{}.pklz".format(cutter_no)
        if not os.path.exists(data_fname):
            fname_wear = "data/phm2010/c{}/c{}_wear.csv".format(cutter_no, cutter_no)
            
            Y = pd.read_csv(fname_wear)
            n_tot = Y.shape[0]
        
            X_l = []
            Y_l = []
            
            for i in range(n_tot):
                fname = "data/phm2010/c{}/c{}/c_{}_{:03d}.csv".format(cutter_no,cutter_no,cutter_no,i+1)
                X = pd.read_csv(fname, names=columns)
                X_l.append(X)
                Y_l.append(Y.iloc[i].values[1:])
            
            T_min[cutter_no] = min([X.shape[0] for X in X_l])
            
            X = torch.cat([torch.from_numpy(X.values[:T_min]).unsqueeze(0) for X in X_l]).to(dtype=torch.float32)
            Y = torch.cat([torch.from_numpy(Y).unsqueeze(0) for Y in Y_l]).to(dtype=torch.float32)

            Y_timeline = np.cumsum([X.shape[0] /fs /60 for X in X_l])
            
            #def pairwisedistance(x1,x2):
            #    return ((x1-x2)**2)**(0.5)
            #print(pairwisedistance(Y_timeline.reshape(-1,1),Y_timeline.reshape(1,-1)))
            
            #Y_timeline = np.arange(Y_timeline.shape[0])        
            #print(pairwisedistance(Y_timeline.reshape(-1,1),Y_timeline.reshape(1,-1)))
            
            print(X.shape, Y.shape)
            
            thedata = {"X": X, "Y":Y, "timeline": torch.from_numpy(Y_timeline).to(dtype=torch.float32)}
            write_pklz(data_fname,thedata)
        else:
            thedata = read_pklz(data_fname)
        
        T_min[cutter_no] = thedata["X"].shape[1]
        data[cutter_no] = thedata
    T = min(T_min.values())

    for cutter_no,thedata in data.items():
        if hparams["batch_size"]>1:
            thedata["X"] = thedata["X"][:,:T,:]
        
        thedata["calX"] = {
                "Force":     thedata["X"][:,:,0:3],
                "Vibration": thedata["X"][:,:,3:6], 
                "AE":        thedata["X"][:,:,6:7]
            }
        
        
    return data, thedata["timeline"]

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

def get_masked(data, inference_timeline,hparams,device="cpu"):
    batches = []
    modality_names = ["Force","Vibration","AE"]

    for cutter_no, thedata in data.items():
        n_tot = thedata["Y"].shape[0]

        N = hparams["data"]["N"]
        idx = {}
        for thename in modality_names:
            if hparams["data"]["spacing"] == "regular":
                idx[thename] = np.arange(0, n_tot, np.floor(n_tot/N))
            
            elif hparams["data"]["spacing"] == "random":
                idx[thename] = np.sort(np.random.permutation(n_tot-1)[:(N-1)]+1)#np.arange(0, n_tot, n_tot//N)
                idx[thename] = np.concatenate([np.zeros(1),idx[thename]])

        thedata["idx"] = idx
        
        timelines  = {m: thedata["timeline"][thedata["idx"][m]].unsqueeze(-1).to(device) for m in  thedata["dimensions"].keys()}

        model_params = hparams["model"]
        model_params["modalities_dimension"] = {m: model_params["h_dim"] for m in  thedata["dimensions"].keys()}
        model_params["init_tau"] = torch.diff(thedata["timeline"]).min().item()*5
        batches.append({"cutter_no":cutter_no,"idx":idx, "timelines":timelines, "inference_timeline":inference_timeline.to(device), "Y":thedata["Y"].to(device)})
    
    return batches, model_params


class TheDataset(Dataset):
    def __init__(self, batches, data,device="cpu"):
        self.batches = batches
        self.data_calX = {cutter_no: {m:data[cutter_no]["calX"][m].to(device=device)for m in data[cutter_no]["calX"].keys()} for cutter_no in data.keys()}
        self.mu=None
        self.sigma=None
        self.device = device
    def __len__(self):
        return len(self.batches)

    def get_calX(self, i):
        cutter_no = self.batches[i]["cutter_no"]
        return {m: self.data_calX[cutter_no][m][self.batches[i]["idx"][m]] for m in self.data_calX[cutter_no].keys()}
    
    def __getitem__(self, i):
        #idx={}
        #idx[thename] = np.sort(np.random.permutation(n_tot-1)[:(N-1)]+1)#np.arange(0, n_tot, n_tot//N)
        #idx[thename] = np.concatenate([np.zeros(1),idx[thename]])
        if not (self.mu is None):
            self.batches[i]["Y_n"] = (self.batches[i]["Y"]-self.mu)/self.sigma
        return {**self.batches[i], "calX":self.get_calX(i)}


def get_batches(data,inference_timeline,hparams,device="cpu",n_draws = 2):
    B = []
    for thedraw in range(n_draws):
        batches, model_params = get_masked(data,inference_timeline,hparams,device=device)
        B.append(batches)
        #all_calX.append(calX)
    B = sum(B, [])
    
    model_params["dimensions"] = {"Force": 3, "Vibration": 3, "AE": 1}

    return TheDataset(B, data, device=device), model_params

class lTrainer(L.LightningModule):
    def __init__(self, model=None, hparams=None):
        super(lTrainer, self).__init__()
        self.model = model
        #self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.val_scores = {}
        self.val_scores_msa = {}
        self.val_scores_dms = {}
        self.loss_function = torch.nn.MSELoss()

        self.train_recon_figure     = plt.subplots(figsize=(10,6))
        self.val_recon_figure     = plt.subplots(figsize=(10,6))
        self.spectrogram_figure     = [plt.subplots(figsize=(10,6)) for _ in range(4)]
        self.val_attn_matrix     = plt.subplots(figsize=(10,6))

        self.the_training_step  = 0

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
        {"mse/val": torch.nan, "mse/train": torch.nan})

        self.val_scores_msa = {}
        self.val_scores_dms = {}
    
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = 0.0
        self.the_training_step+=1
        log_dict = {}
        y = batch["Y_n"]
        y_n = y#(y-mu)/sigma

        yhat = self.model(batch)
        loss = self.loss_function(y_n, yhat)
        self.log("mse/train", loss)

        if batch_idx == 0:
            i=0
            timeline = batch["inference_timeline"].cpu()[i]
            #fig, ax = plt.subplots()
            ax = self.train_recon_figure[1]
            ax.cla()
            ax.set_prop_cycle(color=["darkred", "darkgreen","orange"])
            ax.plot(timeline,y_n.detach().cpu()[i], label="Data",color="black")
            ax.plot(timeline,yhat.detach().cpu()[i], "--",label="Prediction")
            ax.set_xlabel("Prediction time (min)")
            ax.set_ylabel("Output amplitude")
            self.logger.experiment.add_figure("recon_figure/train", self.train_recon_figure[0], self.the_training_step)
            cnn_out = self.model.feature_extractor.modality_models["Vibration"].cnn.data_out.detach().cpu()

            for j in range(4):
                self.spectrogram_figure[j][1].cla()
                plot_spectrogram(self.spectrogram_figure[j][1], cnn_out[i,j], self.model.feature_extractor.modality_models["Vibration"].cnn.kernel_size)
                self.logger.experiment.add_figure("spectrogram_figure_{}/train".format(j), self.spectrogram_figure[j][0], self.the_training_step)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y = batch["Y_n"]
        y_n = y#(y-mu)/sigma

        yhat = self.model(batch)
        loss_val = self.loss_function(y_n, yhat)
        self.log("mse/val", loss_val)

        if batch_idx == 0:
            i=0
            timeline = batch["inference_timeline"].cpu()[i]

            #fig, ax = plt.subplots()
            ax = self.val_recon_figure[1]
            ax.cla()
            ax.set_prop_cycle(color=["darkred", "darkgreen","orange"])
            ax.plot(timeline,y_n.detach().cpu()[i], label="Data",color="black")
            ax.plot(timeline,yhat.detach().cpu()[i], "--",label="Prediction")
            ax.set_xlabel("Prediction time (min)")
            ax.set_ylabel("Output amplitude")
            self.logger.experiment.add_figure("recon_figure/val", self.val_recon_figure[0], self.the_training_step)


            #fig,ax = plt.subplots()
            timelines = {m:batch["timelines"][m][i] for m in batch["timelines"].keys()}

            ax = self.val_attn_matrix[1]
            ax.cla()
            ax.imshow(self.model.fusion_model.estimate_fusion.A.detach().cpu()[i,0].T, aspect="auto", cmap="Greys", origin="lower",
                extent=[timeline.min(), timeline.max(), timelines["AE"].cpu().min(), timelines["AE"].cpu().max()])
            ax.set_ylabel("Measurement time (min)")
            ax.grid(True)
            ax.set_xlabel("Prediction time (min)")
            self.logger.experiment.add_figure("attn_matrix/val", self.val_attn_matrix[0], self.the_training_step)

            #fig.savefig(output_fname.replace(".pklz","/attention_matrix_cutter.pdf"))


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
    exp_name = exp_name_##os.path.join(exp_name_, os.path.basename(dms_dataset_name), "fold{}".format(fold_idx))
    
    seed=12345
    torch.manual_seed(seed)
    np.random.seed(2*seed)
    torch.set_float32_matmul_precision('medium')
    plot = args.plot
    save = args.save
    show = args.show

    with open(cfg_fname,"r") as fp:
        hparams = json.load(fp)["params"]
    
    lr = hparams["training"]["lr"]   ###=1e-3
    n_epochs = hparams["training"]["n_epochs"]

    data, inference_timeline = get_data(hparams["data"])
    dataset, model_params = get_batches(data, inference_timeline,hparams, n_draws=hparams["data"]["n_draws"])
    groups = [dataset.batches[i]["cutter_no"] for i in range(len(dataset.batches))]
    all_fold_results = []

    for fold_idx,(train_index, test_index) in enumerate(LeaveOneGroupOut().split(dataset,groups=groups)):
        
        mu,sigma = get_mu_sigma(dataset,train_index)

        training_set = Subset(dataset, train_index)
        test_set = Subset(dataset, test_index)
        dataset.mu = mu
        dataset.sigma = sigma
        train_dataloader = DataLoader(training_set, batch_size=hparams["data"]["batch_size"], shuffle=True)
        val_dataloader = DataLoader(test_set, batch_size=hparams["data"]["batch_size"], shuffle=False)
        log_dir = "lightning_logs"

        logger = TensorBoardLogger(log_dir, name=exp_name, default_hp_metric=False)
        os.makedirs(os.path.dirname(logger.log_dir), exist_ok=True)

        model = Predictor(model_params)#.to(device)
        ltrainer = lTrainer(model=model,hparams=hparams)
        
        log_every_n_steps = len(train_dataloader)
        check_val_every_n_epoch = 1
        trainer = L.Trainer(max_epochs=n_epochs,logger=logger,log_every_n_steps=log_every_n_steps,  
                            check_val_every_n_epoch=check_val_every_n_epoch, enable_progress_bar=False, enable_checkpointing=False)
        
        trainer.fit(ltrainer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        last_checkpoint = os.path.join(logger.log_dir, "checkpoints", "last.ckpt")
        trainer.save_checkpoint(last_checkpoint)
        
        outputfname = os.path.join(log_dir, os.path.dirname(exp_name), "results.pklz.fold{}".format(fold_idx))

        results = trainer.validate(ltrainer, dataloaders=[train_dataloader,val_dataloader])
        results.append(last_checkpoint)
        write_pklz(outputfname, results)
        all_fold_results.append(results)
    
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
    