import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os, sys
import matplotlib
import socket
import numpy as np
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

if (socket.gethostname() == "cmm0958"):
    matplotlib.use('tkagg') 
else:
    matplotlib.use('agg') 

from utils_plots.utils_plots import better_lookin
from src.fusionattn import FusionAttn
from src.kalman import Kalman

from src.modalities import compute_loss, update_modality_level_estimator, LinearMeasurementModel, SensorSystem
from utils_tbox.utils_tbox import write_pklz


def test2():
    from pykalman import KalmanFilter
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import ma

    # enable or disable missing observations
    use_mask = 1

    # reading data (quick and dirty)
    Time = []
    X =[ ]

    for line in open('data/dataset_01.csv'):
        f1, f2  = line.split(';')
        Time.append(float(f1))
        X.append(float(f2))
    

    if (use_mask):
        start_idx = 450

        end_idx = start_idx + 120

        #start_idx,end_idx = 500,575
        #X = np.array(X)
        #X = X + np.random.randn(*X.shape)
        X = ma.asarray(X)
        X[start_idx:start_idx+50] = ma.masked
        X[end_idx-50:end_idx] = ma.masked

    # Filter Configuration

    # time step
    dt = Time[2] - Time[1]

    # transition_matrix  
    F = [[1,  dt,   0.5*dt*dt], 
        [0,   1,          dt],
        [0,   0,           1]]  

    # observation_matrix   
    H = [1, 0, 0]

    # transition_covariance 
    Q = [[   1,     0,     0], 
        [   0,  1e-4,     0],
        [   0,     0,  1e-6]] 

    # observation_covariance 
    R = [0.04] # max error = 0.6m

    # initial_state_mean
    X0 = [0,
        0,
        0]

    # initial_state_covariance
    P0 = [[ 10,    0,   0], 
        [  0,    1,   0],
        [  0,    0,   1]]

    n_timesteps = len(Time)
    n_dim_state = 3

    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    # Kalman-Filter initialization
    kf = KalmanFilter(transition_matrices = F, 
                    observation_matrices = H, 
                    transition_covariance = Q, 
                    observation_covariance = R, 
                    initial_state_mean = X0, 
                    initial_state_covariance = P0)

    # iterative estimation for each new measurement
    for t in range(n_timesteps):
        if t == 0:
            filtered_state_means[t] = X0
            filtered_state_covariances[t] = P0
        else:
            filtered_state_means[t], filtered_state_covariances[t] = (
            kf.filter_update(
                filtered_state_means[t-1],
                filtered_state_covariances[t-1],
                observation = X[t])
            )

    position_sigma = np.sqrt(filtered_state_covariances[:, 0, 0]);        

    theplotmask = X.mask

    # plot of the resultin trajectory  
    fig,ax=plt.subplots(figsize=(5,4))    
    ax.scatter(np.array(Time)[~theplotmask],X[~theplotmask] ,color="darkgreen", label="With data", s=20)
    ax.scatter(np.array(Time)[theplotmask], X.data[theplotmask],  color="darkgreen", label="_Without data", s=20, alpha=0.2)

    ax.scatter(np.array(Time)[theplotmask], filtered_state_means[theplotmask, 0], marker="x", color="black", label="Without data", s=20)
    ax.fill_between(Time,
                    y1=filtered_state_means[:, 0] - position_sigma, 
                    y2=filtered_state_means[:, 0] + position_sigma, color="gray", label="Standard. Dev.", alpha=0.2,zorder=-12)
    
    ax.legend(loc="upper left", title='Estimation')
    ax.set_xlabel("Time")
    ax.set_ylabel("State component 1")  
    offset = 50
    ax.set_xlim([start_idx - offset, end_idx+offset])
    ax.set_ylim([X[start_idx - offset], X[end_idx+offset+2]])
    better_lookin(ax,fontsize=12,grid=False,legend=True)
    ax.legend(title='Estimation')
    plt.tight_layout()
    fig.savefig("plots/example_missing_interval.pdf", dpi=300, bbox_inches="tight")
    plt.show()   

def test():
    """ One kalman filter"""
    d = 2
    h = 2
    A = torch.randn(h, h)
    #A = torch.eye(h)

    B = torch.randn(d, d)
    #B = torch.eye(d)

    s_0 = torch.zeros(h)
    R = torch.eye(d)
    Q = torch.eye(h)#*0.01
    
    N = 100

    timeline = torch.sort(torch.rand(N)).values
    timeline_process = torch.linspace(0,1,N)
    deltas = timeline_process.diff(prepend=torch.tensor([timeline_process[0]]))
    
    random_walk = torch.randn(N,h)
    for t in range(1, N):
        random_walk[t] = torch.linalg.matrix_exp(A*deltas[t-1]).mv(random_walk[t-1]) + Q.sqrt().mv(torch.randn(h))
    from numpy import ma
    from pykalman import KalmanFilter
    idim = 0
    for p in [1]:
        thekalman = Kalman(A,B,Q,R,s_0,timeline[0])
        
        Nmeasurements = int(N*p)
        timeline_idx = torch.randperm(N)[:Nmeasurements].sort().values#torch.sort(torch.rand(T)).values
        timeline = timeline_process[timeline_idx]
        mask = torch.zeros_like(timeline_process,dtype=bool)
        mask[timeline_idx] = True
        measurements = random_walk[timeline_idx] @ B.T + torch.randn(Nmeasurements,d)@R.sqrt().T #thekalman.draw(random_walk[timeline_idx], timeline)
        mask[N//4: 3*(N//4)] = False
        measurements = ma.asarray(measurements.numpy())
        measurements[~mask] = ma.masked
        # =  None 
        kf = KalmanFilter(transition_matrices = A, observation_matrices = B)

        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)

        #thekalman.track(measurements, timeline_process, mask=mask)
        #mean_estimate, covariance_estimate = thekalman.get_tracking_data("state_estimates"), thekalman.get_tracking_data("cov_estimates")
        true_state = random_walk
        
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
        
        ax.plot(timeline if timeline_process is None else timeline_process, true_state[:,idim],'.-',color="black", label="Ground truth")
        
        ax.plot(timeline, measurements[:,0],'.-',color="darkgreen", label="Measurements")
        ax.plot([timeline,timeline], [torch.ones_like(timeline)*measurements[:,idim].min(),measurements[:,idim]],'--',color="gray", label="_Measurements",lw=1)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Component 1")
        ymin,ymax=ax.get_ylim()
        #ax.set_ylim([ymin,ymax*1.1])
        ax.plot(timeline, filtered_state_means[:,idim], '.-',color="darkred", label="Prediction")
        ax.fill_between(timeline,
            y1=filtered_state_means[:,idim]-np.sqrt(filtered_state_covariances[:,idim,idim]),
            y2=filtered_state_means[:,0]+np.sqrt(filtered_state_covariances[:,idim,idim]),
            alpha=0.5,color="darkred", label="Cov")
        

        #thekalman.plot(ax, random_walk, measurements, mean_estimate, covariance_estimate, timeline, timeline_process=timeline_process)

        #axes[0].legend(ncol=1)
        #axes[0].get_legend().set_bbox_to_anchor(bbox=(1,1.2))
        ax.legend()
        ax.set_title("Missing measurements {}%".format(int(round(100*(1-p)))))
        better_lookin(ax, fontsize=14,grid="off",legend=True, ncol=2)

        plt.tight_layout()
        #fig.savefig("plots/missing_{}.pdf".format(int(round(100*(1-p)))),dpi=300, bbox_inches="tight")
        
        plt.show()
        print("")


def infer_model_hparams(cfg):
    M = cfg["M"]
    if isinstance(cfg["P"], list):
        assert(isinstance(cfg["D"],list))
        assert(isinstance(cfg["SMNR_db"],list))
        P = cfg["P"]
        SMNR_db = cfg["SMNR_db"]
        assert(isinstance(cfg["modalities_dimension"],dict))
        modalities_dimension = cfg["modalities_dimension"]
        cfg["M"] = len(modalities_dimension)
        assert(len(P)==M)
        assert(len(SMNR_db)==M)
        assert(len(modalities_dimension)==M)
    else:
        cfg["P"] = [cfg["P"]] * M
        cfg["SMNR_db"] =  [cfg["SMNR_db"]] * M
        cfg["modalities_dimension"] = {"Modality {}".format(i+1): cfg["modalities_dimension"] for i in range(M)}
    return  M, cfg["P"], cfg["SMNR_db"] ,  cfg["modalities_dimension"]

class Predictor(torch.nn.Module):
    def __init__(self,hparams):
        super(Predictor, self).__init__()
        self.hparams = hparams
        #{"reference": 1, *}
        #hparams["modalities_dimension"] = {"reference": 1, **{m: d for m,d in hparams["modalities_dimension"].items()}}
        self.fusion_model = FusionAttn(hparams)

    def forward(self, calX, pool=None,only_last=True):
        state_and_cov_estimates = self.fusion_model.estimate_fusion( {m: calX[m]["data"] for m in calX.keys()} )
        
        current_state_mean = state_and_cov_estimates[0, :, :self.hparams["d_out"]//2]
        scale_tril_hat = state_and_cov_estimates[0, :, self.hparams["d_out"]//2:]

        current_state_covariance = covariance_model(scale_tril_hat)
        return current_state_mean, current_state_covariance

def covariance_model(scale_tril_hat):
    return torch.diag_embed(torch.nn.functional.softplus(scale_tril_hat).sqrt())

def torch_causal_scaled_dot_product_attention(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

def run_epoch(fusionmodel, optimizer, batch, timeline_predictions, fused_mean_estimate,fused_covariance_estimate,verbose=0):
    optimizer.zero_grad()
    h = fused_mean_estimate.shape[1]

    epoch_nllh = 0.

    ## Fusion
    current_state_mean, current_state_covariance = fusionmodel(batch)

    loss = compute_loss(batch, current_state_mean, current_state_covariance)

    if loss.grad_fn:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_nllh += loss.item()
    
    # LOG output   
    fused_mean_estimate[:] = current_state_mean.detach()
    fused_covariance_estimate[:] = current_state_covariance.detach()
    return epoch_nllh 

def pointwise_NMSE(prediction, target):
    """tensors of size (N,h)"""

    target_variance = (target - target.mean(0)).square().sum(1) ###.reshape(1,-1)

    pw_nmse = (prediction - target).square().sum(1).detach()  /target_variance.sum()
    return pw_nmse

def run_experiment(cfg, plot=False,save=False,show=False,idim = 0, verbose=0, profiler=False):
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"

    manual_seed = cfg["manual_seed"]
    if not (manual_seed is None):
        torch.manual_seed(manual_seed)
    infer_model_hparams(cfg["sensors"])

    cfg["model"]["d_out"] = 2 * cfg["system"]["h"]

    thesystem = SensorSystem(cfg["system"], device=device)

    timeline, random_walk, Filters, Modalities, data_dimensions = thesystem.get_samples_sensors(cfg["sensors"])
    timeline_predictions = timeline
    timeline_predictions_idx = torch.isin(timeline, timeline_predictions)
    
    batch = thesystem.get_local_predictions(timeline_predictions, Filters, Modalities)

    timeline_val, random_walk_val, Filters_val, Modalities_val, _ = thesystem.get_samples_sensors(cfg["sensors"], N=cfg["sensors"]["N_val"], Modalities=Modalities)
    timeline_predictions_val = timeline_val
    timeline_predictions_val_idx = torch.isin(timeline_val, timeline_predictions_val)
    
    batch_val = thesystem.get_local_predictions(timeline_predictions_val, Filters_val, Modalities_val)

    modalities_dimension = {}
    if cfg["model"]["qk_type"]=="time":
        modalities_dimension["q"] = 1
        modalities_dimension["k"] = {m: 1 for m in data_dimensions.keys()}
        modalities_dimension["v"] = {m: d for m,d in data_dimensions.items()}

    elif cfg["model"]["qk_type"] == "data":
        modalities_dimension["q"] = 1
        modalities_dimension["k"] = {m:d+2 for m,d in data_dimensions.items()}
        modalities_dimension["v"] = {m:d+2 for m,d in data_dimensions.items()}

    cfg["model"]["modalities_dimension"] = modalities_dimension

    if plot:
        all_colors = ["darkred", "darkblue", "orange", "cyan", "maroon"]*10
        color_modalities = {m: all_colors[i] for i,m in enumerate(Modalities.keys())}
        figs = {}
        axes = {}
        lines = {}
        lines2 = {}
        lines3 = {}
        lines4 = {}
        # State estimate

        for m in Modalities.keys():
            figs[m],  axes[m] = plt.subplots(figsize=(12, 4))
            lines[m] = None
            lines2[m] = None
        
        # NMSE vs epochs
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        # NMSE vs time
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        plotted = False
        fig_nllh, ax_nllh = plt.subplots(figsize=(5,4))
    
    # Make regular predictions at every time steps
    timeline_predictions = timeline
    timeline_predictions_idx = torch.isin(timeline, timeline_predictions)
    
    fused_CF_mean_estimate = thesystem.get_closed_form_fusion(batch)
    fused_CF_mean_estimate_val = thesystem.get_closed_form_fusion(batch_val)

    results = {"NMSE": {}, "cfg": cfg}

    ## Baselines
    ### Modality wise NMSE
    def get_modality_losses(Filters,random_walk, batch, fused_CF_mean_estimate, results_nmse,append_name=""):

        modality_losses = {}
        for m in Filters.keys():
            on_update_mean_estimate = Filters[m].get_tracking_data("state_estimates")

            modality_losses[m] = pointwise_NMSE(on_update_mean_estimate,  random_walk.cpu()).view(1,-1)#.square()).sum(1))/signal_variance.cpu().sum()).view(1,-1)
            results_nmse[m+append_name] = modality_losses[m].mean().item()

        ### The best possible predictions by an oracle using the best individual prediction from modalities at each timestep.
        envelop = torch.min(torch.cat(list(modality_losses.values())).cpu(), dim=0).values
        results_nmse["Min. Modalities"+append_name] = envelop.mean().item()
        
        loss_CF = pointwise_NMSE(fused_CF_mean_estimate.cpu(),random_walk.cpu()).view(1,-1)
        results_nmse["CF"+append_name] = loss_CF.mean().item()
        return modality_losses ,envelop, loss_CF
    
    modality_losses, envelop, loss_CF = get_modality_losses(Filters,random_walk, batch,fused_CF_mean_estimate, results["NMSE"]) 

    _,_,_ = get_modality_losses(Filters_val,random_walk_val, batch_val,fused_CF_mean_estimate_val, results["NMSE"], append_name=" (val)") 

    # Model & training
    fusionmodel = Predictor(cfg["model"])   
    fusionmodel.to(device)
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=cfg["training"]["lr"])

    every_epochs = 10
    n_epochs = cfg["training"]["n_epochs"]

    # Will be editted in place in each epoch
    fused_mean_estimate = torch.zeros(timeline_predictions.shape[0],       cfg["system"]["h"], device=device)
    fused_covariance_estimate = torch.zeros(timeline_predictions.shape[0], cfg["system"]["h"], cfg["system"]["h"], device=device)
    
    NLLH = []
    NLLH_val = []

    if profiler:
        activities = ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU])
        sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    
    for epoch in range(-1, n_epochs):
        if verbose > 1:
            start_time = datetime.now()
        
        if profiler and (epoch >= 10):
            with profile(activities=activities, record_shapes=True) as prof:
                epoch_nllh = run_epoch(fusionmodel, optimizer, batch, timeline_predictions, fused_mean_estimate,fused_covariance_estimate,verbose=verbose)
            
            print(prof.key_averages().table(sort_by= "cpu_time_total", row_limit=10))
            
            if torch.cuda.is_available():
                print(prof.key_averages().table(sort_by= "cuda_time_total", row_limit=10))
            sys.exit(0)
        else:
            epoch_nllh = run_epoch(fusionmodel, optimizer, batch, timeline_predictions,fused_mean_estimate,fused_covariance_estimate,verbose=verbose)
        
        if verbose > 1:
            print("Epoch={}/{}".format(epoch, n_epochs), "Elapsed=", (datetime.now()-start_time).total_seconds())
        
        NLLH.append(epoch_nllh)

        # Validation
        with torch.no_grad():
            fused_mean_estimate_val, fused_covariance_estimate_val = fusionmodel(batch_val)
            NLLH_val.append(compute_loss(batch_val, fused_mean_estimate_val, fused_covariance_estimate_val).item())

        # Compute loss metrics and plots
        ### Pointwise NMSE
        epoch_mseloss_val_vs_time = pointwise_NMSE(fused_mean_estimate_val[timeline_predictions_val_idx], random_walk_val[timeline_predictions_val_idx])
        epoch_mseloss_val = epoch_mseloss_val_vs_time.sum()/ timeline_predictions_val.shape[0]
        
        results["NMSE"]["Us (val)"] = epoch_mseloss_val.item()

        epoch_mseloss_vs_time = pointwise_NMSE(fused_mean_estimate[timeline_predictions_idx], random_walk[timeline_predictions_idx])
        epoch_mseloss = epoch_mseloss_vs_time.sum()/ timeline_predictions.shape[0]
        
        results["NMSE"]["Us"] = epoch_mseloss.item()

        ### The best possible predictions by an oracle using the best individual prediction from modalities at each timestep.
        # envelop = torch.min(torch.cat(list(modality_losses.values())).cpu(), dim=0).values
        # results["NMSE"]["Min. Modalities"] = envelop.mean().item()
    
        # MAKE PLOTS
        if plot:
            # At the first epoch, compute the error of each individual modality

            if epoch == -1:

                ax2.plot([epoch,n_epochs], [envelop.mean()]*2, color="black", label=("_" if plotted else "") + "Closest")
                ax2.plot([epoch,n_epochs], [results["NMSE"]["CF"]]*2, color="orange", label=("_" if plotted else "CF"))

                ax3.plot(timeline_predictions.cpu(), envelop, "-x", color="black",label="Closest")
                
                for m in Modalities.keys():
                    # Plot ground truth
                    axes[m].plot(timeline.cpu(), random_walk[:,idim].cpu(), color="black", label="Ground truth", lw=2)
                    
                    # Plot predicted state
                    on_update_mean_estimate = Filters[m].get_tracking_data("state_estimates")
                    on_update_timeline = Filters[m].get_tracking_data("update_time")

                    tl_idx = Modalities[m].timeline_idx.cpu()
                    axes[m].plot(on_update_timeline, on_update_mean_estimate[:, idim].cpu(), "-", 
                            label="KF",               #+" (log-$\\tau_{}={:.02f}$)".format(im+1,logtau),
                            color=color_modalities[m])
                    axes[m].plot(on_update_timeline, fused_CF_mean_estimate[:, idim].cpu(), "-", 
                            label="CF",                        #+" (log-$\\tau_{}={:.02f}$)".format(im+1,logtau),
                            color="orange")
                    axes[m].scatter(on_update_timeline[tl_idx], on_update_mean_estimate[tl_idx,idim].cpu() ,color=color_modalities[m],s=10)
                    axes[m].plot([on_update_timeline[tl_idx], on_update_timeline[tl_idx]], 
                            [torch.ones_like(on_update_timeline[tl_idx])*on_update_mean_estimate[tl_idx, idim].min(), on_update_mean_estimate[tl_idx, idim].cpu()],
                            '--', color=color_modalities[m], alpha=0.5, label="_Measurements",lw=1)
                    
                    # Plot NMSE
                    ax3.plot(timeline_predictions.cpu(), modality_losses[m].view(-1), "-x", color=color_modalities[m],label=m,alpha=0.3)
                    ax2.plot([epoch,n_epochs], [modality_losses[m].mean()]*2, color=color_modalities[m], label=("_" if plotted else "KF ") + m[-1])

            # epoch is the first, last, or a multiple of every_epochs
            if (epoch % (every_epochs)) == 0 or epoch == (n_epochs-1) or (epoch == -1):
                if epoch >= 0:
                    line_mse[0].remove()
                    
                line_mse = ax3.plot(timeline_predictions.cpu(), epoch_mseloss_vs_time.cpu(), "-x", color="darkgreen", label="Global")

                ax_nllh.scatter([epoch], [NLLH[-1]], color="darkred", label="_" if plotted else "Training")
                ax_nllh.scatter([epoch], [NLLH_val[-1]], color="darkgreen", label="_" if plotted else "Validation")
                ax_nllh.set_xlim([0, n_epochs])
                ax_nllh.set_ylabel("Negative log-likelihood")
                ax_nllh.set_xlabel("Epoch")
                ax_nllh.set_yscale("log")
                ax_nllh.legend()

                for m in Modalities.keys():
                    if epoch  >= 0 :
                        lines[m][0].remove()
                        lines2[m].remove()
                        lines3[m][0].remove()
                        lines4[m].remove()

                    lines[m] = axes[m].plot(timeline_predictions.cpu(), fused_mean_estimate[:, idim].cpu(), "-", label="ITNet (train)", color="darkgreen")
                    lines2[m] = axes[m].fill_between(timeline_predictions.cpu(),
                                        y1=fused_mean_estimate[:, idim].cpu() - fused_covariance_estimate[:,idim,idim].sqrt().cpu(),
                                        y2=fused_mean_estimate[:, idim].cpu() + fused_covariance_estimate[:,idim,idim].sqrt().cpu(),
                                        label="Cov. (train)", color="darkgreen", alpha=0.3)
                    
                    lines3[m] = axes[m].plot(timeline_predictions_val.cpu(), fused_mean_estimate_val[:, idim].cpu(), "-", label="_ITNet", color="darkgreen")
                    lines4[m] = axes[m].fill_between(timeline_predictions_val.cpu(),
                                                    y1=fused_mean_estimate_val[:, idim].cpu() - fused_covariance_estimate_val[:,idim,idim].sqrt().cpu(),
                                                    y2=fused_mean_estimate_val[:, idim].cpu() + fused_covariance_estimate_val[:,idim,idim].sqrt().cpu(),
                                                    label="_Cov.", color="darkgreen", alpha=0.3)

                    if epoch == (n_epochs-1):
                        on_update_mean_estimate =  Filters[m].get_tracking_data("state_estimates")
                        on_update_timeline = Filters[m].get_tracking_data("update_time")

                        tl_idx = Modalities[m].timeline_idx.cpu()
                        #axes[m].plot(on_update_timeline, on_update_mean_estimate[:, idim].cpu(), "-.", 
                        #        label=m+ " Final",                           #+" (log-$\\tau_{}={:.02f}$)".format(im+1,logtau),
                        #        color=color_modalities[m] )
                        axes[m].scatter(on_update_timeline[tl_idx], on_update_mean_estimate[tl_idx,idim].cpu() ,color=color_modalities[m],s=10)
                        axes[m].plot([on_update_timeline[tl_idx], on_update_timeline[tl_idx]], 
                                [torch.ones_like(on_update_timeline[tl_idx])*on_update_mean_estimate[tl_idx, idim].min(), on_update_mean_estimate[tl_idx, idim]],
                                '--', color=color_modalities[m], alpha=0.5, label="_Measurements",lw=1)

                    axes[m].set_xlabel("Time")
                    axes[m].set_ylabel("{}\nComponent {}".format(m,idim+1))
                    axes[m].relim()
                    axes[m].set_xlim([0,cfg["sensors"]["N"]//10])
                    axes[m].legend(ncols=6,bbox_to_anchor=(1, 1.1))

                ax2.set_xlabel("Number of epochs")
                ax2.set_ylabel("Normalized MSE")
                ax2.scatter(epoch, epoch_mseloss.item(),color="darkred",marker="x",label=("_" if plotted else "")+ "ITNet (train)")
                ax2.scatter(epoch, epoch_mseloss_val.item(),color="darkgreen",marker="x",label=("_" if plotted else "")+ "ITNet")

                ax3.set_xlabel("Time")
                ax3.set_yscale("log")
                ax3.set_ylabel("Normalized SE")
                ax3.relim()
                ax3.legend()
                if not plotted:
                    plotted = True

                plt.pause(0.05)
        
        # Profiler
        # Early stopping:
        if epoch >= n_epochs//10:
            if NLLH[-1] > NLLH[-2]:
                pass
    
    # End all training epochs
    modality_losses_final = {}
    for m in Modalities.keys():
        on_update_mean_estimate = Filters[m].get_tracking_data("state_estimates")

        modality_losses_final[m] = pointwise_NMSE(on_update_mean_estimate, random_walk.cpu()).view(1,-1)
        results["NMSE"][m + " Final"] = modality_losses_final[m].mean().item()

    if plot:
        ax2.set_xlim([0, n_epochs])
        ax2.legend()
        ax2.set_yscale("log")

        better_lookin(ax_nllh, fontsize=12, grid=False, legend=True)
        better_lookin(ax3, fontsize=12, grid=False, legend=True, ncol=2)
        better_lookin(ax2, fontsize=12, grid=False, legend=True, ncol=2)
        for m in Modalities.keys():
            better_lookin(axes[m], fontsize=12, grid=False, legend=True, ncol=6,legend_bbox=(1,1.1))
        
        if show:
            plt.show()
        
        if save:
            plt.tight_layout()
            for m in Modalities.keys():
                figs[m].savefig("plots/fusion_regression_data_{}_{}.pdf".format(manual_seed,m), bbox_inches="tight", dpi=300)
            fig2.savefig("plots/fusion_regression_mse_{}.pdf".format(manual_seed), bbox_inches="tight", dpi=300)
            fig3.savefig("plots/fusion_regression_mse_vs_time_{}.pdf".format(manual_seed), bbox_inches="tight", dpi=300)
            fig_nllh.savefig("plots/fusion_regression_nllh_vs_time_{}.pdf".format(manual_seed), bbox_inches="tight", dpi=300)
    return results

def main(args):
    cfg_fname = args.i

    output_fname = args.o


    
    plot = args.plot
    save = args.save
    show = args.show
    all_seeds = range(10)
    #all_seeds = [5]
    R = []
    for manual_seed in all_seeds:
        with open(cfg_fname,"r") as fp:
            cfg = json.load(fp)["params"]

        cfg["manual_seed"] = manual_seed
        if args.v>0:
            print("Seed=",manual_seed)

        R.append(run_experiment(cfg, plot=plot, save=save, show=show, idim = 0, verbose=args.v, profiler=args.profile))
        
        if plot:
            plt.close("all")
    write_pklz(output_fname.replace(".pkl",".pklz"), R)


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