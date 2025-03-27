import matplotlib.pyplot as plt
import torch

import numpy as np
import pandas as pd
import random
from utils_plots.utils_plots import better_lookin


def compute_target(data,timelines,d_out):
    Tmax=data["reference"].shape[1]
    N =data["reference"].shape[0]
        
    y = torch.zeros(N, Tmax, d_out)

    time_ref = timelines["reference"]
    for itime, t in enumerate(time_ref):
        x_max_previous = torch.zeros(N,d_out)
        for k in data.keys():
            previous_data = data[k][:,timelines[k] <= t]
            if previous_data.shape[1]>0:
                x_max_previous += data[k][:,timelines[k] <= t].max(1).values
        y[:, itime, :] = x_max_previous
    return y


def compute_target2(data,timelines,d_out):
    Tmax = data["m1"].shape[1]
    N = data["m1"].shape[0]

    y = torch.zeros(N, Tmax, d_out)
    time_ref = timelines["m1"]

    for k in data.keys():
        ikeep = torch.tensor([0])#torch.randint(timelines[k].shape[0],(1,))
        y += data[k][:, [ikeep[0]], :]

    return y

def compute_target3(data,timelines,d_out):
    Tmax = data["m1"].shape[1]
    N = data["m1"].shape[0]

    y = torch.rand(N, Tmax, d_out)
    #time_ref = timelines["m1"]

    #for k in data.keys():
    #    ikeep = torch.tensor([0])#torch.randint(timelines[k].shape[0],(1,))
    #    y += data[k][:, [ikeep[0]], :]

    return y

def compute_target4(data, timelines, d_out):
    Tmax = data["m1"].shape[1]
    N = data["m1"].shape[0]

    y = torch.rand(N, Tmax, d_out)
    #time_ref = timelines["m1"]

    #for k in data.keys():
    #    ikeep = torch.tensor([0])#torch.randint(timelines[k].shape[0],(1,))
    #    y += data[k][:, [ikeep[0]], :]

    return y

def plot_data(data, timelines, target, prediction=None, dim=0, n=0,figsize=None,masks=False,ax=None):
    images=[]
    if masks:
        for k, tl in timelines.items():
            fig, ax = plt.subplots()
            colors={True:"darkgreen",False:"darkred"}
            for iq in range(timelines["m1"].shape[0]):
                c=timelines["m1"][iq]>=tl
                ax.scatter(tl, [timelines["m1"][iq]]*len(tl), c=[colors[cc.item()] for cc in c])
            ax.plot([timelines["m1"][0],timelines["m1"][-1]],[timelines["m1"][0],timelines["m1"][-1]], color="black")
            #im=ax.imshow(timelines["m1"].view(-1,1) >= tl.view(1,-1),cmap="summer",extent=[tl[0],tl[-1],timelines["m1"][0],timelines["m1"][-1]],origin="lower",interpolation="none")
            #ax.scatter([tl[0]]*timelines["m1"].shape[0],timelines["m1"],c="k",marker="X")
            #ax.scatter(tl, [timelines["m1"][0]]*tl.shape[0],c="k")

            #plt.colorbar(im)
            ax.set_ylabel("m1")
            ax.set_xlabel(k)
            #ax.set_xticks(np.linspace(0, len(tl)-1, len(tl)))  # This automatically spaces the ticks
            #ax.set_xticklabels(tl.numpy())  # Set the corresponding t1 labels
            #ax.set_yticks(np.linspace(0, len(timelines["m1"])-1, len(timelines["m1"])))  # This automatically spaces the ticks
            #ax.set_yticklabels(timelines["m1"].numpy())  # Set the corresponding t1 labels

            #ax.set_xticks(tl)  # Set the corresponding t1 labels on the x-axis

            #ax.set_xticklabels(tl)  # Set the corresponding t1 labels on the x-axis
            #ax.set_xticks(tl)
            images.append([fig,ax])
    return_plot=None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_plot=[fig,ax]
    for i,k in enumerate(data.keys()):
        X = data[k][n,0,:,dim]
        timel = timelines[k]
        ax.plot(timel, X-i,"-", label=k.capitalize(), marker='o',linewidth=2,markersize=5)
    ax.plot(timelines["reference"], target[0,:,dim],linewidth=2,marker="o",color="black",label="Target")
    try:
        ax.plot(timelines["reference"], sum(list(data.values()))[n,:,dim],linewidth=2,marker="o",color="gray",label="Sum data")
    except:
        pass

    if not (prediction is None):
        ax.plot(timelines["reference"], prediction[:,dim],linewidth=2,marker="o",color="darkred",label="Prediction")

    ax.legend()
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Data")
    better_lookin(ax, fontsize=14, ncol=4,legend_bbox=(1,1.1),legend=True)
    #ax.legend()
    return return_plot,  images


def prep_data(data, timelines, device="cpu"):
    # Compute timeseries deltas
    deltas = {k: torch.diff(t, prepend=torch.tensor([t[0]])).unsqueeze(0).view(1,t.shape[0],1) for k,t in timelines.items()}
    N=data["reference"].shape[0]
    # Concatenate data and timelines
    calX = {k: torch.cat([data[k], deltas[k].expand(N,-1,-1), timelines[k].unsqueeze(0).unsqueeze(-1).expand(N,-1,-1)], dim=2).unsqueeze(1).to(device) for k in data.keys()}
    return calX

def encode_time(X, device="cpu",tlim=None):
    timelines = {}
    for k,v in X.items():
        colnames = v.columns.tolist()
        v["deltas"] = np.diff(v.index.values,prepend=0)
        v["timeline"] = (v.index-v.index.min())/(v.index.max()-v.index.min())
        timelines[k] = torch.from_numpy(v["timeline"].copy().values)[:tlim]
        v = v.reset_index(drop=True)
        additional_columns = ["deltas","timeline"]
        #additional_columns = ["timeline"]
        #colnames=[]
        X[k] = torch.from_numpy(v[colnames+additional_columns].values.astype(np.float32)[:tlim]).to(device=device).unsqueeze(0).unsqueeze(0)
    return X, timelines

def corrupt(X, p=0):
    """X: a dictionary of modalities {modality_name: pd.DataFrame, ...}
        $p in [0,1]$: the percentage of missing data in the output.
    """
    assert(p < 1)
    if p == 0:
        return X
        
    for k, v in X.items():
        if k != "reference":
            idx_keep = np.zeros(v.shape[0], dtype=bool)
            idx_keep[np.random.permutation(v.shape[0])[int(v.shape[0]*p):]] =True
            X[k] = v[idx_keep].copy()
    return X


def draw_pattern(T=100):
    
    min_w = 10
    max_w = 15
    n=4
    the_diffs = torch.randint(max_w+1, T//n, (n,)).sort().values
    the_diffs[0]-=max_w
    the_starts = torch.cumsum(the_diffs,0)
    
    the_ends = the_starts + torch.randint(3,max_w,the_starts.shape)
    the_ends[-1]=min([the_ends[-1],T-1])
    
    the_val= torch.randint(10,20,(n,)).cumsum(0)
    indexes = torch.randperm(the_val.shape[0])
    the_val = the_val[indexes]
    
    names=["tri","rect", "tri","rect"]
    random.shuffle(names)
    return [{"start":start.item(),"end":end.item(),"val":val.item(),"type":name} for (start,end,val,name) in zip(the_starts,the_ends,the_val,names)]


def add_noise(x,s=0.1):
    return x+torch.randn_like(x)*s

def interp(x_np):
    xx = np.arange(x_np.shape[0])
    x_notnan_b = ~np.isnan(x_np).bool().numpy()
    return torch.from_numpy(np.interp(xx, xx[x_notnan_b], x_np[x_notnan_b]))

def find_target(l_d):
    lout=[{k:v for k,v in dd.items()} for dd in l_d]
    avg_d= {k:np.mean([dd["val"] for dd in l_d if dd["type"] ==k])
            for k in ["tri","rect"]}
    for dd in lout:
        dd["val"]=avg_d[dd["type"]]
    return lout

def apply_shape(xx, shp):
    xx[shp["start"]: shp["end"]]=torch.nan
    
    if shp["type"] == "tri":
        xx[shp["start"] + (shp["end"]-shp["start"])//2]=shp["val"]
    elif shp["type"] == "rect":
        xx[shp["start"]:shp["end"]]=shp["val"]
    
    return xx

def gen_data(N,T,s=0):
    X=torch.zeros(N,1,T)
    Y=torch.zeros(N,1,T)

    ltrain = [draw_pattern(T=T) for _ in range(N)]

    for i, ll in enumerate(ltrain):
        for shp,shp_target in zip(ll,find_target(ll)):
            X[i,0,:] = add_noise(interp(apply_shape(X[i,0,:], shp)),s=s).float()
            Y[i,0,:] = add_noise(interp(apply_shape(Y[i,0,:], shp_target)),s=s).float()
    return X,Y

def get_data(thetype="synthetic",device="cpu",tlim=None,p=0.5,noise_std=0.):
    if thetype == "shapes":
        N=1
        T=100
        s_noise = noise_std
        X, Y    = gen_data(N, T, s=s_noise)
        X = X.transpose(1,2)
        Y = Y.transpose(1,2)
        X = X[0]
        Y = Y[0]
        #plt.close(); plt.plot(X);plt.plot(Y); plt.show()

        #df = pd.DataFrame(data = X.T.astype(np.float32),columns=["reference", "m1"])
        df = pd.DataFrame(data = X.T.astype(np.float32),columns=["reference"])
        data = {"reference":df for k in df.columns}
        X, timelines = encode_time(corrupt(data, p=p))
        
    if thetype == "ssm":
        np.random.seed(9)
        d = 2
        f = 10
        N = 100
        xx = np.linspace(0,1,N)
        u=(np.sin(2*np.pi*f*xx)>0.25)
        U = np.array([(u).tolist()]*d)*0.
        A1 = np.random.randn(d,d)
        A2 = np.random.randn(d,d)
        A3 = np.random.randn(d,d)

        W = np.random.randn(d,N)*noise_std
        
        s_0 = np.zeros((d,1))
        S = [s_0]
        for t in range(1,N):
            s_t = A1 @ S[-1] + A2 @ U[:,t].reshape(-1,1) + W[:, t].reshape(-1,1)
            S.append(s_t)
        S = np.concatenate(S, axis=1)
        if False:
            fig, ax = plt.subplots()
            ax.plot(S[0], label="Modality 1")
            ax.plot(S[1], label="Modality 2")
            ax.plot(U[0], label="Command (target)")
            ###     plt.close(); plt.plot(y); 
            plt.show()
        df = pd.DataFrame(data = S.T.astype(np.float32),columns=["reference", "m1"])
        data = {k:df[[k]] for k in df.columns}
        X, timelines = encode_time(corrupt(data, p=p))
        y = [torch.from_numpy(A3@S).T.unsqueeze(0).to(torch.float32), ["U0","U1"]]

    if thetype == "ssm1":
        np.random.seed(9)
        d = 2
        f = 10
        N = 10
        xx = np.linspace(0,1,N)
        u=(np.sin(2*np.pi*f*xx)>0.25)
        U = np.array([(u).tolist()]*d)*0.

        #A1 = np.random.randn(d,d)
        A1 = np.array([[-1,0.5], [-0.5,-1]])*0.9
        
        A2 = np.random.randn(d,d)
        #A3 = np.random.randn(d,d)
        A3 = np.array([[-2,-1], [1,-3]])

        W = np.random.randn(d,N)*noise_std
        
        s_0 = np.ones((d,1))*0.1
        
        S = [s_0]
        for t in range(1,N):
            s_t = A1 @ S[-1] + A2 @ U[:,t].reshape(-1,1) + W[:, t].reshape(-1,1)
            S.append(s_t)
        S = np.concatenate(S, axis=1)
        if False:
            fig, ax = plt.subplots()
            ax.plot(S[0], label="Modality 1")
            ax.plot(S[1], label="Modality 2")
            ax.plot(U[0], label="Command (target)")
            ###     plt.close(); plt.plot(y); 
            plt.show()
        Y = torch.from_numpy(A3@S).T.unsqueeze(0).to(torch.float32)

        df = pd.DataFrame(data = Y[0], columns=["reference", "m1"])
        data = {"reference": df for k in df.columns}
        X, timelines = encode_time(corrupt(data, p=p))
        #Y = Y.roll(-1,dims=1)#[:,0,:] = 0
        y = [torch.from_numpy(S).T.unsqueeze(0).to(torch.float32), ["U0","U1"],
                torch.nn.Parameter(torch.from_numpy(A1).to(torch.float32)),
                torch.nn.Parameter(torch.from_numpy(A3).to(torch.float32))]
        
        print("")

    if thetype == "robot":
        # https://archive.ics.uci.edu/dataset/963/ur3+cobotops
        fname = "data/dataset_02052023.xlsx"
        df = pd.read_excel(fname).drop(columns=["Num"])
        df.columns = [s.strip() for s in df.columns]
        df["cycle"] /= df["cycle"].max()

        for stem in ["Temperature"]:
            all_cols = [s for s in df.columns if s.startswith(stem)]
            themin=df[all_cols].min().min()
            themax=df[all_cols].max().max()

            df[all_cols] = (df[all_cols]-themin)/(themax-themin)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"].apply(lambda s: s.replace("\"","")),format="%Y-%m-%dT%H:%M:%S.%f%z")
        timeline = (df["Timestamp"]-df["Timestamp"].iloc[0]).dt.total_seconds().values/60
        df.index = timeline
        df.drop(columns=["Timestamp"],inplace=True)
        M = 6
        reference_cols = ["Tool_current", "cycle"]
        df = df[(df[reference_cols].notna().sum(1)==2).values].copy()
        target_cols = ['Robot_ProtectiveStop', 'grip_lost']

        colnames = {**{"reference":reference_cols},**{"m{}".format(m):[s for s in df.columns if s.endswith(str(m))] for m in range(M)}}
        X = {m: df[cols].dropna().copy() for m,cols in colnames.items()}
        df.drop(columns=sum(list(colnames.values()),[]),inplace=True)
        
        df[target_cols[0]] = df[target_cols[0]].fillna(0)
        ydata = torch.from_numpy(df[target_cols].values.astype(np.float32)[:tlim]).unsqueeze(0)
        y=[ydata,target_cols]
        X = corrupt(X, p=p)
        X, timelines = encode_time(X, device=device,tlim=tlim)
        assert(ydata.shape[1] == X["reference"].shape[2])
        #plt.close(); plt.plot(df[targets[1]].values); plt.show()

        print("")

    if thetype == "synthetic":
        M = 3
        Tmax = 100
        Dmax = 2
        N = 4
        torch.manual_seed(2)
        d_out = 2
        
        names = ["reference"] + ["Modality {}".format(i+1) for i in range(1,M)]

        # Create signals from M modalities, all with the same dimension and length, with irregular sampling
        D = torch.ones(M).long() * Dmax
        T = [Tmax] + torch.randint(3, Tmax//2, (M-1,)).long().numpy().tolist()
        #T = [Tmax] + [Tmax]*(M-1)  #torch.randint(3, Tmax, (M-1,)).long().numpy().tolist()

        timelines = {k: torch.sort(torch.rand(t).abs(), descending=False).values for i,(k,t) in enumerate(zip(names, T))}
        timelines = {k: tl/(i+1) + (torch.rand(1)-max(tl/(i+1)).item()) for i,(k,tl) in enumerate(timelines.items())}
        #timelines = {k: torch.arange(t, dtype=torch.float) for k,t in zip(names, T)}

        data = {k: torch.rand(N, t, d) for i, (k,t,d) in enumerate(zip(names, T, D))}

        y = [compute_target(data, timelines, d_out), None]
        X = prep_data(data, timelines, device=device)
    return X, timelines, y

    