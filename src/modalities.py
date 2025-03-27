import torch
from src.kalman import Kalman

class LinearMeasurementModel(torch.nn.Module):
    def __init__(self,d,h,device="cpu"):
        super(LinearMeasurementModel, self).__init__()

        # Declare measurement model
        self.d = d
        self.h = h
        B = torch.randn(d, h, device=device)
        B = torch.linalg.qr(B).Q
        
        self.register_buffer("B", B)
        self.register_buffer("R_id", torch.eye(d, device=device))
        self.register_buffer("R", torch.eye(d, device=device))

    def sample(self, states, smnr_db=None):
        if not (smnr_db is None):
            Psignal = (states @ self.B.T).square().mean(1).mean() # 1/D *1/T *sum_d sum_t x_{d,t}^2
            noise_var = Psignal*(10**(-smnr_db/10))
            N,h = states.shape
            self.R = self.R_id* noise_var
        measurements = states @ self.B.T + torch.randn(N, self.d, device=states.device) @ self.R.sqrt().T
        return measurements

    def measure(self, s_t, smnr_db=None):
        if not (smnr_db is None):
            Psignal = (self.B.mv(s_t)).square().mean() # 1/D *sum_d x_{d}^2
            noise_var = Psignal*(10**(-smnr_db/10))
            N,h = states.shape
            self.R = self.R_id * noise_var
        measurement = self.B.mv(s_t) + self.R.sqrt().mv(torch.randn(self.d, device=s_t.device))
        return measurement


class Modality(torch.nn.Module):
    def __init__(self, d, h,measurementmodel=None):
        super(Modality, self).__init__()
        if measurementmodel is None:
            self.measurementmodel = LinearMeasurementModel(d,h)
        else:
            self.measurementmodel = measurementmodel
        self.h = h
        self.d = d
        self.update_time_init = [torch.tensor([0])]

    def sample(self,states, timeline_true, p, smnr_db):
        N = timeline_true.shape[0]
        Nmeasurements = int(N * p)

        timeline_idx = torch.randperm(N)[:Nmeasurements].sort().values.to(device=states.device)
        mask = torch.zeros_like(timeline_true,dtype=bool,device=states.device)
        mask[timeline_idx] = True
        self.mask = mask
        self.timeline = timeline_true[timeline_idx]
        self.timeline_idx = timeline_idx
        self.measurements = self.measurementmodel.sample(states[mask], smnr_db=smnr_db)
        return self.measurements   #####self.measurementmodel.draw(states[mask], smnr_db)
    
    def reset(self,):
        #self.filter.reset()
        device = self.measurementmodel.B.device
        #self.shat = [torch.zeros(self.h,device=device).unsqueeze(0)]
        #self.Pcov = [torch.eye(self.h,device=device).unsqueeze(0)]
        #self.scale_tril = [self.Pcov[0][0].diag().unsqueeze(0)]

        #self.update_time = [x.to(device) for x in self.update_time_init] ##[timeline_true[[0]]-1]
        #self.is_data = [True]
        #self.start_idx = 0
        self.measurements = None


class SensorSystem(torch.nn.Module):
    def __init__(self, hparams, device="cpu"):
        super(SensorSystem, self).__init__()
        self.hparams = hparams
        self.device = device
        self.h = hparams["h"]
        self.process_noise_var = 10**(hparams["process_noise_var_db"]/10)
        A = (torch.rand(self.h, self.h, device=self.device)).triu()#*torch.eye(self.h, device=self.device)
        A[0,1]*=A[0,0]
        self.register_buffer("A", A)
        self.register_buffer("Q", torch.eye(self.h, device=self.device) * self.process_noise_var)
        self.register_buffer("Q_sqrt",self.Q.sqrt())
        self.reset()

    def reset(self):
        self.register_buffer("s_0", torch.zeros(self.h,device=self.device))
        self.register_buffer("t_0", torch.tensor(0, device=self.device))

        self.register_buffer("current_state", torch.zeros(self.h,device=self.device))
        self.register_buffer("current_time", torch.tensor(0, device=self.device))

    def step(self, delta=1):
        self.current_state = self.A.mv(self.current_state) + self.Q_sqrt.mv(torch.randn(self.h, device=self.device))
        self.current_time = self.current_time + delta

    def measure(self, model):
        return model.measure(self.current_state)

    def sample(self, N):
        timeline = torch.arange(N, dtype=torch.float32, device=self.device) + self.current_time #.astype(float)
        states = torch.randn(N,self.h, device=self.device)       
        states[0] = self.current_state
        
        for t in range(1, N):
            self.step()
            states[t] = self.current_state
            timeline[t] = self.current_time

        return timeline, states

    def get_modalities(self, random_walk, timeline_true, M, P, SMNR_db, modalities_dimension, plot=False, Modalities=None):   
        """Get a dictionary of modality data.

            P, list: Percentage of available data for each modality
            SMNR_db, list: Measurement SMNRdb for each modality
            M, int: Number of modality
            A, tensor: process matrix
            Q, tensor: process noise covariance matrix
            d, int: dimension of modality
        """

        assert len(P) == M
        assert len(SMNR_db) == M
        
        N, h = random_walk.shape
        Filters = {}
        Modalities_ = {}
        
        for smnr_db, p, (modality_name,d) in zip(SMNR_db,P,modalities_dimension.items()):
            if (Modalities is None):
                themodality = Modality(d, h)
            else:
                themodality = Modality(d, h, measurementmodel=Modalities[modality_name].measurementmodel)
            themodality.to(self.device)
            themodality.reset()

            measurements = themodality.sample(random_walk, timeline_true, p, smnr_db)
            
            thekalman = Kalman(self.A, themodality.measurementmodel.B, self.Q, themodality.measurementmodel.R, self.s_0, timeline_true[0])

            # Save to dictionaries
            Filters[modality_name] = thekalman
            Modalities_[modality_name] = themodality
        
        if plot:
            fig, ax = plt.subplots()
            for im, m in enumerate(Modalities.keys()):
                ax.scatter(Modalities[m].timeline, torch.ones(Modalities[m].timeline.shape[0])*im, marker='o',s=50,zorder=10)
            ax.set_xlabel("Time")
            ax.set_ylabel("Modality")
            ax.set_yticks(list(range(M)),list(range(1,M+1)))
            better_lookin(ax)
        
        return Filters, Modalities_
    
    def get_samples_sensors(self, hparams, N=None, Filters=None,Modalities=None):
        if N is None:
            N = hparams["N"]
        timeline, states = self.sample(N)
        Filters, Modalities = self.get_modalities(states, timeline, 
                                    hparams["M"], hparams["P"], hparams["SMNR_db"], hparams["modalities_dimension"], Modalities=Modalities)
        return timeline, states, Filters, Modalities, hparams["modalities_dimension"]
    
    def get_local_predictions(self, timeline_predictions, Filters, Modalities):
        # Make predictions with kalman filters on each modalities
        for i, t in enumerate(timeline_predictions):
            for m in Filters.keys():
                update_modality_level_estimator(Filters[m], Modalities[m], t)
        
        batch = gather_prediction_data(Filters, Modalities, timeline_predictions)
        return batch

    def get_closed_form_fusion(self, batch):
        # Update global estimator
        ## Gather all previous state estimator statistics from the modalities
        Cov_inv_m = [torch.diag_embed((batch[m]["data_CF"][0, 0, 1:, self.h:-2])**(-1)) for m in batch.keys() if m != "reference"]
        inv_sum_Cov_inv_m = torch.linalg.inv(sum(Cov_inv_m))
        sum_Cinv_scov = sum([torch.einsum("nhh,nh->nh",C_inv, batch[m]["data_CF"][0,0, 1:, :self.h]) for m, C_inv in zip(batch.keys(),Cov_inv_m) if m != "reference"])
        fused_CF_mean_estimate = torch.einsum("nhh,nh->nh", inv_sum_Cov_inv_m, sum_Cinv_scov)
        return fused_CF_mean_estimate

def get_data(N, timeline_true, random_walk, M, P, SMNR_db, modalities_dimension):
    device = A.device
    h = A.shape[-1]
    t_0 = torch.tensor(0, device=device)
    s_0 = torch.zeros(h, device=device)
    
    # Get the true random process
    timeline_true = torch.arange(N, dtype=torch.float32,device=device)#.astype(float)

    random_walk = torch.randn(N, h,device=device)
    random_walk[0] = 0
    matrix_exp_A = A
    for t in range(1, N):
        random_walk[t] = matrix_exp_A.mv(random_walk[t-1]) + Q.sqrt().mv(torch.randn(h,device=device))
    
    # Create data modalities
    Filters, Modalities, Data = get_modalities(random_walk, timeline_true, A, Q, M, P, SMNR_db, modalities_dimension, s_0)
    return timeline_true, random_walk, Filters, Modalities, Data


def gather_prediction_data(Filters, Modalities, t):
    """Gather all the prediction data from all modalities"""
    calX = {}
    for m in Modalities.keys():
        # Where are the true measuremn
        is_data = Filters[m].tracking_data["is_data"]

        is_measurements = torch.tensor(is_data).view(-1,1)

        shat_measurements = torch.cat([shat for shat in Filters[m].tracking_data["state_estimates"]])
        scale_tril_measurements = torch.cat([shat for shat in Filters[m].tracking_data["scale_tril_estimates"]])
        update_time_measurements = torch.cat([t for t in Filters[m].tracking_data["update_time"]]).view(-1,1)
        
        data = torch.cat([  Modalities[m].measurements,
                            Modalities[m].timeline.diff(prepend=Modalities[m].timeline[[0]]).unsqueeze(-1),
                            Modalities[m].timeline.unsqueeze(-1),
                            #shat_measurements, 
                            #scale_tril_measurements,
                            #is_measurements.to(device=update_time_measurements.device),
                            #update_time_measurements
                        ], dim=1).unsqueeze(0).unsqueeze(0).detach()
        
        data_CF = torch.cat([  shat_measurements, 
                            scale_tril_measurements,
                            is_measurements.to(device=update_time_measurements.device),
                            update_time_measurements
                        ], dim=1).unsqueeze(0).unsqueeze(0).detach()

        calX[m] = {}
        calX[m]["data"] = data
        calX[m]["data_CF"] = data_CF
        calX[m]["timeline_idx"] = Modalities[m].timeline_idx#.unsqueeze(0)
        calX[m]["measurements"] = Modalities[m].measurements
        calX[m]["model"] = {"B": Modalities[m].measurementmodel.B, "R": Modalities[m].measurementmodel.R}

    calX["reference"] = {"data": t.unsqueeze(0).unsqueeze(0).unsqueeze(-1)}
    return calX

def multivar_gaussian_log_pdf(x, mu, cov):
    cov_diag = torch.diagonal(cov,dim1=-2,dim2=-1)
    cov_inv = torch.diag_embed((cov_diag**(-1)))#.diag()#torch.linalg.inv(cov)
    return -0.5*( cov_diag.log().sum(1) + torch.einsum("nh,nh->n",(x-mu),torch.einsum("nhh,nh->nh",cov_inv,(x-mu))) + x.shape[1]*torch.tensor(2*torch.pi).log() )

def compute_loss(batch, current_state_mean, current_state_covariance):
    """Compute the likelihood of the measurement data at time t"""

    log_likelihood = torch.tensor([[0.]], dtype=torch.float32, device=current_state_mean.device)

    for m in batch.keys():
        if m!= "reference":
            x = batch[m]["measurements"]
            x_idx = batch[m]["timeline_idx"]
            B = batch[m]["model"]["B"]
            R = batch[m]["model"]["R"]
            
            mu = (current_state_mean[x_idx]) @ B.T
            cov = R + B @ (current_state_covariance[x_idx] @ B.T)
            
            log_likelihood += multivar_gaussian_log_pdf(x, mu, cov).sum()
    return -log_likelihood

def tril2vec(L):
    return L[...,torch.tril(torch.ones_like(L)) == 1]

def update_modality_level_estimator(thekalman, modality, t, plot=False):
    """Perform state update on a modality."""
    device = thekalman.A.device
    timeline = modality.timeline 
    thedata = modality.measurements

    start_idx = thekalman.tracking_data["start_idx"]

    causal_mask = (timeline <= t)
    #thekalman = modality["filter"]
    measurements = None
    
    # Look for data between now and the 
    if causal_mask[start_idx:].any():
        end_idx = torch.where(causal_mask)[0][-1].item()+1
        assert (end_idx - start_idx) == 1, "Warning: NYI multiple points between this and previous time predictions"
        measurements = thedata[start_idx:end_idx]
        thetimeline = timeline[start_idx:end_idx]
        mask = None
        is_data = True

    else:
        end_idx = start_idx
        mask = torch.zeros(1,dtype=bool)
        thetimeline = torch.tensor([t],device=device)
        measurements = torch.zeros(1,thedata.shape[1],device=device)
        is_data = False

    _, _ = thekalman.track(measurements, thetimeline, mask=mask)

    thekalman.tracking_data["start_idx"] = end_idx
    thekalman.tracking_data["update_time"].append(thetimeline[[-1]])
    thekalman.tracking_data["is_data"].append(is_data)

    thekalman.tracking_data["state_estimates"].append(thekalman.current_state.unsqueeze(0))
    thekalman.tracking_data["cov_estimates"].append(thekalman.current_cov.unsqueeze(0))
    thekalman.tracking_data["scale_tril_estimates"].append(thekalman.current_cov.diag().unsqueeze(0))
    
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(8,6))
        thekalman.plot(axes, random_walk, measurements, shat, Pcov, timeline_true[mask], timeline_process=timeline_true)
        axes[0].legend()
        axes[0].set_title("Missing measurements {}%".format(int(round(100*(1-p)))))
        better_lookin(axes[0], fontsize=12, grid="off", legend=True, ncol=2)
        better_lookin(axes[1], fontsize=12, grid="off")

