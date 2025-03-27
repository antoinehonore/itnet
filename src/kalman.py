import torch

class Kalman(torch.nn.Module):
    def __init__(self, A, B, Q, R, s_0, t_0):
        """
        A: (...,h,h)
        B: (...,d,h)
        Q: (h,h)
        R: (d,d)
        s_0: (h,)
        """
        super(Kalman, self).__init__()
        self.register_buffer('s_0', s_0)
        self.register_buffer('t_0', t_0)
        self.register_buffer('A', A)
        self.register_buffer("B", B)
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)
        self.timeline = []
        self.deltas = []
        self.d, self.h = B.shape[-2:]
        self.register_buffer("Id", torch.eye(self.h,device=A.device))
        self.register_buffer("zero_gain", torch.zeros(A.shape[1],B.shape[0],device=A.device))
        self.register_buffer("dummy_observation", torch.zeros(B.shape[0],device=A.device))
        self.reset()

    def reset(self):
        self.register_buffer("current_prior_state",self.s_0)
        self.register_buffer("current_prior_cov", torch.eye(self.h,device=self.A.device))   ###s_0.reshape(-1,1) @ s_0.reshape(1,-1)

        self.register_buffer("current_state", self.s_0)
        self.register_buffer("current_cov", torch.eye(self.h,device=self.A.device))         ###s_0.reshape(-1,1) @ s_0.reshape(1,-1)
        
        self.register_buffer("prev_timestamp", self.t_0.unsqueeze(0))

        self.tracking_data = {  
                            "state_estimates":      [self.current_state.unsqueeze(0)],
                            "cov_estimates":        [self.current_cov.unsqueeze(0)], 
                            "scale_tril_estimates": [self.current_cov.diag().unsqueeze(0)], 
                            "update_time":          [self.t_0.unsqueeze(0)], 
                            "is_data":              [True],
                            "start_idx":            0
                            }

    def predict(self, delta=1):
        delta = 1
        self.matrix_exp = self.A# * delta)
        #if delta!=1:
        #    self.matrix_exp = torch.linalg.matrix_exp(self.A * delta)

        self.current_prior_state = self.matrix_exp.mv(self.current_state)
        self.current_prior_cov = self.matrix_exp @ self.current_cov  @ self.matrix_exp.T  + self.Q*delta  ####*self.deltas[-1]
        
        #self.current_prior_cov = self.current_prior_cov

    def update(self, x):
        if (x is None):
            self.kalmangain = self.zero_gain
        else:
            self.kalmangain = self.current_cov @ self.B.T @ torch.linalg.inv(self.B @ self.current_prior_cov @ self.B.T + self.R)
        
        self.current_cov = (self.Id - self.kalmangain @ self.B) @ self.current_prior_cov
        self.current_state = self.current_prior_state + (self.kalmangain.mv(x - self.B.mv(self.current_prior_state)) if not (x is None) else 0.)

    def track(self, X, timeline, mask=None):
        """If defined, mask is such that timeline[mask] corresponds to the timeline of the samples in X"""

        i_next_data = 0
        
        T = timeline.shape[0]
        
        for i in range(T):
            delta = timeline[i] - self.prev_timestamp
            
            self.predict(delta=delta)
            
            if (mask is None) or mask[i]:
                observation = X[i_next_data]
                i_next_data += 1
            else:
                # Observation leading to innnovation = 0, i.e. he state update evolves accroding to process
                observation = None
            
            self.update(observation)
            self.prev_timestamp = timeline[i]
        
        return None, None   ###self.Shat, self.Pcov
    
    def plot(self, ax, true_state, measurements, Shat, Pcov,timeline,timeline_process=None,idim=0): 
        ax.plot(timeline if timeline_process is None else timeline_process, true_state[:,idim],'.-',color="black", label="Ground truth")
        
        ax.plot(timeline, measurements[:,0],'.-',color="darkgreen", label="Measurements")
        ax.plot([timeline,timeline], [torch.ones_like(timeline)*measurements[:,idim].min(),measurements[:,idim]],'--',color="gray", label="_Measurements",lw=1)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Component 1")
        ymin,ymax=ax.get_ylim()
        ax.set_ylim([ymin,ymax*1.1])
        ax.plot(timeline_process, Shat[:,idim], '.-',color="darkred", label="Prediction")
        ax.fill_between(timeline_process, y1=Shat[:,idim]-Pcov[:,idim,idim].sqrt(),y2=Shat[:,0]+Pcov[:,idim,idim].sqrt(), alpha=0.5,color="darkred", label="Cov")
        return ax
    
    def forward(self,*args,**kwargs):
        return self.track(*args,**kwargs)

    def get_tracking_data(self,name):
        return torch.cat(self.tracking_data[name])[1:].cpu()
        
def scale_tril2cov(scale_tril, h):
    """Vector of tril coefficients to covariance matrix : P = L L.T """
    tril_indices = torch.tril_indices(row=h, col=h, offset=0)

    Pcov = torch.zeros(h,h)
    Pcov[tril_indices[0], tril_indices[1]] = scale_tril
    Pcov = Pcov @ Pcov.T
    return Pcov