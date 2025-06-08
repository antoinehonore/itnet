import torch


class TSdata:
    data: torch.Tensor
    timeline: torch.Tensor
    def __init__(self,data,timeline):
        self.data = data
        self.timeline = timeline
    def clone(self):
        return TSdata(self.data.clone(), self.timeline.clone())