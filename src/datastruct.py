import torch


class TSdata:
    data: torch.Tensor
    timeline: torch.Tensor
    def __init__(self,data,timeline):
        self.data = data
        self.timeline = timeline
