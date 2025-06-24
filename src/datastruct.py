import torch

class TSdata:
    data: torch.Tensor
    timeline: torch.Tensor
    def __init__(self, data, timeline, idx):
        self.data = data
        self.idx = idx
        #self.tg_data = Data(x=data, edge_index=get_ts_edge_index(data.shape[0]),edge_attr=timeline)
        self.timeline = timeline
    def clone(self):
        return TSdata(self.data.clone(), self.timeline.clone(), self.idx.clone())