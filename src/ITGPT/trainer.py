import lightning as L
import torch
import matplotlib.pyplot as plt
import pandas as pd

import torchmetrics

from torchmetrics.classification import BinaryStatScores
from torchmetrics import ConfusionMatrix
from torcheval.metrics.functional import binary_auprc
import numpy as np
from torcheval.metrics.functional.classification import topk_multilabel_accuracy
from torcheval.metrics.functional import multiclass_accuracy,multiclass_f1_score,multiclass_precision, multiclass_recall,multiclass_auprc,multiclass_auroc

# Matplotlib unique combinations of color (colorblind friendly), marker and styles, from ChatGPT
color_marker_style = [
    {"color": "#E69F00", "marker": "o", "linestyle": "-"},   # orange
    {"color": "#56B4E9", "marker": "s", "linestyle": "--"},  # sky blue
    {"color": "#009E73", "marker": "^", "linestyle": "-."},  # bluish green
    {"color": "#F0E442", "marker": "D", "linestyle": ":"},   # yellow
    {"color": "#0072B2", "marker": "*", "linestyle": "-"},   # blue
    {"color": "#D55E00", "marker": "x", "linestyle": "--"},  # vermillion
    {"color": "#CC79A7", "marker": "+", "linestyle": "-."},  # reddish purple
    {"color": "#999999", "marker": "v", "linestyle": ":"},   # grey
    {"color": "#117733", "marker": "o", "linestyle": "--"},  # dark green
    {"color": "#88CCEE", "marker": "s", "linestyle": "-"},   # light blue
    {"color": "#332288", "marker": "^", "linestyle": ":"},   # dark purple
    {"color": "#44AA99", "marker": "D", "linestyle": "-."},  # teal
    {"color": "#DDCC77", "marker": "*", "linestyle": "--"},  # sand
    {"color": "#AA4499", "marker": "x", "linestyle": "-"},   # purple
    {"color": "#882255", "marker": "+", "linestyle": ":"},   # wine
    {"color": "#661100", "marker": "v", "linestyle": "-."},  # dark brown
    {"color": "#999933", "marker": "o", "linestyle": "--"},  # olive
    {"color": "#6699CC", "marker": "s", "linestyle": ":"},   # steel blue
    {"color": "#888888", "marker": "^", "linestyle": "-"},   # mid gray
    {"color": "#F28500", "marker": "D", "linestyle": "--"},  # vivid orange
]

class lTrainer(L.LightningModule):
    def __init__(self, hparams=None, model=None):
        super(lTrainer, self).__init__()
        self.model_params = hparams["model"]
        self.save_hyperparameters(hparams)

        self.loss_fun_name = hparams["training"]["loss"] 

        if self.loss_fun_name == "BCE":
            self.loss_fun = torch.nn.functional.binary_cross_entropy_with_logits#torch.nn.functional.cross_entropy
        
        elif self.loss_fun_name == "CE":
            self.loss_fun = torch.nn.functional.cross_entropy#torch.nn.functional.cross_entropy
        
        elif self.loss_fun_name == "MSE":
            self.loss_fun = torch.nn.functional.mse_loss
        else:
            raise Exception("Loss function "+self.loss_fun_name+" is NYI.")

        self.train_recon_figure     = plt.subplots(figsize=(5,4))
        self.val_recon_figure       = [plt.subplots(figsize=(5,4)) for _ in range(2)]
        self.val_senspec_figure     = plt.subplots(2,1,figsize=(12,6))
        self.train_senspec_figure   = plt.subplots(figsize=(5,3))

        self.val_attn_matrix        = None  #{k:plt.subplots(figsize=(10,6)) for k in model.fusion_model.estimate_fusion.attn_matrices.keys()}
        self.automatic_optimization = False
        self.the_training_step  = 0
        self.model = model

        self.init_val_scores()
        self.train_scores = self.init_dict()#{"y": [],   "logits": [], "yclass":[], "norms": []}
        self.test_scores = self.init_dict() # {"y": [],   "logits": [], "yclass":[], "norms": []}
        
        self.cost_matrix = torch.tensor([[0,7,8,9,10,0], [200,0,7,8,9,0], [300,200,0,7,8,0], [400,300,200,0,7,0], [500,400,300,200,0,0],[0,0,0,0,0,0]])
        self.class_names = [">48", "48-24", "24-12", "12-6", "<6", "U"]

        self.compute_confmat = ConfusionMatrix(task="multiclass", num_classes=self.cost_matrix.shape[-1])
    
    def configure_model(self):
        if self.model is not None:
            return

    def configure_optimizers(self):
        optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], 
                lr=self.hparams["training"]['lr'])
        return optim

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
        {"mse/val": torch.nan, "mse/train": torch.nan})

        #def compute_norm(the_loader):
        #    all_variable_names = [[k for k in list(batch["data"].keys()) if (k!="specs" and k!="reference")] for batch in the_subset][0]
        #    all_data = {k: torch.cat([batch["data"][k] for batch in the_subset]) for k in all_variable_names}
        #    normalization = {k:{"mu": var_data.mean(0).reshape(1,-1),"sigma":var_data.std(0).reshape(1,-1)} for k,var_data in all_data.items()}
        #    return normalization

        self.train_scores = self.init_dict()#{"y": [],   "logits": [], "yclass":[], "norms": []}
        self.test_scores =  self.init_dict()#{"y": [],   "logits": [], "yclass":[], "norms": []}
        self.init_val_scores()

    def init_val_scores(self):
        self.val_scores =   [self.init_dict() for _ in range(2)]

    def init_dict(self):
        return {"y": [],   "logits": [], "yclass":[], "xhat_var":[]}
    
    def on_train_epoch_start(self,):
        self.model.itgpt.reset_running_slopes()

    def compute_loss(self, batch):
        yclass = batch["targets_int"]
        _, logits = self.model(batch)

        if not ("targets_OH" in batch.keys()):
            y = torch.eye(logits.shape[-1], device=logits.device)[yclass.long()]
        else:
            y = batch["targets_OH"]
        self.train_scores["y"].append(y.squeeze(0))
        self.train_scores["yclass"].append(yclass.squeeze(0))
        self.train_scores["logits"].append(logits.detach().squeeze(0))
        sample_weights = batch["class_weights"][0][yclass.long()].unsqueeze(-1)

        if self.loss_fun_name == "BCE":
            y_n = y
        elif self.loss_fun_name == "CE":
            sample_weights =sample_weights[0,:,0]
            logits = logits[0]
            y_n = yclass.long()[0]
        else:
            logits = logits[0]
            y_n = y.squeeze(0)
        
        loss = (self.loss_fun(logits, y_n, reduction="none")*sample_weights).mean()  ###.squeeze(-1).T.long())
        return loss
    
    def get_scores(self, y, logits, yclass, suffix=""):
        logits = logits.to(torch.float)
        y = y.to(torch.float)
        yclass = yclass.long()
        yhat_sigmoid = torch.nn.functional.sigmoid(logits)
        yhat_softmax = torch.nn.functional.softmax(logits, dim=-1)
        
        thescores = {"mse" + suffix:  torchmetrics.functional.mean_squared_error(yhat_softmax, y)}
        thescores["BCE" + suffix] =   torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        thescores["CE" + suffix] =    torch.nn.functional.cross_entropy(logits, yclass.long())
        thescores["Acc"+suffix] =     multiclass_accuracy(logits, yclass, average="micro")
        thescores["F1score"+suffix] = multiclass_f1_score(logits, yclass, average="micro", num_classes=logits.shape[-1])
        thescores["Prec"+suffix] =    multiclass_precision(logits, yclass, average="micro", num_classes=logits.shape[-1])
        thescores["Recall"+suffix] =  multiclass_recall(logits, yclass, average="micro", num_classes=logits.shape[-1])
        thescores["AUROC"+suffix] =   multiclass_auroc(logits, yclass, num_classes=logits.shape[-1])
        thescores["AUPRC"+suffix] =   multiclass_auprc(logits, yclass, num_classes=logits.shape[-1])
        
        cm = self.compute_confmat(yhat_softmax, yclass)
        
        thescores["cost" + suffix] = (self.cost_matrix.to(device=cm.device) * cm).sum() / cm.sum()

        thescores["topk2/exact"+ suffix] =   topk_multilabel_accuracy(logits, y, criteria="exact_match", k=2)
        thescores["topk2/hamming"+ suffix] = topk_multilabel_accuracy(logits, y, criteria="hamming", k=2)
        thescores["topk2/overlap"+ suffix] = topk_multilabel_accuracy(logits, y, criteria="overlap", k=2)
        thescores["topk2/contain"+ suffix] = topk_multilabel_accuracy(logits, y, criteria="contain", k=2)
        thescores["topk2/belong"+ suffix] =  topk_multilabel_accuracy(logits, y, criteria="belong", k=2)
        
        return thescores
        

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        opt = self.optimizers()
        self.the_training_step += 1
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        
        #if loss > 2:
        #    print()

        if self.the_training_step % self.hparams["training"]["grad_step_every"]:
            opt.step()
            opt.zero_grad()

        self.log("{}/train".format(self.loss_fun_name), loss, on_epoch=False, batch_size=1, on_step=True)

    def on_train_epoch_end(self):
        y = torch.cat(self.train_scores["y"]).squeeze(-1)
        logits = torch.cat(self.train_scores["logits"]).squeeze(-1)
        yclass = torch.cat(self.train_scores["yclass"]).squeeze(-1)

        scores = self.get_scores(y, logits, yclass, suffix="/train")
        i = 0
        ax = self.train_recon_figure[1]
        ax.cla()
        plot_confusion_matrix(ax, yclass.cpu(), logits.argmax(1).cpu(), normalize=True, num_classes=logits.shape[-1], class_names=self.class_names)
        if self.logger is not None:
            self.logger.experiment.add_figure("recon_figure/train", self.train_recon_figure[0], self.the_training_step)
        
        self.log_dict(scores, on_epoch=True,on_step=False,batch_size=1)
        self.train_scores = self.init_dict()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        yclass = batch["targets_int"]
        max_values = {k: (v[...,:-2]+1).log().var().item() for k,v in batch["data"].items() if (k!= "specs") and (k!="reference")}
        xhat, logits = self.model(batch)
        xhat_var = {k: v.data.var(-1).mean(-1).item() for k,v in xhat.items() if (k!= "specs") and (k!="reference")}

        if not ("targets_OH" in batch.keys()):
            y = torch.eye(logits.shape[-1], device=logits.device)[yclass.long()]
        else:
            y = batch["targets_OH"]
        
        self.val_scores[dataloader_idx]["y"].append(y.squeeze(0))
        self.val_scores[dataloader_idx]["yclass"].append(yclass.squeeze(0))
        self.val_scores[dataloader_idx]["logits"].append(logits.squeeze(0))
        self.val_scores[dataloader_idx]["xhat_var"].append(xhat_var)
       
    def on_validation_epoch_end(self):
        for dataloader_idx in range(len(self.val_scores)):
            if len(self.val_scores[dataloader_idx]["y"])>0:
                y = torch.cat(self.val_scores[dataloader_idx]["y"]).squeeze(-1)
                logits = torch.cat(self.val_scores[dataloader_idx]["logits"]).squeeze(-1)
                yclass = torch.cat(self.val_scores[dataloader_idx]["yclass"]).squeeze(-1)
                suffix = "/val{}".format(dataloader_idx)
                dict_xhat_var = pd.DataFrame(self.val_scores[dataloader_idx]["xhat_var"]).mean(0).to_dict()
                dict_xhat_var = {"xhat_var/"+k+suffix: v for k,v in dict_xhat_var.items()}

                logits_var = logits.var(1).mean(0)
                dict_xhat_var["logits_var"+suffix] = logits_var

                scores = self.get_scores(y, logits, yclass, suffix=suffix)
                self.log_dict({**scores, **dict_xhat_var}, on_epoch=True, on_step=False)

                ax = self.val_recon_figure[dataloader_idx][1]
                ax.cla()
                plot_confusion_matrix(ax, yclass.cpu(), logits.argmax(1).cpu(), normalize=dataloader_idx==0, num_classes=logits.shape[1], class_names=self.class_names)
                if self.logger is not None:
                    self.logger.experiment.add_figure("recon_figure/val{}".format(dataloader_idx), self.val_recon_figure[dataloader_idx][0], self.the_training_step)

        self.init_val_scores()
        return scores
     
    def test_step(self,batch,batch_idx, dataloader_idx=0):
        if batch_idx ==0:
            self.test_scores = self.init_dict()#{"y": [],   "logits": [], "yclass":[], "xhat_var": []}

        _, logits = self.model(batch)
        yclass = None
        y = None
        if "targets_int" in batch.keys():
            yclass = batch["targets_int"]
            self.test_scores["yclass"].append(yclass.squeeze(0))

        if "targets_OH" in batch.keys():
            y = batch["targets_OH"]
            self.test_scores["y"].append(y.squeeze(0))

        self.test_scores["logits"].append(logits.detach().squeeze(0))
    
    def on_test_epoch_end(self):
        scores = {}
        logits = torch.cat(self.test_scores["logits"]).squeeze(-1)

        if len(self.test_scores["yclass"]) >0:
            yclass = torch.cat(self.test_scores["yclass"]).squeeze(-1)
            y = torch.eye(logits.shape[-1], device=logits.device)[yclass.long()]
            scores = self.get_scores(y, logits, yclass, suffix="/test")
        else:
            scores["logits"] = logits

        self.test_scores = self.init_dict()
        return scores

def plot_timeline_contribution(axes,timeline,norms,logits,yclass,batch_idx):
    ax = axes[0]
    ax.cla()
    plot_data = torch.cat(list(norms.values()),dim=-1)[batch_idx, 0].cpu().float().numpy()
    labels = list(norms.keys())
    if len(timeline)>1:
        for j in range(plot_data.shape[1]):
            ax.plot(timeline, plot_data[:,j], label=labels[j],**color_marker_style[j])
    else:
        idx = np.argsort(plot_data[0])[::-1]
        ax.bar(np.array(labels)[idx], plot_data[0][idx])
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Modality contribution (%)")
    ax = axes[1]
    ax.cla()
    ax.plot(timeline, logits[batch_idx].argmax(-1).cpu().float().numpy(),label="Predicted ",marker="x")
    ax.plot(timeline, yclass[batch_idx].cpu().float().numpy(),label="True ",marker="o")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Class index")
    return axes


def plot_confusion_matrix(ax, y_true, y_pred, class_names=None, normalize=False, cmap="Blues", num_classes=6):
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
        class_names = [str(i) for i in range(num_classes)]
    
    # Compute confusion matrix
    num_classes = len(class_names)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    cm = confmat(y_pred, y_true).cpu().numpy()
    cm_n = (cm.astype('float') / cm.sum(axis=1, keepdims=True))

    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot using matplotlib
    cax = ax.matshow(cm_n, cmap=cmap)

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
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}\n({cm_n[i, j]:.2f})", ha='center', va='center', color='black')
    
    return ax
