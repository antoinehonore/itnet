import lightning as L
import torch
import matplotlib.pyplot as plt

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

        self.val_scores = {"y":[], "yhat":[]}
        self.train_scores = {"y":[], "yhat":[]}
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
        self.val_recon_figure       = plt.subplots(figsize=(5,4))
        self.val_senspec_figure     = plt.subplots(2,1,figsize=(12,6))
        self.train_senspec_figure   = plt.subplots(figsize=(5,3))

        self.val_attn_matrix        = None  #{k:plt.subplots(figsize=(10,6)) for k in model.fusion_model.estimate_fusion.attn_matrices.keys()}
        self.automatic_optimization = False
        self.the_training_step  = 0
        self.model = model

        self.val_scores =   {"y": [],   "yhat": [], "yclass":[], "norms": []}
        self.train_scores = {"y": [],   "yhat": [], "yclass":[], "norms": []}
        self.test_scores =  {"y": [],   "yhat": [], "yclass":[], "norms": []}

        self.cost_matrix = torch.tensor([[0,7,8,9,10], [200,0,7,8,9], [300,200,0,7,8], [400,300,200,0,7], [500,400,300,200,0]])
        
        self.compute_confmat = ConfusionMatrix(task="multiclass", num_classes=self.cost_matrix.shape[-1])
    def configure_model(self):
        if self.model is not None:
            return

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
        {"mse/val": torch.nan, "mse/train": torch.nan})

        
        self.train_scores = {"y": [],   "yhat": [], "yclass":[], "norms": []}
        self.test_scores =  {"y": [],   "yhat": [], "yclass":[], "norms": []}
        self.init_val_scores()

    def init_val_scores(self):
        self.val_scores =   [{"y": [],   "yhat": [], "yclass":[], "norms": []} for _ in range(2)]

    def compute_loss(self, batch):
        y = batch["targets_OH"]
        yclass = batch["targets_int"]
        yhat = self.model(batch)

        self.train_scores["y"].append(y.squeeze(0))
        self.train_scores["yclass"].append(yclass.squeeze(0))
        self.train_scores["yhat"].append(yhat.detach().squeeze(0))
        sample_weights = batch["class_weights"][0][yclass.long()].unsqueeze(-1)

        if self.loss_fun_name == "BCE":
            y_n = y
        elif self.loss_fun_name == "CE":
            sample_weights=sample_weights[0,:,0]
            yhat = yhat[0]
            y_n = yclass.long()[0]
        else:
            yhat = yhat[0]
            y_n = y.squeeze(0)
        
        loss = (self.loss_fun(yhat, y_n, reduction="none")*sample_weights).mean()  ###.squeeze(-1).T.long())
        return loss
    
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        opt = self.optimizers()
        loss = 0.0
        self.the_training_step += 1
        log_dict = {}
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        norms = self.model.itnet.MMA.norms

        if self.the_training_step % self.hparams["training"]["grad_step_every"]:
            opt.step()
            opt.zero_grad()
        
        self.log("{}/train".format(self.loss_fun_name), loss, on_epoch=False, batch_size=1, on_step=True)
        if batch_idx == 0:
            norms_mean = {"norm/"+k+"/train":norms[k].mean() for k in norms.keys()}
            self.log_dict(norms_mean, on_epoch=False,on_step=True,batch_size=1)

    def test_step(self,batch,batch_idx, dataloader_idx=0):
        if batch_idx ==0:
            self.test_scores = {"y": [],   "yhat": [], "yclass":[], "norms": []}

        yhat = self.model(batch)
        norms = self.model.fusion_model.estimate_fusion.norms
        yclass = None
        y = None
        if "targets_int" in batch.keys():
            yclass = batch["targets_int"]
            self.test_scores["yclass"].append(yclass.squeeze(0))

        if "targets_OH" in batch.keys():
            y = batch["targets_OH"]
            self.test_scores["y"].append(y.squeeze(0))

        self.test_scores["yhat"].append(yhat.detach().squeeze(0))
        self.test_scores["norms"].append(norms)
    
    def on_test_epoch_end(self):
        scores = {}
        if len(self.test_scores["yclass"]) >0:
            yhat = torch.cat(self.test_scores["yhat"]).squeeze(-1)
            yclass = torch.cat(self.test_scores["yclass"]).squeeze(-1)
            y = torch.eye(yhat.shape[-1], device=yhat.device)[yclass.long()]
            scores = self.get_scores(y, yhat, yclass, suffix="/test")

        self.test_scores = {"y": [],   "yhat": [], "yclass":[], "norms":[]}
        return scores

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        yclass = batch["targets_int"]
        yhat = self.model(batch)
        norms = self.model.itnet.MMA.norms
        
        if not ("targets_OH" in batch.keys()):
            y = torch.eye(yhat.shape[-1], device=yhat.device)[yclass.long()]
        else:
            y = batch["targets_OH"]
        
        if (batch_idx == 0) and (self.logger is not None):
            norms_mean = {"norm/"+k+"/val{}".format(dataloader_idx):norms[k].mean() for k in norms.keys()}

            self.log_dict(norms_mean, on_epoch=False,on_step=True,batch_size=1)

            timeline = batch["data"]["reference"][batch_idx].cpu().numpy()
            fig, axes = self.val_senspec_figure
            ax = axes[0]
            ax.cla()
            plot_data = torch.cat(list(norms.values()),dim=-1)[batch_idx, 0].cpu().float().numpy()
            labels = list(norms.keys())
            for j in range(plot_data.shape[1]):
                ax.plot(timeline, plot_data[:,j], label=labels[j],**color_marker_style[j])
            ax.legend(bbox_to_anchor=(1,1))
            ax.set_xlabel("Time")
            ax.set_ylabel("Modality contribution (%)")
            ax = axes[1]
            ax.cla()
            ax.plot(timeline, yhat[batch_idx].argmax(-1).cpu().float().numpy(),label="Predicted ",marker="x")
            ax.plot(timeline, yclass[batch_idx].cpu().float().numpy(),label="True ",marker="o")
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Class index")
            self.logger.experiment.add_figure("mod_contributions/val{}".format(dataloader_idx), fig, self.the_training_step)

        self.val_scores[dataloader_idx]["y"].append(y.squeeze(0))
        self.val_scores[dataloader_idx]["yclass"].append(yclass.squeeze(0))
        self.val_scores[dataloader_idx]["yhat"].append(yhat.squeeze(0))
    
    def get_scores(self, y, yhat, yclass, suffix=""):
        yhat = yhat.to(torch.float)
        y = y.to(torch.float)
        yclass = yclass.long()
        yhat_sigmoid = torch.nn.functional.sigmoid(yhat)
        yhat_softmax = torch.nn.functional.sigmoid(yhat)
        
        thescores = {"mse" + suffix: torchmetrics.functional.mean_squared_error(yhat_softmax, y)}
        thescores["BCE" + suffix] = torch.nn.functional.binary_cross_entropy_with_logits(yhat, y)
        thescores["CE" + suffix] = torch.nn.functional.cross_entropy(yhat, yclass.long())
        thescores["Acc"+suffix] = multiclass_accuracy(yhat, yclass, average="micro")
        thescores["F1score"+suffix] = multiclass_f1_score(yhat, yclass, average="micro", num_classes=yhat.shape[-1])
        thescores["Prec"+suffix] = multiclass_precision(yhat, yclass, average="micro", num_classes=yhat.shape[-1])
        thescores["Recall"+suffix] = multiclass_recall(yhat, yclass, average="micro", num_classes=yhat.shape[-1])
        thescores["AUROC"+suffix] = multiclass_auroc(yhat, yclass, num_classes=yhat.shape[-1])
        thescores["AUPRC"+suffix] = multiclass_auprc(yhat, yclass, num_classes=yhat.shape[-1])
        cm = self.compute_confmat(yhat, yclass)
        thescores["cost" + suffix] = (self.cost_matrix.to(device=cm.device) * cm).sum() / cm.sum()

        thescores["topk2/exact"+ suffix] =   topk_multilabel_accuracy(yhat, y, criteria="exact_match", k=2)
        thescores["topk2/hamming"+ suffix] = topk_multilabel_accuracy(yhat, y, criteria="hamming", k=2)
        thescores["topk2/overlap"+ suffix] = topk_multilabel_accuracy(yhat, y, criteria="overlap", k=2)
        thescores["topk2/contain"+ suffix] = topk_multilabel_accuracy(yhat, y, criteria="contain", k=2)
        thescores["topk2/belong"+ suffix] =  topk_multilabel_accuracy(yhat, y, criteria="belong", k=2)
        return thescores
        
    def on_train_epoch_end(self):
        y = torch.cat(self.train_scores["y"]).squeeze(-1)
        yhat = torch.cat(self.train_scores["yhat"]).squeeze(-1)
        yclass = torch.cat(self.train_scores["yclass"]).squeeze(-1)

        scores = self.get_scores(y, yhat, yclass, suffix="/train")
        i = 0
        ax = self.train_recon_figure[1]
        ax.cla()
        plot_confusion_matrix(ax, yclass.cpu(), yhat.argmax(1).cpu(), normalize=True, num_classes=yhat.shape[-1])
        if self.logger is not None:
            self.logger.experiment.add_figure("recon_figure/train", self.train_recon_figure[0], self.the_training_step)
        
        self.log_dict(scores, on_epoch=True,on_step=False,batch_size=1)
        self.train_scores = {"y": [],   "yhat": [], "yclass":[], "norms": []}

    def on_validation_epoch_end(self):
        for dataloader_idx in range(len(self.val_scores)):
            y = torch.cat(self.val_scores[dataloader_idx]["y"]).squeeze(-1)
            yhat = torch.cat(self.val_scores[dataloader_idx]["yhat"]).squeeze(-1)
            yclass = torch.cat(self.val_scores[dataloader_idx]["yclass"]).squeeze(-1)

            scores = self.get_scores(y, yhat, yclass, suffix="/val{}".format(dataloader_idx))
            self.log_dict(scores, on_epoch=True,on_step=False,batch_size=1)

            i = 0
            ax = self.val_recon_figure[1]
            ax.cla()
            class_names = [">48", "48-24", "24-12", "12-6", "<6"]
            plot_confusion_matrix(ax, yclass.cpu(), yhat.argmax(1).cpu(), normalize=dataloader_idx==0, num_classes=yhat.shape[1], class_names=class_names)
            if self.logger is not None:
                self.logger.experiment.add_figure("recon_figure/val{}".format(dataloader_idx), self.val_recon_figure[0], self.the_training_step)

        self.init_val_scores()
        return scores

    def configure_optimizers(self):
        optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], 
                lr=self.hparams["training"]['lr'])
        return optim


def plot_confusion_matrix(ax, y_true, y_pred, class_names=None, normalize=False, cmap="Blues", num_classes=5):
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
    cm_n = (100*cm.astype('float') / cm.sum(axis=1, keepdims=True)).round(2)

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
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}\n({int(cm_n[i, j])})", ha='center', va='center', color='black')
    
    return ax
