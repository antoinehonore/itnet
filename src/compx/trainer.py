import lightning as L
import torch
import matplotlib.pyplot as plt

import torchmetrics

from torchmetrics.classification import BinaryStatScores
from torchmetrics import ConfusionMatrix
from torcheval.metrics.functional import binary_auprc
import numpy as np
from torcheval.metrics.functional.classification import topk_multilabel_accuracy


class lTrainer(L.LightningModule):
    def __init__(self, hparams=None, model=None):
        super(lTrainer, self).__init__()
        self.model_params=hparams["model"]
        self.save_hyperparameters(hparams)

        self.val_scores = {"y":[],"yhat":[]}
        self.train_scores = {"y":[],"yhat":[]}
        self.loss_fun_name = hparams["training"]["loss"] 

        if self.loss_fun_name == "BCE":
            self.loss_fun = torch.nn.functional.binary_cross_entropy_with_logits#torch.nn.functional.cross_entropy
        
        elif self.loss_fun_name == "MSE":
            self.loss_fun = torch.nn.functional.mse_loss

        self.train_recon_figure     = plt.subplots(figsize=(10,6))
        self.val_recon_figure       = plt.subplots(figsize=(10,6))
        self.val_senspec_figure     = plt.subplots(figsize=(5,3))
        self.train_senspec_figure   = plt.subplots(figsize=(5,3))

        self.val_attn_matrix        = None  #{k:plt.subplots(figsize=(10,6)) for k in model.fusion_model.estimate_fusion.attn_matrices.keys()}
        self.automatic_optimization = False
        self.the_training_step  = 0
        self.model = model

    def configure_model(self):
        if self.model is not None:
            return

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
        {"mse/val": torch.nan, "mse/train": torch.nan})

        self.val_scores =   {"y": [],   "yhat": []}
        self.train_scores = {"y": [],   "yhat": []}
    
    def compute_loss(self,batch):
        y = batch["targets"]
        yhat = self.model(batch)

        self.train_scores["y"].append(y.squeeze(0))
        self.train_scores["yhat"].append(yhat.detach().squeeze(0))
        if self.loss_fun_name == "BCE":
            sample_weights = 1 
            y_n = y
        else:
            yhat = yhat[0]
            sample_weights = 1
            y_n = y.squeeze(0)

        loss = (self.loss_fun(yhat, y_n, reduction="none")*sample_weights).mean()#.squeeze(-1).T.long())
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        opt = self.optimizers()
        loss = 0.0
        self.the_training_step += 1
        log_dict = {}
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        
        if self.the_training_step % self.hparams["training"]["grad_step_every"]:
            opt.step()
            opt.zero_grad()
        self.log("{}/train".format(self.loss_fun_name), loss, on_epoch=False, batch_size=1, on_step=True)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y = batch["targets"]
        y_n = y
        
        yhat = self.model(batch)
        if batch_idx == 0 and (self.logger is not None):
            if not (self.val_attn_matrix is None):
                for modality_name,(fig,ax) in self.val_attn_matrix.items():
                    A = self.model.fusion_model.estimate_fusion.attn_matrices[modality_name]
                    if A is not None:
                        ax.cla()
                        ax.imshow(A[0,0].T, aspect="equal")
                        ax.set_title("{}: $\\tau={:.03f}$".format(modality_name, self.model.fusion_model.estimate_fusion.history_temperature[modality_name].item()))
                        ax.set_ylabel("Measurements timeline")
                        ax.invert_yaxis()
                        ax.set_xlabel("Predictions timeline")
                        self.logger.experiment.add_figure("attn_figure/{}/val".format(modality_name), fig, self.the_training_step)

        self.val_scores["y"].append(y.squeeze(0))
        self.val_scores["yhat"].append(yhat.detach().squeeze(0))
    
    def get_scores(self, y, yhat, suffix=""):
        thescores = {"mse" + suffix: torchmetrics.functional.mean_squared_error(yhat, y)    }
        thescores["BCE"+ suffix] = torch.nn.functional.binary_cross_entropy_with_logits(yhat,y)
        thescores["topk2/exact"+ suffix] =   topk_multilabel_accuracy(yhat, y, criteria="exact_match", k=2)
        thescores["topk2/hamming"+ suffix] = topk_multilabel_accuracy(yhat, y, criteria="hamming", k=2)
        thescores["topk2/overlap"+ suffix] = topk_multilabel_accuracy(yhat, y, criteria="overlap", k=2)
        thescores["topk2/contain"+ suffix] = topk_multilabel_accuracy(yhat, y, criteria="contain", k=2)
        thescores["topk2/belong"+ suffix] =  topk_multilabel_accuracy(yhat, y, criteria="belong", k=2)

        return thescores
        
    def on_train_epoch_end(self):
        y = torch.cat(self.train_scores["y"]).squeeze(-1)
        yhat = torch.cat(self.train_scores["yhat"]).squeeze(-1)
        scores = self.get_scores(y, yhat, suffix="/train")

        self.log_dict(scores,on_epoch=True,on_step=False,batch_size=1)
        self.train_scores = {"y": [], "yhat": []}

    def on_validation_epoch_end(self):
        y = torch.cat(self.val_scores["y"]).squeeze(-1)
        yhat = torch.cat(self.val_scores["yhat"]).squeeze(-1)
        
        scores = self.get_scores(y, yhat, suffix="/val")

        self.val_scores = {"y": [], "yhat": []}

    def configure_optimizers(self):
        optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], 
                lr=self.hparams["training"]['lr'])
        return optim




def plot_confusion_matrix(ax, y_true, y_pred, class_names=None, normalize=False, cmap="Blues"):
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
        class_names=["0","1"]
    
    # Compute confusion matrix
    num_classes = len(class_names)
    confmat = ConfusionMatrix(task="binary", num_classes=num_classes)
    cm = confmat(y_pred, y_true).cpu().numpy()
    
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
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}", ha='center', va='center', color='black')
    
    return ax
