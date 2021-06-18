# helper function to show an image
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import plotly.figure_factory as ff 
from abc import ABC
import typing
import dataclasses
from torch.utils.tensorboard import SummaryWriter


def matplotlib_imshow(img, one_channel=False, normalized=False):
    """plot image tensors in matplotlib"""
    if one_channel:
        img = img.mean(dim=0)
    if normalized:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_confusion_matrix(conf_mat, label_names, height=500, width=500):
    fig = ff.create_annotated_heatmap(conf_mat, x=label_names, y=label_names)
    fig.update_layout(yaxis = dict(categoryorder = 'category descending'))
    fig.update_layout(xaxis = dict(categoryorder = 'category ascending'))
    fig.update_xaxes(title_text='Predicted Class')
    fig.update_yaxes(title_text='True Class')
    fig.update_layout(
        autosize=False,
        width=width,
        height=height)
    return fig

def list_file_paths(dirs):
    """list full file paths from a directory"""
    all_files_paths = []
    for dir in dirs:
        files = os.listdir(dir)
        files_path = [os.path.join(dir, f) for f in files]
        all_files_paths += files_path

    return sorted(all_files_paths)

def freeze(module: torch.nn.Module):
    """Freeze model parameters"""
    # module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
 
def unfreeze(module: torch.nn.Module):
    """Unfreeze model parameters"""
    # module.train()
    for param in module.parameters():
        param.requires_grad_(True)

def accuracy(y_pred, y):
  return torch.sum(y == y_pred) / len(y)

class CoreSlices (torch.utils.data.Dataset):
    """Dataset class for well core image slices"""
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):

        img = self.transform(Image.open(self.imgs[idx]))

        #get the root name of the file (no file extension) and extract the label
        #label = int(os.path.splitext(os.path.basename(self.imgs[idx]))[0].split("_")[-1])
        label = self.get_label(idx)
        return img, torch.tensor(label)

    def get_label(self, idx):
        """Method to allow retrieval of label without having to download the image when it is remote"""
        return int(os.path.splitext(os.path.basename(self.imgs[idx]))[0].split("_")[-1])

    def get_mid_point(self,idx):
        """Returns the depth of the mid-point of the slice"""
        depth_start, depth_end = os.path.splitext(os.path.basename(self.imgs[idx]))[0].split("_")[3:5]
        depth_start = float(depth_start)/10 #/10 to convert to mm
        depth_end = float(depth_end)/10
        return (depth_end + depth_start) / 2

def get_weights(modes, slices):
    labels = {}
    labels_cnt = {}
    labels_weights = {}
    weights = {}
    for mode in modes:
        labels[mode] = []

        # get all labels
        for i in range(len(slices[mode])):
            labels[mode].append(slices[mode].get_label(i))
        
        labels[mode] = np.sort(labels[mode]) #sort so that counts and weights are in the right index
        labels_cnt[mode] = np.bincount(labels[mode]) #label frequency
        #labels_cnt[mode] = labels_cnt[mode] / min(labels_cnt[mode]) #normalise, don't need to do this
        labels_weights[mode] = 1. / labels_cnt[mode]

        #create list with a weight for each sample
        weights[mode] = []
        for i in range(len(slices[mode])):
            label = slices[mode].get_label(i)
            weights[mode].append(labels_weights[mode][label]) 

        weights[mode] = torch.tensor(weights[mode])
    
    return weights

@dataclasses.dataclass
class Runner():
    model: torch.nn.Module
    optimizer: torch.nn.optim
    criterion: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    device: torch.device
    summarywrite=None: SummaryWriter
    epoch_scheduler=None: torch.optim.lr_scheduler
    batch_scheduler=None: torch.optim.lr_scheduler
    save_path = None: str

    def __init__(self):
        self.model = self.model.to(device)
        self.train_steps = 0
        self.val_steps = 0
        self.best_accuracy = 0

        #TODO implement the class based metrics used int the AiCourse notebooks
        self.metrics = {
            "train":{
                "loss":0,
                "accuracy":0
            },
            "val":{
                "loss":0,
                "accuracy":0
            }
        }

    def predict(self, outputs):
        return torch.argmax(outputs, dim=1)

    def predict_proba(self, outputs):
        return torch.nn.functional.softmax(outputs, dim=1)

    def reset_metrics(self, mode:str):
        self.metrics[mode]["accuracy"] = 0

    def train(self, dataloader, epoch):
        self.reset_metrics("train")
        #switch to train mode
        self.model.train()
        step = 0
        
        #loop over each sample
        for X,y in dataloader:
            step+=1
            #self.train_steps+=1

            X,y = X.to(self.device), y.to(self.device)
            outputs = self.model.forward(X)

            # back prop
            loss = self.criterion(outputs, y)
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            y_pred = self.predict(outputs)
            #step_accuracy = accuracy(y_pred, y)
            #self.metrics["train"]["accuracy"] += step_accuracy

            #run scheduler per step
            if self.batch_scheduler:
                self.batch_scheduler.step()
            
            #self.metrics["train"]["loss"] += loss

            #output to tensorboard
            if self.summarywriter:
                self.metrics_to_summarywriter(self)
                # self.summarywriter.add_scalar("batch_loss/training", loss, self.train_steps)
                # self.summarywriter.add_scalar("batch_accuracy/training", step_accuracy, self.train_steps)

            yield step, loss

        #calculate final metrics 
        self.metrics["train"]["loss"] /= step
        self.metrics["train"]["accuracy"] /= step

        #output to tensorboard
        if self.summarywriter:
            self.summarywriter.add_scalar("epoch_loss/training", self.metrics["train"]["loss"], epoch)
            self.summarywriter.add_scalar("epoch_accuracy/training", self.metrics["train"]["accuracy"], epoch)

        #run scheduler per epoch
        if self.epoch_scheduler:
                    self.epoch_scheduler.step()
    
    def evaluate(self, dataloader, epoch):
        self.reset_metrics("val")
        self.model.eval()
        step = 0

        with torch.no_grad():
            for X,y in dataloader:
                step+=1
                self.val_steps+=1

                X,y = X.to(self.device), y.to(self.device)
                logits = self.model.forward(X)

                y_pred = self.predict(logits)
                step_accuracy = accuracy(y_pred, y)
                self.metrics["val"]["accuracy"] += step_accuracy

                loss = self.criterion(logits, y)
                self.metrics["val"]["loss"] += loss

                            #output to tensorboard
                if self.summarywriter:
                    self.summarywriter.add_scalar("batch_loss/evaluation", loss, self.val_steps)
                    self.summarywriter.add_scalar("batch_accuracy/evaluation", step_accuracy, self.val_steps)

                yield step, loss

        #calculate final metrics 
        self.metrics["val"]["loss"] /= step
        self.metrics["val"]["accuracy"] /= step

        #output to tensorboard
        if self.summarywriter:
            self.summarywriter.add_scalar("epoch_loss/evaluation", self.metrics["val"]["loss"], epoch)
            self.summarywriter.add_scalar("epoch_accuracy/evaluation", self.metrics["val"]["accuracy"], epoch)
        print("Accuracy: {:.2f}%".format(self.metrics['val']['accuracy']*100))

        #if accuracy improves, save the model
        if self.save_path and (self.metrics['val']['accuracy'] > self.best_accuracy):
          torch.save(self.model.state_dict(), self.save_path)

    def test(self, dataloader):
      self.model.eval()
      y_pred = []
      step = 0
      test_accuracy = 0
      with torch.no_grad():
        for X,y in dataloader:
            step += 1
            X,y = X.to(self.device), y.to(self.device)
            outputs = self.model.forward(X)

            y_pred_step = list(self.predict(outputs))
            y_pred += y_pred_step
            test_accuracy += accuracy(torch.tensor(y_pred_step).to(self.device), y)

      return y_pred, test_accuracy/step

    def fit(self, epochs, dataloaders):
        for epoch in range(epochs):
            print(f"")
            #TRAIN
            for step, loss in self.train(dataloaders["train"], epoch):
                print(f"EPOCH: {epoch+1} | Training Step: {step} | Loss: {loss.item():.3f}")

            #EVALUATE
            for step, loss in self.evaluate(dataloaders["val"], epoch):
                print(f"EPOCH: {epoch+1} | Validation Step: {step} | Loss: {loss.item():.3f}")

    def feed_metrics(self, outputs, y):
        for metric in self.metrics:
            metric(outputs, y)

    def print_metrics(self):
        for metric in self.metrics:
            print(f"{metric.__class__.__name__}: {metric.evaluate()}")

    def metrics_to_summarywriter(self, step: int, mode="train": str):
        for metric in self.metrics:
            self.summarywriter.add_scalar(f"metric.{__class__.__name__}/{mode}", metric.evaluate(), step)

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                self.feed_metrics(outputs, y)
        self.print_metrics()

class Metric(ABC):
    def __init__(self):
        self.cache = 0
        self.i = 0

    def __call__(self, logits, labels):
        self.i += 1
        self.cache += self.forward(logits.detach(), labels)

    def evaluate(self):
        result = self.cache / self.i
        self.cache = 0
        self.i = 0
        return result
    
    @ABC.abstractmethod
    def forward(self):
        pass

class CrossEntropyLoss(Metric):
    def forward(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")

class Accuracy(Metric):
    def forward(self, logits, labels):
        return torch.mean((torch.argmax(logits, dim=-1) == labels).float())