import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import plotly.figure_factory as ff 
import abc
import typing
import dataclasses
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from shutil import copyfile



#import signal
#import logging

# class DelayedKeyboardInterrupt():
#     """Prevent a keyboard interrupt in sensitive areas of code"""

#     def __enter__(self):
#         self.signal_received = False
#         self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
#     def handler(self, sig, frame):
#         self.signal_received = (sig, frame)
#         logging.debug('SIGINT received. Delaying KeyboardInterrupt.')
    
#     def __exit__(self, type, value, traceback):
#         signal.signal(signal.SIGINT, self.old_handler)
#         if self.signal_received:
#             self.old_handler(*self.signal_received)

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
    """Plot the k*k confusion matrix in plotly""" 
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
    """Calculate the weights for each sample, to be fed into the pytorch weighted random sampler"""
    labels = {}
    labels_cnt = {}
    labels_weights = {}
    weights = {}
    for mode in modes: #train/va/test
        labels[mode] = []

        # get a list of all labels
        for i in range(len(slices[mode])):
            labels[mode].append(slices[mode].get_label(i))
        
        labels[mode] = np.sort(labels[mode]) #sort so that counts and weights go into the right index
        labels_cnt[mode] = np.bincount(labels[mode]) #label frequency
        labels_weights[mode] = 1. / labels_cnt[mode] #invert so that low frequency becomes high weight

        #create list with a weight for each sample
        weights[mode] = []
        for i in range(len(slices[mode])):
            label = slices[mode].get_label(i)
            weights[mode].append(labels_weights[mode][label]) 

        weights[mode] = torch.tensor(weights[mode])
    
    return weights

@dataclasses.dataclass
class Runner():
    """ """
    model: torch.nn.Module
    optimizer: torch.optim
    criterion: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    device: torch.device
    metrics: typing.Dict[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    batch_scheduler: torch.optim.lr_scheduler = None
    epoch_scheduler: torch.optim.lr_scheduler = None
    summary_writer: SummaryWriter = None
    checkpoint_path: str = None #path to checkpoint every epoch
    checkpoint_best_path: str = None #path to save best state
    resume_path: str = None #path to file to resume training
    debug: bool = False #only does one step per epoch

    def __post_init__(self):
        self.model = self.model.to(self.device)
        self.best_accuracy = 0
        self.epoch = 0

        if self.resume_path:
            self.resume_from_checkpoint()

    def checkpoint(self, path):
        # with DelayedKeyboardInterrupt:
        #prevent a keyboard interrupt potentially corrupting the checkpoint
        print(f"Saving checkpoint: {path}")

        try:
          copyfile(path, path + "_backup")
        except:
          print(f"Could not backup file: {path}")

        checkpoint_data = {
            'epoch': self.epoch,
            'metrics': self.metrics,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'criterion': self.criterion,
            'best_accuracy': self.best_accuracy
        }

        if self.batch_scheduler:
          checkpoint_data["batch_scheduler"] = self.batch_scheduler.state_dict()
        if self.epoch_scheduler:
          checkpoint_data["epoch_scheduler"] = self.epoch_scheduler.state_dict()

        torch.save(checkpoint_data, path)

    def resume_from_checkpoint(self):
        print(f"Resuming from checkpoint: {self.resume_path}")
        checkpoint = torch.load(self.resume_path)
        self.epoch = checkpoint['epoch'] + 1
        self.metric = checkpoint['metrics']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if "batch_scheduler" in checkpoint:
          self.batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
        if "epoch_scheduler" in checkpoint:
          self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
        self.criterion = checkpoint['criterion']
        self.best_accuracy = checkpoint['best_accuracy']
        
    def train(self, dataloader):

        #switch to train mode
        self.model.train()
        step = 0
        
        #loop over each sample
        for X,y in dataloader:
            step+=1

            X,y = X.to(self.device), y.to(self.device)
            logits = self.model.forward(X)

            # back prop
            loss = self.criterion(logits, y)
            loss.backward()
            
            #step
            self.optimizer.step()
            self.optimizer.zero_grad()

            #run scheduler per step
            if self.batch_scheduler:
                self.batch_scheduler.step()

            self.feed_metrics(logits, y)
            
            yield step, loss
            if self.debug:
              break

        #output to tensorboard
        self.evaluate_metrics()
        if self.summary_writer:
            self.metrics_to_summary_writer(self.epoch, "train")

        #run scheduler per epoch
        if self.epoch_scheduler:
            self.epoch_scheduler.step()
    
    def evaluate(self, dataloader):
        self.model.eval()
        step = 0

        with torch.no_grad():
            for X,y in dataloader:
                step+=1

                X,y = X.to(self.device), y.to(self.device)
                logits = self.model.forward(X)

                loss = self.criterion(logits, y)
                self.feed_metrics(logits, y)
                yield step, loss
                if self.debug:
                  break

        #output to tensorboard
        self.evaluate_metrics()
        if self.summary_writer:
            self.metrics_to_summary_writer(self.epoch, "eval")

        #metrics[0] == accuracy
        print(f"Accuracy: {self.metrics['accuracy'].score*100:.2f}%")

        #if accuracy improves, save the model
        if self.checkpoint_best_path and (self.metrics['accuracy'].score > self.best_accuracy):
            self.best_accuracy = self.metrics['accuracy'].score
            self.checkpoint(self.checkpoint_best_path)

    def test(self, dataloader):
      self.model.eval()
      step = 0
      test_accuracy = 0
      all_logits = torch.tensor([]).to(self.device)
      with torch.no_grad():
        for X,y in dataloader:
            step += 1
            X,y = X.to(self.device), y.to(self.device)
            logits = self.model.forward(X)
            all_logits = torch.cat((all_logits, logits))
            self.feed_metrics(logits, y)
        
        self.evaluate_metrics()

      return all_logits

    def fit(self, epochs, dataloaders):
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            #TRAIN
            for step, loss in self.train(dataloaders["train"]):
                print(f"EPOCH: {epoch+1} | Training Step: {step} | Loss: {loss.item():.3f}")

            #EVALUATE
            for step, loss in self.evaluate(dataloaders["val"]):
                print(f"EPOCH: {epoch+1} | Validation Step: {step} | Loss: {loss.item():.3f}")
            
            if self.checkpoint_path:
                self.checkpoint(self.checkpoint_path)

    def feed_metrics(self, logits, y):
        for _, metric in self.metrics.items():
            metric(logits, y)

    def evaluate_metrics(self):
        for _, metric in self.metrics.items():
            metric.evaluate()

    def metrics_to_summary_writer(self, step: int, mode="train"):
        for metric_name, metric in self.metrics.items():
            self.summary_writer.add_scalar(f"{metric_name}/{mode}", metric.score, step)

class Metric(abc.ABC):
    def __init__(self):
        self.cache = 0
        self.i = 0
        self.score = 0 #stores the last evaluated metric score

    def __call__(self, logits, labels):
        self.i += 1
        self.cache += self.forward(logits.detach(), labels)

    def evaluate(self):
        self.score = self.cache / self.i
        self.cache = 0
        self.i = 0
        return self.score

    @abc.abstractmethod
    def forward(self):
        pass

class CrossEntropyLoss(Metric):
    def forward(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")

class Accuracy(Metric):
    def forward(self, logits, labels):
        return torch.mean((torch.argmax(logits, dim=1) == labels).float())

def TP_FP_rate(logits, labels):
  """Calculates per class TP & FP rates"""
  conf_mat = confusion_matrix(labels, torch.argmax(logits, dim=1))

  condition_positive = np.sum(conf_mat, axis=1)
  predicted_positive = np.sum(conf_mat, axis=0)

  TP = np.diagonal(conf_mat)
  FN = condition_positive - TP
  FP = predicted_positive - TP
  TN = np.sum(conf_mat) - (FP + FN + TP)

  TPR = TP / (TP + FN)
  FPR = FP / (FP + TN)

  return TPR, FPR