import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import autocase, GradScaler

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix


def train(
  model: torch.nn.Module,
  epochs: int = 1, 
  log_every: int, 
  train_loader: torch.utils.DataLoader, 
  test_loader: torch.utils.DataLoader
): 
  device = model.device # ***
  
  for epoch in range(1, epochs + 1): 
      model.train()
      epoch_loss = 0.0
      n_rows_seen = 0

      probs_epoch, labels_epoch = [], []

      running_loss = 0.0
      running_rows = 0
      probs_running, labels_running = [], []

      for batch_idx, (i, j, y) in enumerate(train_oader, start=1):
          i = i.to(device, non_blocking=True) # ***
          j = j.to(device, non_blocking=True) # ***
          y = y.to(device, non_blocking=True) # ***        

          optimizer.zero_grad(set_to_none=True) # ***
    

