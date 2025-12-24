import torch
from torch import optim
from torch.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix


def instantiate_model(
    model_class: torch.nn.Module,
    I: int,
    J: int, 
    k: int,
    device: torch.device
):
    '''
    model_class: model from models.py
    I: number of rows in latent matrix U
    J: number of rows in latent matrix V
    k: latent dimsneion size; number of columns in {U, V}
    '''
    return model_class(I, J, k).to(device)

def train(
    model: torch.nn.Module,
    epochs: int = 1, 
    log_every: int, 
    train_loader: torch.utils.DataLoader, 
    test_loader: torch.utils.DataLoader,
    lr: float = 1e-3
): 
    device = model.device # ***

    use_bf16 = (device.type == 'cuda' and torch.cuda.is_bf16_supported()) or device.type == 'cpu'
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = (device.type == 'cuda' and not use_bf16)
    scaler = GradScaler(enabled=use_scaler)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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
    


