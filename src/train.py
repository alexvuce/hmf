import numpy as np
import torch
from torch import optim
from torch.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

from .utils import regularized_loss

def build_model(
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
    train_steps: int = 1000,
    log_every: int = 100,
    train_loader: torch.utils.data.DataLoader | None = None,
    test_loader: torch.utils.data.DataLoader | None = None,
    lr: float = 1e-4,
    wd: float = 1e-8,
    lambda_: float = 1e-8,
    grad_clip_norm_ceil: float | None = None
):
    device = next(model.parameters()).device
    device_type = "cuda" if device.type == "cuda" else "cpu"

    use_bf16 = (device_type == "cuda" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = (device_type == "cuda" and not use_bf16)
    scaler = GradScaler(enabled=use_scaler)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    step = 1
    while step <= train_steps:
        model.train()
        for i, j, y in train_loader:
            if step > train_steps:
                break

            i, j, y = i.to(device, non_blocking=True), j.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, dtype=amp_dtype):
                logits = model(i, j)
                train_loss, regularizer = regularized_loss(logits, y, model.U, model.V, device, model.k, lambda_)
                train_reg_loss = train_loss + regularizer

            if use_scaler:
                scaler.scale(train_reg_loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm_ceil:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_ceil)
                scaler.step(optimizer)
                scaler.update()
            else:
                train_reg_loss.backward()
                if grad_clip_norm_ceil:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_ceil)
                optimizer.step()

            if step % log_every == 0:
                # Training metrics
                train_batch_loss = train_loss.item()
                train_batch_reg_penalty = regularizer.item()
                train_batch_labels = y.detach().cpu().numpy()
                train_batch_probs = torch.sigmoid(logits.detach()).cpu().numpy()

                # Test metrics
                model.eval()
                test_set_loss_sum = 0.0
                test_set_row_count = 0
                test_set_labels = []
                test_set_probs = []

                with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype):
                    for i, j, y in test_loader:
                        i, j, y = i.to(device), j.to(device), y.to(device)
                        logits = model(i, j)
                        test_loss, _ = regularized_loss(logits, y, model.U, model.V, device, model.k, lambda_)
                        test_set_loss_sum += test_loss.item() * y.size(0)
                        test_set_row_count += y.size(0)
                        test_set_labels.append(y.cpu().numpy())
                        test_set_probs.append(torch.sigmoid(logits).cpu().numpy())

                test_set_labels = np.concatenate(test_set_labels)
                test_set_probs = np.concatenate(test_set_probs)
                test_set_loss = test_set_loss_sum / test_set_row_count

                print(f"Step {step:>5d} | Penalty: {train_batch_reg_penalty:.4f} "
                      f"| Train Loss: {train_batch_loss:.4f} "
                      f"| AUROC: {roc_auc_score(train_batch_labels, train_batch_probs):.4f} "
                      f"| Acc: {accuracy_score(train_batch_labels, train_batch_probs.round()):.4f} "
                      f"| Test Loss: {test_set_loss:.4f} "
                      f"| AUROC: {roc_auc_score(test_set_labels, test_set_probs):.4f} "
                      f"| Acc: {accuracy_score(test_set_labels, test_set_probs.round()):.4f}")

            step += 1

    return model