from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange


def mixup_specs(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    sum_loss = 0.
    correct_cnt = 0
    input_size = 0
    total_batches = 0

    batch_bar = tqdm(loader, desc="train progress", leave=False)
    for inputs, labels in batch_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs, targets_l, targets_ml, lm = mixup_specs(inputs, labels)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred, targets_l) * lm + criterion(pred, targets_ml) * (1 - lm)
        loss.backward()
        optimizer.step()

        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        sum_loss += loss.item()
        correct_cnt += (pred.argmax(dim = 1) == labels).sum().item()
        input_size += inputs.shape[0]
        total_batches += 1

    return sum_loss / total_batches, correct_cnt / input_size

def validate(model, loader, criterion, device):
    model.eval()

    sum_loss = 0.
    correct_cnt = 0
    input_size = 0
    total_batches = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(inputs)
            loss = criterion(pred, labels)

        sum_loss += loss.item()
        correct_cnt += (pred.argmax(dim = 1) == labels).sum().item()
        input_size += inputs.shape[0]
        total_batches += 1

    return sum_loss / total_batches, correct_cnt / input_size


def train_cnn(model, optimizer, train_load, val_load, epoch_n=50, scheduler=None, device="cuda"):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_history = []
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

    best_loss = float("inf")
    best_state = None
    no_progress_counter = 0

    epoch_bar = trange(epoch_n, desc="Training")
    for epoch in epoch_bar:
        train_loss, train_acc = train_epoch(model, train_load, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_load, criterion, device)
        scheduler.step(val_loss)
        loss_history.append(val_loss)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            no_progress_counter = 0
        else:
            no_progress_counter += 1

        if no_progress_counter > 15:
            print('[INFO] early stopping')
            break

    model.load_state_dict(best_state)
    return model, dict(zip(list(range(len(loss_history))), loss_history))
