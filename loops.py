import torch
from tqdm import tqdm

def train_loop(train_loader, net, criterion, optimizer, device,
               mbatch_loss_group=-1):
    net.train()
    running_loss = 0.0
    mbatch_losses = []

    for i, data in enumerate(tqdm(train_loader, desc="Training", leave=True)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # following condition False by default, unless mbatch_loss_group > 0
        if i % mbatch_loss_group == mbatch_loss_group - 1:
            mbatch_losses.append(running_loss / mbatch_loss_group)
            running_loss = 0.0
    if mbatch_loss_group > 0:
        return mbatch_losses
    else:
        return running_loss / len(train_loader)  # add average loss return


def validation_loop(val_loader, net, criterion, num_classes, device,
                    multi_task=False, th_multi_task=0.5, one_hot=True, class_metrics=False):
    net.eval()
    loss = 0
    correct = 0
    size = len(val_loader.dataset)
    class_total = {label:0 for label in range(num_classes)}
    class_tp = {label:0 for label in range(num_classes)}
    class_fp = {label:0 for label in range(num_classes)}
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating", leave=True):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            if not multi_task:    
                predictions = torch.zeros_like(outputs)
                predictions[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1.0
            else:
                predictions = torch.where(outputs > th_multi_task, 1.0, 0.0)
            if not one_hot:
                pass
                
            tps = predictions * labels
            fps = predictions - tps
            
            tps = tps.sum(dim=0)
            fps = fps.sum(dim=0)
            lbls = labels.sum(dim=0)  
                
            for c in range(num_classes):
                class_tp[c] += tps[c]
                class_fp[c] += fps[c]
                class_total[c] += lbls[c]
                    
            correct += tps.sum()

    class_prec = []
    class_recall = []
    freqs = []
    for c in range(num_classes):
        class_prec.append(0 if class_tp[c] == 0 else
                          class_tp[c] / (class_tp[c] + class_fp[c]))
        class_recall.append(0 if class_tp[c] == 0 else
                            class_tp[c] / class_total[c])
        freqs.append(class_total[c])

    freqs = torch.tensor(freqs)
    class_weights = 1. / freqs
    class_weights /= class_weights.sum()
    class_prec = torch.tensor(class_prec)
    class_recall = torch.tensor(class_recall)
    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1 = 2. / (1/prec + 1/recall)
    val_loss = loss / size
    accuracy = correct / freqs.sum()
    results = {"loss": val_loss, "accuracy": accuracy, "f1": f1,\
               "precision": prec, "recall": recall}

    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1 = (0 if p == r == 0 else 2. / (1/p + 1/r))
            class_results.append({"f1": f1, "precision": p, "recall": r})
        results = results, class_results

    return results

import torch

@torch.no_grad()
def collect_val_probs_and_labels(val_loader, model, device):
    """Collect raw sigmoid probabilities and labels from the validation set."""
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def f1_weighted_precision_recall(preds, labels):
    preds = preds.float()
    labels = labels.float()

    tps = (preds * labels).sum(dim=0)
    fps = (preds * (1 - labels)).sum(dim=0)
    freqs = labels.sum(dim=0).clamp(min=1e-9)

    class_prec = tps / (tps + fps + 1e-9)
    class_recall = tps / freqs

    class_weights = 1.0 / freqs
    class_weights = class_weights / class_weights.sum()

    prec = (class_prec * class_weights).sum().item()
    rec = (class_recall * class_weights).sum().item()
    f1 = 0.0 if (prec == 0.0 and rec == 0.0) else 2.0 / (1.0 / prec + 1.0 / rec)
    return f1, prec, rec


def sweep_thresholds(probs, labels, thresholds=None, prefer_precision=None):
    """
    Try several thresholds to find the one giving best F1.
    If prefer_precision is set (e.g. 0.4), keep only thresholds with precision >= that.
    """
    if thresholds is None:
        thresholds = [i / 100 for i in range(5, 96, 5)]  # 0.05 .. 0.95

    results = []
    for t in thresholds:
        preds = (probs >= t).float()
        f1, p, r = f1_weighted_precision_recall(preds, labels)
        results.append((t, f1, p, r))

    if prefer_precision is not None:
        eligible = [x for x in results if x[2] >= prefer_precision]
        if eligible:
            return max(eligible, key=lambda x: x[1])  # best F1 among precision >= target

    return max(results, key=lambda x: x[1])  # best F1 overall
