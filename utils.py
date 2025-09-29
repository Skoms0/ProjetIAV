import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import numpy as np

def train_loop(train_loader, net, criterion, optimizer, device):
    net.train()
    loop = tqdm(train_loader, desc="Training", ncols=100)
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss/(i+1))


def validation_loop(val_loader, net, criterion, device, threshold=0.5):
    net.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validation", ncols=100)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            all_outputs.append(outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_outputs).cpu().numpy()
    labels_np = all_labels.cpu().numpy()

    # Per-class F1 for threshold tuning
    thresholds = np.arange(0.3, 0.71, 0.05)
    best_threshold = threshold
    best_micro_f1 = 0
    for t in thresholds:
        preds = (probs > t).astype(int)
        micro_f1 = f1_score(labels_np, preds, average="micro", zero_division=0)
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            best_threshold = t

    # Compute final metrics with best threshold
    preds = (probs > best_threshold).astype(int)
    micro_f1 = f1_score(labels_np, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels_np, preds, average=None, zero_division=0)

    print(f"\nValidation Loss: {total_loss/len(val_loader.dataset):.4f}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f}")
    print("Per-class F1:", per_class_f1)

    return total_loss / len(val_loader.dataset), micro_f1, macro_f1, per_class_f1, best_threshold


def predict_test(test_loader, net, device, threshold=0.5):
    net.eval()
    results = {}
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing", ncols=100)
        for images, stems in loop:
            images = images.to(device)
            outputs = net(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).cpu().numpy()
            for stem, pred in zip(stems, preds):
                classes = [int(i) for i, v in enumerate(pred) if v]
                results[stem] = classes
    return results


def save_model_weights_json(model, filename="model_weights.json"):
    weights_dict = {name: param.cpu().numpy().tolist() for name, param in model.state_dict().items()}
    with open(filename, "w") as f:
        json.dump(weights_dict, f)
    print(f"Model weights saved to {filename}")


def visualize_predictions(test_loader, net, device, classes, threshold=0.5, n_images=5):
    net.eval()
    images_shown = 0
    with torch.no_grad():
        for images, stems in test_loader:
            images = images.to(device)
            outputs = net(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).cpu().numpy()
            for img_tensor, pred, stem in zip(images.cpu(), preds, stems):
                img = transforms.ToPILImage()(img_tensor)
                plt.figure(figsize=(4,4))
                plt.imshow(img)
                class_names = [classes[i] for i, v in enumerate(pred) if v]
                plt.title(f"{stem}: {class_names}")
                plt.axis('off')
                plt.show()
                images_shown += 1
                if images_shown >= n_images:
                    return
