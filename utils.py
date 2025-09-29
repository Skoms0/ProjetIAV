import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from metrics import compute_metrics, print_metrics

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
    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validation", ncols=100)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            batch_metrics = compute_metrics(outputs, labels, threshold=0.5)
            print_metrics(batch_metrics, prefix="Val")

    return total_loss / len(val_loader.dataset)

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
