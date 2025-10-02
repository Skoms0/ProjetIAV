# =======================
#  test.py — Inference
# =======================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import time
from pathlib import Path

from dataset import COCOTestImageDataset   # <- make sure you use your correct dataset file
from main import get_model                 # we reuse your get_model() to build the network
from config import CONFIG

from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights
)


def get_test_transform(config):
    """
    Returns the deterministic test transform for the selected model.
    Uses the same preprocessing as the original pretrained weights.
    """
    model_name = config["model"].lower()
    pretrained = config.get("pretrained", True)

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    if weights is not None:
        return weights.transforms()
    else:
        return transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def main():
    # ----- Device -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected. Using CUDA for inference.")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")

    # ----- Dataset & DataLoader -----
    test_transform = get_test_transform(CONFIG)
    test_dataset = COCOTestImageDataset(
        img_dir=CONFIG["test_dir"],
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["max_cpus"]
    )

    # ----- Load model -----
    model = get_model(CONFIG, device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    print(f"Loaded model from best_model.pth — starting inference on {len(test_dataset)} images...")
    start_time = time.time()

    # ----- Prediction loop -----
    predictions = {}
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= CONFIG["threshold"]).cpu().numpy()

            for img_id, pred in zip(image_ids, preds):
                predictions[img_id] = pred.nonzero()[0].tolist()

    # ----- Save results -----
    output_json = "test_predictions.json"
    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=4)

    elapsed = time.time() - start_time
    print(f"\n✅ Inference complete! Processed {len(test_dataset)} images "
          f"in {elapsed/60:.2f} min ({elapsed:.1f} s total).")
    print(f"Predictions saved to {output_json}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
