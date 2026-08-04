"""Detection metric shared by training and evaluation."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm


@torch.no_grad()
def compute_map50(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return mAP@0.5 over ``loader``."""
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    for images_L, targets_L in tqdm(loader, desc="validating", leave=False):
        predictions_L = model([image_CHW.to(device) for image_CHW in images_L])
        metric.update(
            [
                {key: value.cpu() for key, value in prediction.items()}
                for prediction in predictions_L
            ],
            [{key: value.cpu() for key, value in target.items()} for target in targets_L],
        )
    return float(metric.compute()["map_50"])
