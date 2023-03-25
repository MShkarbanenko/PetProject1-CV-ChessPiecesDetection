from dataset import *
from metrics import *


def collate_fn(batch):
    """Transforms the batch to appropriate form for dataloader."""
    return tuple(zip(*batch))


class FasterRCNN(LightningModule):
    """Faster R-CNN detection model."""

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            num_classes=13,
            trainable_backbone_layers=1,
            box_score_thresh=0.3
        )
        self.save_hyperparameters()

    def forward(self, images, targets=None):
        if targets is None:
            self.model.eval()
            return self.model(images)
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = torch.stack(list(batch[0]), dim=0), list(batch[1])
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("Train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = torch.stack(list(batch[0]), dim=0), list(batch[1])
        predictions = self.model(images)
        ap_50 = precision_recall_ap(targets, predictions, iou_thresh=0.5)[2]
        ap_75 = precision_recall_ap(targets, predictions, iou_thresh=0.75)[2]
        ap_95 = precision_recall_ap(targets, predictions, iou_thresh=0.95)[2]
        mean_ap = mean_average_precision(targets, predictions)
        acc = accuracy(targets, predictions)
        red = redundancy(targets, predictions)
        metrics_dict = {"Validation AP50": ap_50,
                        "Validation AP75": ap_75,
                        "Validation AP95": ap_95,
                        "Validation mAP": mean_ap,
                        "Validation accuracy": acc,
                        "Validation redundancy": red
                        }
        self.log_dict(metrics_dict)
        return metrics_dict

    def train_dataloader(self):
        train_dataset = CPDDataset(TRAIN_DIR, CLASSES_DICT)
        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)
        return train_loader

    def val_dataloader(self):
        val_dataset = CPDDataset(VAL_DIR, CLASSES_DICT)
        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                collate_fn=collate_fn,
                                shuffle=False,
                                num_workers=NUM_WORKERS)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer
