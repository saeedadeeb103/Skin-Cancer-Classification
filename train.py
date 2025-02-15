from torchvision import transforms
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from datasets import list_datasets, get_dataset_by_name
from encoders.encoders import timm_backbones
from torchaudio import transforms as T
from hydra.core.hydra_config import HydraConfig
from utils.helper_functions import collate_fn
import numpy as np
import random
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os


def generate_plots(trainer, save_dir="reports"):
    """Generate training graphs for Loss, Precision, Recall, and F1-score."""
    os.makedirs(save_dir, exist_ok=True)

    metrics = trainer.callback_metrics
    epochs = list(range(1, len(metrics["val_loss"]) + 1))

    def plot_graph(x, y, title, ylabel, save_path):
        plt.figure()
        plt.plot(x, y, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    plot_graph(epochs, metrics["train_loss"], "Training Loss Over Epochs", "Loss", f"{save_dir}/train_loss.png")
    plot_graph(epochs, metrics["val_loss"], "Validation Loss Over Epochs", "Loss", f"{save_dir}/val_loss.png")
    plot_graph(epochs, metrics["val_f1"], "F1 Score Over Epochs", "F1 Score", f"{save_dir}/f1_score.png")
    plot_graph(epochs, metrics["val_precision"], "Precision Over Epochs", "Precision", f"{save_dir}/precision.png")
    plot_graph(epochs, metrics["val_recall"], "Recall Over Epochs", "Recall", f"{save_dir}/recall.png")

def generate_predictions(model, test_loader, save_path="reports/sample_predictions.png", num_samples=5):
    """Generate and save sample predictions from the test set."""
    model.eval()
    images, preds, labels = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            logits = model(x)
            predictions = torch.argmax(logits, dim=1)

            for img, pred, label in zip(x, predictions, y):
                images.append(img)
                preds.append(pred.item())
                labels.append(label.item())

                if len(images) >= num_samples:
                    break
            if len(images) >= num_samples:
                break
    
    # Save sample images with predictions
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, (img, pred, label) in enumerate(zip(images, preds, labels)):
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"GT: {label} | Pred: {pred}")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def create_pdf_report(cfg, save_dir="reports"):
    """Generate a PDF report containing training results."""
    os.makedirs(save_dir, exist_ok=True)
    output_pdf = f"{save_dir}/training_report.pdf"
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Model Training Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Dataset: {cfg.dataset_name}")
    c.drawString(50, height - 100, f"Model: {cfg.model.encoder}")
    c.drawString(50, height - 120, f"Total Epochs: {cfg.max_epochs}")
    c.drawString(50, height - 140, f"Batch Size: {cfg.batch_size}")

    # Insert Graphs
    y_offset = height - 180
    graph_files = ["train_loss.png", "val_loss.png", "f1_score.png", "precision.png", "recall.png"]
    for graph in graph_files:
        img = ImageReader(f"{save_dir}/{graph}")
        c.drawImage(img, 50, y_offset, width=500, height=200, preserveAspectRatio=True, mask='auto')
        y_offset -= 220

    # Insert Sample Predictions
    y_offset -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_offset, "Sample Predictions:")
    y_offset -= 20

    sample_img = ImageReader(f"{save_dir}/sample_predictions.png")
    c.drawImage(sample_img, 50, y_offset - 200, width=500, height=200, preserveAspectRatio=True, mask='auto')

    c.save()
    print(f"Report saved as {output_pdf}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    hydra_cfg = HydraConfig.get()

    # Determine dataset name based on input type
    dataset_name = cfg.dataset_name

    # Print available datasets for debugging
    print(f"Available datasets: {list_datasets()}")
    print(f"Using dataset: {dataset_name}")

    # Define appropriate transformations
    try:
        target_size = tuple(cfg.dataset.target_size)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="train", transform=transform)
        val_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="val", transform=transform)
        test_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="test", transform=transform)
    except Exception:
        raise ValueError(f"Unsupported input_type: {cfg.input_type}")

    from encoders.encoders import timm_backbones
    import os
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=cfg.batch_size, num_workers=min(os.cpu_count(), 7))
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=cfg.batch_size, num_workers=min(os.cpu_count(), 7))
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=cfg.batch_size, num_workers=min(os.cpu_count(), 7))
    model = timm_backbones(
        encoder=cfg.model.encoder,
        num_classes=cfg.num_classes,
        optimizer_cfg=cfg.model.optimizer,
    )
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=f"{hydra_cfg.runtime.output_dir}/checkpoints/",
        filename="best_model"
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    # Define logger
    logger = TensorBoardLogger(save_dir="logs", name="outputloggs")

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision training
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=2  # Gradient accumulation for memory efficiency
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    train_val_metrics = trainer.logged_metrics.copy()  # Save a copy of the metrics
    generate_plots(trainer)
    generate_predictions(model, test_loader)
    create_pdf_report(cfg)

    trainer.test(model, test_loader)
    
    
    # generate pdf report
    
    # Save the trained model
    model_path = f"{hydra_cfg.runtime.output_dir}/lora_only_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    import os
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()