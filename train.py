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
import datetime
from collections import defaultdict



class MetricsTracker(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(list)
    
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        if 'train_loss' in metrics:
            self.metrics['train_loss'].append(metrics['train_loss'].item())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        for key in ['val_loss', 'val_f1', 'val_recall', 'val_precision']:
            if key in metrics:
                self.metrics[key].append(metrics[key].item())

def generate_report(metrics, trainer, model, test_loader, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set professional styling for plots
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 100
    })

    # ========== Metrics Visualization ==========
    def create_metric_plot():
        fig, ax = plt.subplots()
        metrics_added = False
        
        # Plot validation metrics
        for metric in ['val_f1', 'val_recall', 'val_precision']:
            if metric in metrics:
                values = [float(x) for x in metrics[metric]]
                ax.plot(values, label=metric.replace('val_', '').title(), linewidth=2)
                metrics_added = True
                
        if metrics_added:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Validation Metrics', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            plt.tight_layout()
            return fig
        return None

    # ========== Loss Visualization ==========
    def create_loss_plot():
        fig, ax = plt.subplots()
        metrics_added = False
        
        if 'train_loss' in metrics:
            train_loss = metrics['train_loss']
            if isinstance(train_loss, torch.Tensor):
                train_loss = [train_loss.item()]
            ax.plot(train_loss, label='Train Loss', linewidth=2, color='navy')
            metrics_added = True
            
        if 'val_loss' in metrics:
            val_loss = metrics['val_loss'] if isinstance(metrics['val_loss'], list) else [metrics['val_loss']]
            ax.plot(val_loss, label='Validation Loss', linewidth=2, color='darkorange')
            metrics_added = True
            
        if metrics_added:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            plt.tight_layout()
            return fig
        return None

    # ========== Sample Predictions ==========
    def save_sample_predictions(num_samples=6):
        model.eval()
        samples = []
        denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= 2:  # Take 2 batches
                    break
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                
                for j in range(x.size(0)):
                    if len(samples) >= num_samples:
                        break
                        
                    # Denormalize image
                    img = denormalize(x[j]).cpu().numpy().transpose(1, 2, 0)
                    img = np.clip(img, 0, 1)
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(img)
                    ax.set_title(f'True: {y[j].item()}\nPred: {preds[j].item()}',
                               fontsize=10, color='green' if y[j] == preds[j] else 'red')
                    ax.axis('off')
                    
                    # Save to buffer
                    img_path = os.path.join(output_dir, f'sample_{len(samples)}.png')
                    plt.savefig(img_path, bbox_inches='tight', dpi=120)
                    plt.close()
                    samples.append(img_path)
        return samples

    # Generate visualizations
    loss_plot = create_loss_plot()
    metrics_plot = create_metric_plot()
    sample_images = save_sample_predictions()

    # ========== PDF Generation ==========
    def create_pdf_report():
        pdf_path = os.path.join(output_dir, 'training_report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Styles
        title_style = {'fontName': 'Helvetica-Bold', 'fontSize': 16, 'textColor': (0.2, 0.2, 0.6)}
        section_style = {'fontName': 'Helvetica-Bold', 'fontSize': 12, 'textColor': (0.3, 0.3, 0.3)}
        footer_style = {'fontName': 'Helvetica-Oblique', 'fontSize': 8, 'textColor': (0.4, 0.4, 0.4)}
        
        # Header
        c.setFont(title_style['fontName'], title_style['fontSize'])
        c.setFillColorRGB(*title_style['textColor'])
        c.drawString(72, height - 72, "Training Results Report")
        
        # Metadata
        c.setFont(section_style['fontName'], 10)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        metadata = [
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Model: {model.__class__.__name__}",
            f"Epochs: {len(metrics.get('train_loss', []))}",
            f"Best Val Loss: {min(metrics.get('val_loss', [np.inf])):.4f}"
        ]
        for i, text in enumerate(metadata):
            c.drawString(72, height - 100 - (i*15), text)
        
        y_position = height - 180
        
        # Loss Plot
        if loss_plot:
            loss_path = os.path.join(output_dir, 'loss_plot.png')
            loss_plot.savefig(loss_path, bbox_inches='tight')
            c.setFont(section_style['fontName'], section_style['fontSize'])
            c.drawString(72, y_position - 20, "Training Progress")
            c.drawImage(loss_path, 72, y_position - 220, width=450, height=200)
            y_position -= 250

        # Metrics Plot
        if metrics_plot:
            metrics_path = os.path.join(output_dir, 'metrics_plot.png')
            metrics_plot.savefig(metrics_path, bbox_inches='tight')
            c.drawString(72, y_position - 20, "Validation Metrics")
            c.drawImage(metrics_path, 72, y_position - 220, width=450, height=200)
            y_position -= 250

        # Sample Predictions
        if sample_images:
            c.setFont(section_style['fontName'], section_style['fontSize'])
            c.drawString(72, y_position - 20, "Sample Predictions")
            
            x_offset = 72
            y_offset = y_position - 100
            img_size = 150
            
            for i, img_path in enumerate(sample_images):
                if i > 0 and i % 3 == 0:
                    x_offset = 72
                    y_offset -= img_size + 30
                    
                c.drawImage(img_path, x_offset, y_offset, 
                           width=img_size, height=img_size)
                x_offset += img_size + 20
                
                if y_offset < 100:  # Prevent overflow
                    c.showPage()
                    y_offset = height - 100

        # Footer
        c.setFont(footer_style['fontName'], footer_style['fontSize'])
        c.setFillColorRGB(*footer_style['textColor'])
        c.drawCentredString(width/2, 40, 
                           "Generated by Model Training Framework - Confidential")
        
        c.save()
        return pdf_path

    # Generate the PDF
    pdf_path = create_pdf_report()
    print(f"\n✅ Training report generated at: {pdf_path}")

    # Cleanup temporary files
    for f in os.listdir(output_dir):
        if f.startswith('sample_') or f.endswith('_plot.png'):
            os.remove(os.path.join(output_dir, f))


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
    metrics_tracker = MetricsTracker()

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision=16 if torch.cuda.is_available() else 32,  # Mixed precision training
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, metrics_tracker],
        accumulate_grad_batches=2  # Gradient accumulation for memory efficiency
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    train_val_metrics = trainer.logged_metrics.copy()  # Save a copy of the metrics
    trainer.test(model, test_loader)
    
    
    # generate pdf report
    generate_report(metrics_tracker.metrics, trainer, model, test_loader, hydra_cfg.runtime.output_dir)
    # Save the trained model
    model_path = f"{hydra_cfg.runtime.output_dir}/lora_only_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    import os
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()