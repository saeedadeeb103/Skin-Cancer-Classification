from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import datetime
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from hydra.core.hydra_config import HydraConfig

def generate_report(log_dir, model, test_loader, output_dir, cfg):
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics from TensorBoard
    metrics = load_tensorboard_metrics(log_dir)
    
    # Create document structure
    doc = SimpleDocTemplate(
        os.path.join(output_dir, "training_report.pdf"),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create story elements
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=14,
        textColor=colors.HexColor('#2B3A67')
    )
    
    # Add header
    story.append(Paragraph("Training Report Summary", title_style))
    story.append(Spacer(1, 12))
    
    # Add metadata section
    story += create_metadata_section(model, cfg, styles, metrics=metrics)
    story.append(Spacer(1, 24))
    
    # Add hyperparameters table
    story += create_hyperparameters_table(cfg, styles)
    story.append(PageBreak())
    
    # Add training curves
    story += create_training_curves_section(metrics, output_dir, styles)
    story.append(PageBreak())
    
    # Add sample predictions
    story += create_predictions_section(model, test_loader, output_dir, styles)
    
    # Build final PDF
    doc.build(story)
    cleanup_temp_files(output_dir)
    return os.path.join(output_dir, "training_report.pdf")

def load_tensorboard_metrics(log_dir):
    """Load and parse TensorBoard event files"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    metrics = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        metrics[tag] = [e.value for e in events]
    return metrics

def create_metadata_section(model, cfg, styles, metrics):  # Add metrics parameter
    """Create model metadata section"""
    elements = []
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['BodyText'],
        fontSize=10,
        textColor=colors.HexColor('#4A4A4A')
    )
    
    metadata = [
        f"<b>Report Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"<b>Training Duration:</b> {len(metrics.get('train_loss', []))} epochs",
        f"<b>Best Validation Loss:</b> {min(metrics.get('val_loss', [0])):.4f}",
        f"<b>Final Training Loss:</b> {metrics.get('train_loss', [0])[-1]:.4f}"
    ]
    
    for item in metadata:
        elements.append(Paragraph(item, meta_style))
        elements.append(Spacer(1, 4))
    
    return elements

def create_hyperparameters_table(cfg, styles):
    """Create formatted hyperparameters table"""
    elements = []
    header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['BodyText'],
        fontSize=12,
        textColor=colors.HexColor('#FFFFFF'),
        alignment=1
    )
    
    params = []
    for section in ['model', 'dataset', 'training']:
        if hasattr(cfg, section):
            params.extend([
                (f"{key}", str(value))
                for key, value in getattr(cfg, section).items()
            ])
    
    table_data = [
        [Paragraph('Hyperparameter', header_style), 
         Paragraph('Value', header_style)]
    ] + params
    
    # Create table
    table = Table(table_data, colWidths=[2.5*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3A5A40')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8F9FA')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#DEE2E6')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    elements.append(Paragraph("<b>Training Configuration</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))
    elements.append(table)
    return elements

def create_training_curves_section(metrics, output_dir, styles):
    """Create section with training curves"""
    elements = []
    
    # Create and save plots
    loss_path = os.path.join(output_dir, 'loss_curves.png')
    create_combined_loss_plot(metrics).savefig(loss_path, bbox_inches='tight')
    
    metrics_path = os.path.join(output_dir, 'validation_metrics.png')
    create_metrics_plot(metrics).savefig(metrics_path, bbox_inches='tight')
    
    # Add to report
    elements.append(Paragraph("<b>Training Curves</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    # Create two-column layout
    img_table = Table([
        [Image(loss_path, width=3.5*inch, height=2.5*inch),
         Image(metrics_path, width=3.5*inch, height=2.5*inch)]
    ], colWidths=[4*inch, 4*inch])
    
    elements.append(img_table)
    elements.append(Spacer(1, 24))
    
    # Add metric explanations
    elements.append(Paragraph("<i>Training Metrics Legend:</i>", styles['Italic']))
    elements.append(Paragraph(
        "• <b>Loss:</b> CrossEntropyLoss with label smoothing", 
        styles['BodyText']
    ))
    elements.append(Paragraph(
        "• <b>Validation Metrics:</b> Calculated on full validation set after each epoch", 
        styles['BodyText']
    ))
    
    return elements

def create_predictions_section(model, test_loader, output_dir, styles):
    """Create section with sample predictions"""
    elements = []
    elements.append(Paragraph("<b>Sample Predictions</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    # Generate prediction images
    sample_images = generate_organized_predictions(model, test_loader, output_dir)
    
    # Create image grid (3 per row)
    rows = []
    current_row = []
    for img_path in sample_images:
        current_row.append(Image(img_path, width=1.8*inch, height=1.8*inch))
        if len(current_row) == 3:
            rows.append(current_row)
            current_row = []
    if current_row:
        rows.append(current_row)
    
    # Create prediction table
    pred_table = Table(rows, colWidths=[2*inch]*3)
    pred_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    
    elements.append(pred_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "<i>Note: Green labels indicate correct predictions, red indicates errors</i>",
        styles['Italic']
    ))
    
    return elements

def create_combined_loss_plot(metrics):
    """Create professional loss curves plot"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'train_loss' in metrics:
        ax.plot(metrics['train_loss'], 
                label='Training Loss', 
                linewidth=2,
                color='#2B3A67',
                alpha=0.9)
        
    if 'val_loss' in metrics:
        ax.plot(metrics['val_loss'], 
                label='Validation Loss', 
                linewidth=2,
                color='#7EBDC2',
                linestyle='--')
        
    ax.set_title('Training and Validation Loss', fontsize=14, pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    return fig

def create_metrics_plot(metrics):
    """Create validation metrics plot"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_styles = {
        'val_f1': {'color': '#3A5A40', 'label': 'F1 Score'},
        'val_precision': {'color': '#7EBDC2', 'label': 'Precision'},
        'val_recall': {'color': '#FF6B35', 'label': 'Recall'}
    }
    
    for metric, style in metric_styles.items():
        if metric in metrics:
            ax.plot(metrics[metric], 
                    label=style['label'], 
                    linewidth=2,
                    color=style['color'])
            
    ax.set_title('Validation Metrics', fontsize=14, pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    return fig

def generate_organized_predictions(model, test_loader, output_dir, num_samples=9):
    """Generate organized prediction visualizations"""
    model.eval()
    samples = []
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if len(samples) >= num_samples:
                break
            
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            for idx in range(x.size(0)):
                if len(samples) >= num_samples:
                    break
                
                img = denormalize(x[idx]).cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img)
                ax.set_title(
                    f'True: {y[idx].item()}\nPred: {preds[idx].item()}',
                    fontsize=9,
                    color='#3A5A40' if y[idx] == preds[idx] else '#FF6B35'
                )
                ax.axis('off')
                plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
                
                img_path = os.path.join(output_dir, f'pred_{len(samples)}.png')
                plt.savefig(img_path, dpi=120)
                plt.close()
                samples.append(img_path)
    return samples

def cleanup_temp_files(output_dir):
    """Clean temporary visualization files"""
    for f in os.listdir(output_dir):
        if f.startswith('pred_') or f.endswith(('_curves.png', '_metrics.png')):
            os.remove(os.path.join(output_dir, f))