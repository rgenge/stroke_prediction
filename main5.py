from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception as e:
    print("Missing dependency:", e)
    sys.exit(1)

IMAGE_SIZE = (128, 128)


def load_images_from_folder(folder: Path, label: int, limit: int = 0) -> Tuple[List[np.ndarray], List[int]]:
    imgs = []
    labels = []
    if not folder.exists():
        return imgs, labels
    files = [p for p in folder.iterdir() if p.is_file()]
    if limit and len(files) > limit:
        files = files[:limit]
    for f in files:
        try:
            with Image.open(f) as im:
                im = im.convert("L")
                im = im.resize(IMAGE_SIZE)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=-1)
                imgs.append(arr)
                labels.append(label)
        except Exception:
            continue
    return imgs, labels


def build_dataset(base_dir: Path, limit_per_class: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    stroke_dir = base_dir / "Stroke"
    non_dir = base_dir / "NonStroke"
    xs1, ys1 = load_images_from_folder(stroke_dir, 1, limit_per_class)
    xs0, ys0 = load_images_from_folder(non_dir, 0, limit_per_class)
    all_x = xs1 + xs0
    if not all_x:
        X = np.empty((0, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    else:
        X = np.stack(all_x)
    y = np.array(ys1 + ys0, dtype=int)
    labels = ["NonStroke", "Stroke"]
    return X, y, labels


def build_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_confusion(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sample_grid(X: np.ndarray, y: np.ndarray, labels: List[str], out_path: Path, n=6):
    fig, axs = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for row, label in enumerate([1, 0]):
        idxs = np.where(y == label)[0]
        if len(idxs) == 0:
            for ax in axs[row]:
                ax.axis("off")
            continue
        chosen = np.random.choice(idxs, min(len(idxs), n), replace=False)
        for col in range(n):
            ax = axs[row, col]
            if col < len(chosen):
                arr = X[chosen[col]].reshape(IMAGE_SIZE)
                ax.imshow(arr, cmap="gray")
                ax.set_title(labels[label], fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_pdf_report(out_pdf: Path, metrics: dict, cm_path: Path, grid_path: Path, txt_report: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Stroke Analyser Report", ln=True)
    pdf.ln(2)
    pdf.multi_cell(0, 6, txt_report)
    pdf.ln(4)
    if cm_path.exists():
        pdf.image(str(cm_path), w=120)
        pdf.ln(4)
    if grid_path.exists():
        pdf.image(str(grid_path), w=180)
    pdf.output(str(out_pdf))


def main():
    parser = argparse.ArgumentParser(description="CNN-based Stroke Analyser")
    parser.add_argument("--data_dir", type=str, default=".", help="Base folder containing Stroke/ and NonStroke/")
    parser.add_argument("--limit", type=int, default=200, help="Max images per class (0 = no limit)")
    parser.add_argument("--out", type=str, default="report.pdf", help="Output PDF report path")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    base = Path(args.data_dir).resolve()
    X, y, labels = build_dataset(base, limit_per_class=args.limit)
    if X.size == 0 or y.size == 0:
        print("No images found. Ensure folders 'Stroke' and 'NonStroke' exist under", base)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Training samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

    model = build_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=16,
                        validation_split=0.2, verbose=1)

    preds = (model.predict(X_test) > 0.5).astype("int32").flatten()
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report_text = classification_report(y_test, preds, target_names=labels, zero_division=0)

    print("Accuracy:", acc)
    print(report_text)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    cm_path = out_dir / "confusion.png"
    grid_path = out_dir / "samples.png"
    plot_confusion(cm, labels, cm_path)
    sample_grid(X_test, y_test, labels, grid_path)

    txt = f"Accuracy: {acc:.4f}\n\nClassification report:\n{report_text}"
    make_pdf_report(Path(args.out), {"accuracy": acc}, cm_path, grid_path, txt)

    model.save("models/cnn_model.h5")
    print("Model saved to: models/cnn_model.h5")
    print("Report saved to:", args.out)


if __name__ == "__main__":
    main()
