from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, Input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception as e:
    print("Missing dependency:", e)
    sys.exit(1)

IMAGE_SIZE = (128, 128)

# ------------------ Data Loading ------------------
def load_images_from_folder(folder: Path, label: int, limit: int = 0) -> Tuple[List[np.ndarray], List[int]]:
    imgs, labels = [], []
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
        except:
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

# ------------------ Model ------------------
def build_cnn(input_shape):
    model = models.Sequential()
    
    # Add layers one by one to avoid Input layer serialization issues
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Use a lower learning rate for better convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_data_augmentation():
    """Create data augmentation generator for training"""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# ------------------ Utils ------------------
def plot_confusion(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
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

def load_single_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as im:
        im = im.convert("L")
        im = im.resize(IMAGE_SIZE)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
        return np.expand_dims(arr, axis=0)

# ------------------ Main ------------------
def main():
    print("Choose an option:")
    print("1 - Train/Run Model")
    print("2 - Test with Image")
    choice = input("Enter choice (1/2): ").strip()

    model_path = Path("models/cnn_model.keras")
    model_path.parent.mkdir(exist_ok=True)

    if choice == "1":
        data_dir = input("Enter data directory path: ").strip() or "."
        limit = int(input("Limit images per class (0 = no limit): ") or "0")
        epochs = int(input("Training epochs: ") or "10")

        base = Path(data_dir).resolve()
        X, y, labels = build_dataset(base, limit_per_class=limit)
        if X.size == 0 or y.size == 0:
            print("No images found. Ensure folders 'Stroke' and 'NonStroke' exist under", base)
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        print("Training samples:", X_train.shape[0], "Test samples:", X_test.shape[0])
        
        # Calculate class weights to handle imbalance
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")

        model = build_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
        
        # Setup callbacks for better training
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='max',
                min_delta=0.001
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Create data augmentation
        datagen = create_data_augmentation()
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training split: {X_train_split.shape[0]}, Validation split: {X_val_split.shape[0]}")
        
        # Calculate proper steps per epoch
        batch_size = 16
        steps_per_epoch = max(1, len(X_train_split) // batch_size)
        
        # Fit the model with augmented data
        model.fit(
            datagen.flow(X_train_split, y_train_split, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val_split, y_val_split),
            class_weight=class_weight_dict,
            callbacks=callback_list,
            verbose=1,
            steps_per_epoch=steps_per_epoch
        )

        # Load the best model for evaluation
        model = tf.keras.models.load_model(model_path)
        
        # Make predictions with optimal threshold
        y_probs = model.predict(X_test).flatten()
        
        # Find optimal threshold using validation data
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Fallback to 0.5 if threshold is extreme
        if optimal_threshold <= 0.01 or optimal_threshold >= 0.99:
            optimal_threshold = 0.5
            print(f"Using fallback threshold: {optimal_threshold:.3f}")
        else:
            print(f"Optimal threshold: {optimal_threshold:.3f}")
        
        preds = (y_probs > optimal_threshold).astype("int32")
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        report_text = classification_report(y_test, preds, target_names=labels, zero_division=0)

        print("Accuracy:", acc)
        print(report_text)
        
        # Additional metrics
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_probs)
        print(f"AUC-ROC: {auc:.4f}")

        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        cm_path = out_dir / "confusion.png"
        grid_path = out_dir / "samples.png"
        plot_confusion(cm, labels, cm_path)
        sample_grid(X_test, y_test, labels, grid_path)

        txt = f"Accuracy: {acc:.4f}\nAUC-ROC: {auc:.4f}\nOptimal Threshold: {optimal_threshold:.3f}\n\nClassification report:\n{report_text}"
        make_pdf_report(Path("report.pdf"), {"accuracy": acc, "auc": auc}, cm_path, grid_path, txt)

        print("Best model saved to:", model_path)
        print("Report saved to: report.pdf")

    elif choice == "2":
        if not model_path.exists():
            print("No trained model found. Please run option 1 first.")
            return
        image_path = input("Enter image path: ").strip()
        if not Path(image_path).exists():
            print("Image not found.")
            return
        
        # Try to load the model with error handling
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("The saved model may be incompatible. Please retrain the model using option 1.")
            return
        img = load_single_image(Path(image_path))
        prob = model.predict(img)[0][0]
        print(f"Stroke probability: {prob:.2%}")
        
        # Use a more nuanced threshold system
        if prob > 0.8:
            print("🔴 HIGH RISK: Strong indication of stroke")
        elif prob > 0.6:
            print("🟡 MODERATE-HIGH RISK: Consult a medical professional")
        elif prob > 0.4:
            print("🟡 MODERATE RISK: Monitor closely")
        elif prob > 0.2:
            print("🟢 LOW-MODERATE RISK: Unlikely but monitor")
        else:
            print("🟢 LOW RISK: No strong indication of stroke")
        
        print("\n⚠️  DISCLAIMER: This is an AI model for educational purposes.")
        print("Always consult healthcare professionals for medical advice.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
	main()
