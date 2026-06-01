import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('outputs', exist_ok=True)

CLASSES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224

# Load best model
print("Loading model...")
model = tf.keras.models.load_model('models/efficientnet_best.h5')
print("Model loaded successfully!")

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'dataset/split/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ── Predictions ──────────────────────────────────────────────────
print("\nRunning predictions on test set...")
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes

# ── Classification Report ────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# ── Confusion Matrix ─────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt='d',
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    cmap='Blues'
)
plt.title('Confusion Matrix — EfficientNet')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.show()
print("Saved: outputs/confusion_matrix.png")

# ── Overall Accuracy ─────────────────────────────────────────────
accuracy = np.sum(y_pred == y_true) / len(y_true) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ── Grad-CAM ─────────────────────────────────────────────────────
print("\nGenerating Grad-CAM visualization...")

def generate_grad_cam(model, img_path, layer_name='top_conv'):
    # Load and preprocess image
    img_orig = cv2.imread(img_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(img_orig, (IMG_SIZE, IMG_SIZE))
    img_norm = img_orig / 255.0
    x        = img_norm[np.newaxis, ...]

    # Build grad model
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        pred_class      = tf.argmax(preds[0])
        loss            = preds[:, pred_class]

    grads   = tape.gradient(loss, conv_out)[0]
    cam     = tf.reduce_mean(grads, axis=(0, 1)).numpy()
    cam     = np.maximum(cam, 0)
    cam     = cam / (cam.max() + 1e-8)
    cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)

    return img_orig, overlay, CLASSES[pred_class.numpy()], float(preds[0][pred_class])

# Pick one sample from each class
fig, axes = plt.subplots(4, 2, figsize=(10, 20))
fig.suptitle('Grad-CAM — Original vs Heatmap', fontsize=14)

for i, cls in enumerate(CLASSES):
    cls_folder = f'dataset/split/test/{cls}'
    img_file   = os.listdir(cls_folder)[0]
    img_path   = f'{cls_folder}/{img_file}'

    try:
        orig, overlay, pred_label, confidence = generate_grad_cam(model, img_path)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f'Original — True: {cls}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f'Grad-CAM — Pred: {pred_label} ({confidence*100:.1f}%)')
        axes[i, 1].axis('off')

    except Exception as e:
        print(f"Grad-CAM failed for {cls}: {e}")

plt.tight_layout()
plt.savefig('outputs/grad_cam_all_classes.png', dpi=150)
plt.show()
print("Saved: outputs/grad_cam_all_classes.png")
print("\nEvaluation complete! Check outputs/ folder for all results.")