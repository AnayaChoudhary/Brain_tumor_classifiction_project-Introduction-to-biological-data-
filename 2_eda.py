import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('outputs', exist_ok=True)

BASE = 'dataset/split/train'
classes = os.listdir(BASE)

print("Class image counts (Training set):")
counts = {}
for c in classes:
    count = len(os.listdir(f'{BASE}/{c}'))
    counts[c] = count
    print(f"  {c}: {count} images")

# Plot 1 — Class distribution bar chart
plt.figure(figsize=(8, 4))
sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette='Blues_d')
plt.title('Training Set — Class Distribution')
plt.ylabel('Number of Images')
plt.xlabel('Tumor Class')
plt.tight_layout()
plt.savefig('outputs/class_distribution.png', dpi=150)
plt.show()
print("Saved: outputs/class_distribution.png")

# Plot 2 — Sample image from each class
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Sample MRI Image from Each Class', fontsize=14)

for ax, cls in zip(axes, classes):
    img_folder = f'{BASE}/{cls}'
    img_file   = os.listdir(img_folder)[0]
    img_path   = f'{img_folder}/{img_file}'
    img        = cv2.imread(img_path)
    img        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img        = cv2.resize(img, (224, 224))
    ax.imshow(img)
    ax.set_title(cls.capitalize(), fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/sample_images.png', dpi=150)
plt.show()
print("Saved: outputs/sample_images.png")

# Plot 3 — Pixel intensity distribution
plt.figure(figsize=(10, 4))
colors = ['blue', 'green', 'red', 'orange']
for cls, color in zip(classes, colors):
    img_folder = f'{BASE}/{cls}'
    img_file   = os.listdir(img_folder)[0]
    img        = cv2.imread(f'{img_folder}/{img_file}', cv2.IMREAD_GRAYSCALE)
    img        = cv2.resize(img, (224, 224))
    plt.hist(img.ravel(), bins=50, alpha=0.5, label=cls, color=color)

plt.title('Pixel Intensity Distribution per Class')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/pixel_distribution.png', dpi=150)
plt.show()
print("Saved: outputs/pixel_distribution.png")

print("\nEDA complete! Check your outputs/ folder for all saved plots.")