import splitfolders
import os

print("Starting dataset split...")

splitfolders.ratio(
    'dataset/Training',
    output='dataset/split',
    seed=42,
    ratio=(0.70, 0.15, 0.15)
)

print("\nSplit complete!")
print("Folders created:")
for folder in ['train', 'val', 'test']:
    path = f'dataset/split/{folder}'
    for cls in os.listdir(path):
        count = len(os.listdir(f'{path}/{cls}'))
        print(f"  {folder}/{cls}: {count} images")