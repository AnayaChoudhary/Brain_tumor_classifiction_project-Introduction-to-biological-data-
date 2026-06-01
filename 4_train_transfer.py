import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

IMG_SIZE = 224
BATCH    = 32
CLASSES  = 4

print("Loading data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/split/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    'dataset/split/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical'
)

print(f"Class indices: {train_gen.class_indices}")

# ── PHASE 1: Train classifier head only ──────────────────────────
print("\nPhase 1: Training classifier head...")

base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False

x   = base.output
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dense(256, activation='relu')(x)
x   = layers.Dropout(0.5)(x)
out = layers.Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks_p1 = [
    EarlyStopping(patience=5, restore_best_weights=True,
                  monitor='val_loss', verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3,
                      monitor='val_loss', verbose=1),
    ModelCheckpoint('models/efficientnet_phase1.h5',
                    save_best_only=True,
                    monitor='val_accuracy', verbose=1)
]

history1 = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks_p1,
    verbose=1
)

# ── PHASE 2: Fine-tune top layers ────────────────────────────────
print("\nPhase 2: Fine-tuning top layers...")

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    EarlyStopping(patience=8, restore_best_weights=True,
                  monitor='val_loss', verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=4,
                      monitor='val_loss', verbose=1),
    ModelCheckpoint('models/efficientnet_best.h5',
                    save_best_only=True,
                    monitor='val_accuracy', verbose=1)
]

history2 = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=callbacks_p2,
    verbose=1
)

# ── Plot both phases ─────────────────────────────────────────────
acc  = history1.history['accuracy']     + history2.history['accuracy']
val  = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss']         + history2.history['loss']
vloss= history1.history['val_loss']     + history2.history['val_loss']

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(acc,  label='Train Accuracy')
plt.plot(val,  label='Val Accuracy')
plt.axvline(x=len(history1.history['accuracy'])-1,
            color='gray', linestyle='--', label='Fine-tune start')
plt.title('EfficientNet Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss,  label='Train Loss')
plt.plot(vloss, label='Val Loss')
plt.axvline(x=len(history1.history['loss'])-1,
            color='gray', linestyle='--', label='Fine-tune start')
plt.title('EfficientNet Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/efficientnet_training_curve.png', dpi=150)
plt.show()

print("\nSaved: outputs/efficientnet_training_curve.png")
print("Best model saved to: models/efficientnet_best.h5")