import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Prefer Keras 3 (TF>=2.16); fallback to tf.keras for older versions
try:
    import keras
    from keras import layers
    print(f"[INFO] Using Keras {keras.__version__}")
except Exception:
    from tensorflow import keras
    from tensorflow.keras import layers
    print("[INFO] Using tf.keras fallback")

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# DATASET PATH CONFIGURATION
DATA_DIR = Path(r"C:\Users\Indransh\OneDrive\Desktop\CV Dataset\TestData")  # <-- PUT YOUR PATH HERE

print(f"Looking for dataset in: {DATA_DIR.absolute()}")

# Verify dataset files exist
IMAGES_FILE = DATA_DIR / "challenge_images.npy"
MASKS_FILE = DATA_DIR / "challenge_masks.npy"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR.absolute()}")
if not IMAGES_FILE.exists():
    raise FileNotFoundError(f"Images file not found: {IMAGES_FILE.absolute()}")
if not MASKS_FILE.exists():
    raise FileNotFoundError(f"Masks file not found: {MASKS_FILE.absolute()}")

print(f"Dataset files found!")

# LOAD DATA
images = np.load(IMAGES_FILE)
masks = np.load(MASKS_FILE)

print(f"Loaded images: {images.shape}, dtype={images.dtype}")
print(f"Loaded masks:  {masks.shape}, dtype={masks.dtype}")

# Ensure matching lengths and proper data types
N = min(len(images), len(masks))
if len(images) != N or len(masks) != N:
    print(f"Trimming to common length: {N}")
    images = images[:N]
    masks = masks[:N]

masks = masks.astype(np.int32)
unique_vals = np.unique(masks)
num_classes = int(unique_vals.max() + 1)
print(f"Mask classes: {sorted(unique_vals.tolist())} → num_classes={num_classes}")

# STRATIFICATION LABELS
def get_stratify_labels(masks_arr):
    """Dominant non-background class per image (1..K-1); 0 if only background."""
    labels = []
    for m in masks_arr:
        u, c = np.unique(m, return_counts=True)
        counts = dict(zip(u.tolist(), c.tolist()))
        any_non_bg = any(cls > 0 and counts.get(cls, 0) > 0 for cls in range(1, num_classes))
        if any_non_bg:
            non_bg_counts = {cls: counts.get(cls, 0) for cls in range(1, num_classes)}
            labels.append(max(non_bg_counts, key=non_bg_counts.get))
        else:
            labels.append(0)
    return np.array(labels, dtype=np.int32)

strat_labels = get_stratify_labels(masks)
print(f"Generated stratification labels shape: {strat_labels.shape}")

# TRAIN/VALIDATION/TEST SPLIT (70:15:15)
def safe_splits(images, masks, strat_labels, seed=SEED):
    try:
        X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
            images, masks, strat_labels, test_size=0.30, random_state=seed, stratify=strat_labels
        )
        X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
            X_temp, y_temp, s_temp, test_size=0.50, random_state=seed, stratify=s_temp
        )
        print("Stratified split successful")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except ValueError as e:
        print("Stratified split failed:", e)
        print("Falling back to random split...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, masks, test_size=0.30, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = safe_splits(images, masks, strat_labels)

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val  : X={X_val.shape}, y={y_val.shape}")
print(f"Test : X={X_test.shape}, y={y_test.shape}")

# NORMALIZATION & ONE-HOT ENCODING
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=num_classes)

# CLASS WEIGHTS FOR FOCAL LOSS
flat = y_train.flatten()
counts = np.bincount(flat, minlength=num_classes).astype(np.float32)
freq = counts / counts.sum()
alpha = 1.0 / (freq + 1e-8)
alpha = alpha / alpha.sum()

print("Class pixel counts:", counts.astype(int).tolist())
print("Alpha (per class) :", np.round(alpha, 4).tolist())

# U-NET MODEL ARCHITECTURE
def conv_block(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0: 
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder(x, filters, dropout=0.0):
    c = conv_block(x, filters, dropout)
    p = layers.MaxPool2D(2)(c)
    return c, p

def decoder(x, skip, filters, dropout=0.0):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, dropout)
    return x

def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    s1, p1 = encoder(inputs, 64, 0.05)
    s2, p2 = encoder(p1, 128, 0.05)
    s3, p3 = encoder(p2, 256, 0.10)
    s4, p4 = encoder(p3, 512, 0.10)
    
    b = conv_block(p4, 1024, 0.15)
    
    d1 = decoder(b, s4, 512, 0.10)
    d2 = decoder(d1, s3, 256, 0.10)
    d3 = decoder(d2, s2, 128, 0.05)
    d4 = decoder(d3, s1, 64, 0.05)
    
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d4)
    return keras.Model(inputs, outputs, name='U-Net')

model = build_unet(X_train.shape[1:], num_classes)
model.summary()

# LOSS FUNCTIONS & METRICS
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0,1,2])
    denom = tf.reduce_sum(y_true + y_pred, axis=[0,1,2])
    dice = (2. * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)

def categorical_focal(alpha_vec, gamma=2.0):
    alpha_vec = tf.constant(alpha_vec, dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        alpha_pix = tf.reduce_sum(y_true * alpha_vec, axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        loss = -alpha_pix * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return loss_fn

def combined_loss(y_true, y_pred):
    return 0.6 * categorical_focal(alpha)(y_true, y_pred) + 0.4 * dice_loss(y_true, y_pred)

def class_iou_metric(c):
    def iou(y_true, y_pred):
        y_true_cls = tf.argmax(y_true, axis=-1)
        y_pred_cls = tf.argmax(y_pred, axis=-1)
        y_true_c = tf.cast(tf.equal(y_true_cls, c), tf.float32)
        y_pred_c = tf.cast(tf.equal(y_pred_cls, c), tf.float32)
        inter = tf.reduce_sum(y_true_c * y_pred_c)
        union = tf.reduce_sum(y_true_c) + tf.reduce_sum(y_pred_c) - inter
        return tf.where(union > 0, inter / union, 1.0)
    iou.__name__ = f'iou_class_{c}'
    return iou

metrics = [keras.metrics.MeanIoU(num_classes=num_classes)]
for c in range(num_classes):
    metrics.append(class_iou_metric(c))

# COMPILE & CALLBACKS
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss=combined_loss, metrics=metrics)

# Create output directory
os.makedirs('model_outputs', exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint('model_outputs/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.ModelCheckpoint('model_outputs/best_weights.weights.h5', monitor='val_loss',
                                  save_best_only=True, save_weights_only=True, verbose=1),
]

# TRAIN
EPOCHS = 50
BATCH_SIZE = 12

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# PLOTS: LOSS & IoU
def plot_curves(hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(hist.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(hist.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IoU plot
    if 'mean_io_u' in hist.history:
        ax2.plot(hist.history['mean_io_u'], label='Train IoU', linewidth=2)
        ax2.plot(hist.history['val_mean_io_u'], label='Val IoU', linewidth=2)
    elif 'mean_iou' in hist.history:
        ax2.plot(hist.history['mean_iou'], label='Train IoU', linewidth=2)
        ax2.plot(hist.history['val_mean_iou'], label='Val IoU', linewidth=2)
    
    ax2.set_title('Training vs Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_outputs/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_curves(history)

# EVALUATE & SAVE
print("\nEvaluating on test set...")
test_results = model.evaluate(X_test, y_test_cat, verbose=1)

for name, val in zip(model.metrics_names, test_results):
    print(f"{name}: {val:.4f}")

# Save final model
model.save('model_outputs/model.keras')

print("\nTraining completed!")
print("Saved: model_outputs/best_model.keras")
print("Saved: model_outputs/model.keras") 
print("Saved: model_outputs/training_curves.png")