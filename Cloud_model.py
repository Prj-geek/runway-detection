# ===========================
# CEAM CV Segmentation Script
# Google Colab - Single Cell
# ===========================

# ---- 0) Setup & Imports ----
import os, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- 1) Google Drive ----
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# Path to your dataset on Drive
DATA_DIR = '/content/drive/MyDrive/CV_Dataset/TestData'
IMAGES_FILE = os.path.join(DATA_DIR, 'challenge_images.npy')
MASKS_FILE  = os.path.join(DATA_DIR, 'challenge_masks.npy')

assert os.path.exists(IMAGES_FILE), f"Missing file: {IMAGES_FILE}"
assert os.path.exists(MASKS_FILE),  f"Missing file: {MASKS_FILE}"

# ---- 2) Load & Basic Checks ----
images = np.load(IMAGES_FILE)   # uint8, shape [N, H, W, 3]
masks  = np.load(MASKS_FILE)    # uint8/int, shape [N, H, W]

print(f"Loaded images: {images.shape} dtype={images.dtype}")
print(f"Loaded masks : {masks.shape} dtype={masks.dtype}")

# sanity: lengths must match
N = min(len(images), len(masks))
if len(images) != N or len(masks) != N:
    print(f"Trimming to common length {N}")
    images = images[:N]
    masks  = masks[:N]

# infer classes from masks
unique_vals = np.unique(masks)
num_classes = int(unique_vals.max() + 1)
print(f"Mask classes found: {sorted(unique_vals.tolist())} -> num_classes={num_classes}")

# ---- 3) Make stratification labels from masks ----
def get_stratify_labels(masks_arr):
    """Dominant non-background class per image (1..K-1); 0 if only background."""
    labels = []
    for m in masks_arr:
        u, c = np.unique(m, return_counts=True)
        counts = dict(zip(u.tolist(), c.tolist()))
        # Any non-background?
        any_non_bg = any(cls > 0 and counts.get(cls, 0) > 0 for cls in range(1, num_classes))
        if any_non_bg:
            non_bg_counts = {cls: counts.get(cls, 0) for cls in range(1, num_classes)}
            labels.append(max(non_bg_counts, key=non_bg_counts.get))
        else:
            labels.append(0)
    return np.array(labels, dtype=np.int32)

strat_labels = get_stratify_labels(masks)
print("Generated strat_labels shape:", strat_labels.shape)

# ---- 4) Train/Val/Test split (70/15/15) with safe fallback ----
def safe_splits(images, masks, strat_labels, seed=SEED):
    try:
        X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
            images, masks, strat_labels, test_size=0.30, random_state=seed, stratify=strat_labels
        )
        X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
            X_temp, y_temp, s_temp, test_size=0.50, random_state=seed, stratify=s_temp
        )
        print("✅ Stratified split successful")
    except ValueError as e:
        print("⚠️ Stratified split failed:", e)
        print("➡️ Falling back to random split...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, masks, test_size=0.30, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed
        )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = safe_splits(images, masks, strat_labels)

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val  : X={X_val.shape},   y={y_val.shape}")
print(f"Test : X={X_test.shape},  y={y_test.shape}")

# ---- 5) Normalization & One-hot ----
X_train = (X_train.astype('float32')) / 255.0
X_val   = (X_val.astype('float32'))   / 255.0
X_test  = (X_test.astype('float32'))  / 255.0

y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val_cat   = keras.utils.to_categorical(y_val,   num_classes=num_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  num_classes=num_classes)

# ---- 6) Class weights for Focal (alpha per class, inverse frequency) ----
# compute pixel frequencies on training masks
flat = y_train.flatten()
counts = np.bincount(flat, minlength=num_classes).astype(np.float32)
freq = counts / counts.sum()
alpha = 1.0 / (freq + 1e-8)
alpha = alpha / alpha.sum()  # normalize to sum=1 for stability
print("Class pixel counts:", counts.astype(int).tolist())
print("Alpha (per class) :", np.round(alpha, 4).tolist())

# ---- 7) Model: U-Net (BN + Dropout) ----
def conv_block(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0: x = layers.Dropout(dropout)(x)
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

    s1, p1 = encoder(inputs, 64,  0.05)
    s2, p2 = encoder(p1,     128, 0.05)
    s3, p3 = encoder(p2,     256, 0.10)
    s4, p4 = encoder(p3,     512, 0.10)

    b = conv_block(p4, 1024, 0.15)

    d1 = decoder(b,  s4, 512, 0.10)
    d2 = decoder(d1, s3, 256, 0.10)
    d3 = decoder(d2, s2, 128, 0.05)
    d4 = decoder(d3, s1, 64,  0.05)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d4)
    return keras.Model(inputs, outputs, name='U-Net')

model = build_unet(X_train.shape[1:], num_classes)
model.summary()

# ---- 8) Losses & Metrics (Focal + Dice, per-class IoU) ----
def dice_loss(y_true, y_pred, smooth=1e-6):
    # per-pixel multi-class dice
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
        # gather alpha per pixel class
        alpha_pix = tf.reduce_sum(y_true * alpha_vec, axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        loss = -alpha_pix * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return loss_fn

def combined_loss(y_true, y_pred):
    return 0.6 * categorical_focal(alpha)(y_true, y_pred) + 0.4 * dice_loss(y_true, y_pred)

# Per-class IoU metrics
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

metrics = [tf.keras.metrics.MeanIoU(num_classes=num_classes)]
for c in range(num_classes):
    metrics.append(class_iou_metric(c))

# ---- 9) Compile & Callbacks ----
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss=combined_loss, metrics=metrics)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint('best_unet_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# ---- 10) Train ----
EPOCHS = 50
BATCH_SIZE = 12  # adjust based on GPU memory

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ---- 11) Plots: Loss & Dice ----
def plot_curves(hist):
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('loss_curve.png'); plt.show()

    # Dice (reconstruct from loss parts approx – optional to compute directly)
    # If you want explicit dice metric, you can add it to metrics and plot it here.
plot_curves(history)

# ---- 12) Evaluate & Save ----
print("\nEvaluating on test set...")
test_results = model.evaluate(X_test, y_test_cat, verbose=1)
for name, val in zip(model.metrics_names, test_results):
    print(f"{name}: {val:.4f}")

model.save('model.keras')
print("\n✅ Saved: best_unet_model.h5 and model.keras, plus loss_curve.png")
