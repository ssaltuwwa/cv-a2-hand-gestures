"""Shared code for the A2 notebook and the demo app.

Model input: RGB float32 images in [0, 1], shape (224, 224, 3).
The model rescales internally to [-1, 1] (MobileNetV2 preprocessing).
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

CLASSES = ["peace-sign", "thumbs-up", "stop"]
IMG_SIZE = 224


# ----------------------------------------------------------------- model
def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", name="aug_flip"),
        layers.RandomRotation(0.04, name="aug_rot"),
        layers.RandomZoom(0.1, name="aug_zoom"),
        layers.RandomTranslation(0.1, 0.1, name="aug_shift"),
        layers.RandomContrast(0.1, name="aug_contrast"),
        layers.RandomBrightness(0.2, value_range=(0, 1), name="aug_bright"),
    ], name="augment")


def build_mobilenet(use_aug=True, imagenet=True, dropout=0.2, n_classes=3):
    """MobileNetV2 transfer model (same head as A1). All layers have fixed names."""
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = build_augmentation()(inp) if use_aug else inp
    x = layers.Rescaling(2.0, offset=-1.0, name="rescale")(x)   # [0,1] -> [-1,1]
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet" if imagenet else None, name="mobilenetv2")
    base.trainable = False
    x = base(x, training=False)          # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="pred")(x)
    return models.Model(inp, out, name="gesture_mobilenetv2")


def build_baseline_cnn(n_classes=3):
    """Simple CNN baseline from A1."""
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augmentation()(inp)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inp, out, name="baseline_cnn")


# ------------------------------------------------------------ preprocessing
def to_model_input(img):
    """PIL image or HxWx3 uint8/float array -> (224,224,3) float32 in [0,1]."""
    arr = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32) / 255.0


# ------------------------------------------------------- stress conditions
def _rng(seed):
    return np.random.default_rng(seed)


def perturb(img01, condition, seed=0):
    """Apply a simulated environment condition to a [0,1] image (H,W,3)."""
    x = img01.astype(np.float32)
    h, w = x.shape[:2]
    if condition == "clean":
        return x
    if condition == "dark":                       # low exposure
        return np.clip((x ** 1.8) * 0.45, 0, 1)
    if condition == "overexposed":
        return np.clip(x * 1.7 + 0.15, 0, 1)
    if condition == "motion_blur":
        k = np.zeros((15, 15), np.float32)
        k[7, :] = 1.0 / 15
        return cv2.filter2D(x, -1, k)
    if condition == "noise":
        return np.clip(x + _rng(seed).normal(0, 0.08, x.shape), 0, 1).astype(np.float32)
    if condition == "low_res":
        small = cv2.resize(x, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if condition == "rotated":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0)
        return cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if condition == "occlusion":
        r = _rng(seed)
        ph, pw = int(h * 0.3), int(w * 0.3)
        y0, x0 = r.integers(0, h - ph), r.integers(0, w - pw)
        x = x.copy()
        x[y0:y0 + ph, x0:x0 + pw] = 0
        return x
    raise ValueError(condition)


CONDITIONS = ["clean", "dark", "overexposed", "motion_blur",
              "noise", "low_res", "rotated", "occlusion"]


# ---------------------------------------------------------------- Grad-CAM
def grad_cam(model, img01, class_idx=None):
    """Grad-CAM on the last conv block of MobileNetV2 (layer 'out_relu').

    Returns (heatmap HxW in [0,1] at 224x224, probs, class_idx).
    """
    base = model.get_layer("mobilenetv2")
    rescale = model.get_layer("rescale")
    gap = model.get_layer("gap")
    pred = model.get_layer("pred")
    x = tf.convert_to_tensor(img01[None].astype(np.float32))
    with tf.GradientTape() as tape:
        conv = base(rescale(x), training=False)
        tape.watch(conv)
        logits = tf.matmul(gap(conv), pred.kernel) + pred.bias
        probs = tf.nn.softmax(logits)
        if class_idx is None:
            class_idx = int(tf.argmax(probs[0]))
        score = logits[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))[0].numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, probs[0].numpy(), class_idx


def overlay_cam(img01, cam, alpha=0.45):
    """Return uint8 RGB image with a JET heatmap overlay."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - alpha) * img01 + alpha * heat
    return np.uint8(np.clip(out, 0, 1) * 255)


# ------------------------------------------------------- synthetic labeling
def watermark_frame(frame_rgb_uint8, text="SYNTHETIC - AI-generated"):
    """Burn a visible 'synthetic' label into a frame (for display only, never for training)."""
    f = frame_rgb_uint8.copy()
    h, w = f.shape[:2]
    scale = max(0.5, w / 900)
    th = int(34 * scale)
    cv2.rectangle(f, (0, 0), (w, th), (0, 0, 0), -1)
    cv2.putText(f, text, (8, int(th * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * scale, (255, 220, 0), max(1, int(2 * scale)), cv2.LINE_AA)
    return f


def read_video_frames(path, n=None, every=1):
    """Read RGB uint8 frames from a video. n = evenly spaced count (optional)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if n is not None and len(frames) > n:
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames[::every]
