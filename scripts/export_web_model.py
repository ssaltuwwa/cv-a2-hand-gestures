"""Convert the trained Keras model into an ONNX model for the browser app (web/).

Usage: python scripts/export_web_model.py path/to/gesture_model.keras web/model
Outputs:
  web/model/gesture.onnx   input "image" (1,224,224,3) float32 in [0,1]
                           outputs "features" (1,7,7,1280) and "probs" (1,3)
  web/model/head.json      dense-layer weights, used for the class activation map in JS
"""
import sys, json, os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf2onnx

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
m = tf.keras.models.load_model(src, compile=False)
base, rescale = m.get_layer("mobilenetv2"), m.get_layer("rescale")
gap, pred = m.get_layer("gap"), m.get_layer("pred")
W, b = [v.numpy() for v in pred.weights]          # (1280, 3), (3,)


@tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="image")])
def infer(image):
    feats = base(rescale(image), training=False)
    logits = tf.matmul(tf.reduce_mean(feats, axis=[1, 2]), W) + b
    return {"features": tf.identity(feats, name="features"),
            "probs": tf.nn.softmax(logits, name="probs")}


tf2onnx.convert.from_function(infer, input_signature=infer.input_signature, opset=17,
                              output_path=str(out / "gesture.onnx"))
json.dump({"W": W.round(6).tolist(), "b": b.round(6).tolist(),
           "classes": ["peace-sign", "thumbs-up", "stop"]}, open(out / "head.json", "w"))

# check: ONNX gives the same probabilities as Keras
import onnxruntime as ort
x = np.random.default_rng(0).random((1, 224, 224, 3), dtype=np.float32)
sess = ort.InferenceSession(str(out / "gesture.onnx"))
names = [o.name for o in sess.get_outputs()]
res = dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x})))
p_onnx = [v for k, v in res.items() if v.shape == (1, 3)][0]
p_keras = m(x, training=False).numpy()
print("outputs:", {k: v.shape for k, v in res.items()}, "| input:", sess.get_inputs()[0].name)
print("max |keras - onnx| =", float(np.abs(p_onnx - p_keras).max()))
print("size MB:", round(os.path.getsize(out / "gesture.onnx") / 1e6, 1))
