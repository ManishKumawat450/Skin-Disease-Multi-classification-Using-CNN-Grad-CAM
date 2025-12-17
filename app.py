import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("skin_resnet50_safe.h5")
base_model = model.get_layer("resnet50")
last_conv_layer = base_model.get_layer("conv5_block3_out")
# Class names
class_names = [
    'Eczema',
    'Warts Molluscum and other Viral Infections',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma (BCC)',
    'Melanocytic Nevi (NV)',
    'Benign Keratosis-like Lesions (BKL)',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections'
]


def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.resnet.preprocess_input(arr)

def gradcam(img_array):
    with tf.GradientTape() as tape:
        conv_out = last_conv_layer(base_model(img_array, training=False))
        tape.watch(conv_out)

        x = model.get_layer("global_average_pooling2d")(conv_out)
        x = model.get_layer("dropout")(x, training=False)
        preds = model.get_layer("dense")(x)

        idx = tf.argmax(preds[0])
        class_score = preds[:, idx]

    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap, int(idx), preds.numpy()[0]


def predict(image):
    img_array = preprocess(image)
    heatmap, idx, probs = gradcam(img_array)

    img_np = np.array(image)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return overlay, class_names[idx]



CSS = """
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    font-family: 'Segoe UI', sans-serif;
}

.container {
    max-width: 1100px;
    margin: auto;
}

.card {
    background: #020617;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #1e293b;
    box-shadow: 0 15px 35px rgba(0,0,0,0.6);
}

.title {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    color: #e5e7eb;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 35px;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 8px;
}

.prediction {
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    color: #22d3ee;
}

.disclaimer {
    font-size: 13px;
    color: #cbd5f5;
    line-height: 1.6;
}
"""

with gr.Blocks(css=CSS) as app:

    with gr.Column(elem_classes="container"):

        gr.HTML("""
        <div class="title">🧬 Skin Disease Classification</div>
        <div class="subtitle">
        CNN (ResNet50) with Grad-CAM Explainability
        </div>
        """)

        # =============================
        # Image Section
        # =============================
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">📤 Upload Skin Lesion Image</div>')
                input_image = gr.Image(type="pil", label=None, elem_classes="card")

            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">🔥 Grad-CAM Attention Map</div>')
                output_image = gr.Image(label=None, elem_classes="card")

        # =============================
        # Buttons (between upload & prediction)
        # =============================
        with gr.Row():
            submit_btn = gr.Button("🔍 Analyze Image", variant="primary")
            clear_btn = gr.Button("🧹 Clear", variant="secondary")

        # =============================
        # Prediction Section
        # =============================
        with gr.Column(elem_classes="card"):
            gr.HTML('<div class="section-title">🩺 Predicted Skin Disease</div>')
            output_text = gr.Textbox(
                placeholder="Prediction will appear here",
                interactive=False,
                label=None,
                elem_classes="prediction"
            )

        

    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

    clear_btn.click(
        fn=lambda: (None, None, ""),
        outputs=[input_image, output_image, output_text]
    )


        # =============================
        # Disclaimer
        # =============================
    gr.HTML("""
        <div class="disclaimer">
        <br>
        <b>Model Scope</b><br>
        The model is trained on <b>only 10 skin disease categories</b>
        and does not represent all real-world skin conditions.
        <br><br>
        <b>Supported Classes:</b>
        <ul>
        <li>Eczema</li>
        <li>Warts Molluscum and other Viral Infections</li>
        <li>Melanoma</li>
        <li>Atopic Dermatitis</li>
        <li>Basal Cell Carcinoma (BCC)</li>
        <li>Melanocytic Nevi (NV)</li>
        <li>Benign Keratosis-like Lesions (BKL)</li>
        <li>Psoriasis / Lichen Planus</li>
        <li>Seborrheic Keratoses and other Benign Tumors</li>
        <li>Tinea Ringworm / Fungal Infections</li>
        </ul>
        </div>
        """)

app.launch(
    server_name="127.0.0.1",
    server_port=7860
)

