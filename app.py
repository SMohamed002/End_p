from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# تحميل النموذج
model = load_model('models/Model100.h5')

# الفئات المحتملة
classes = ['Pre-B', 'Early Pre-B', 'Pro-B', 'Benign', 'Healthy']

# دالة لإعادة تحجيم الصورة وتعديل قيم البيكسل
def prepare_image(image, target):
    try:
        # إعادة تحجيم الصورة
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target)
        image = img_to_array(image)
        # تعديل قيم البيكسل
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise RuntimeError(f"Error in prepare_image: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        try:
            # التحقق من أن طلب يحتوي على ملف
            if 'image' not in request.files:
                return jsonify({"error": "No image provided"}), 400

            # الحصول على الملف من الطلب
            file = request.files["image"]

            # قراءة الصورة
            image = Image.open(io.BytesIO(file.read()))

            # تجهيز الصورة
            prepared_image = prepare_image(image, target=(224, 224))

            # القيام بالتنبؤ باستخدام النموذج
            preds = model.predict(prepared_image)

            # استخراج التصنيف من التنبؤات
            pred_class_index = np.argmax(preds, axis=1)[0]
            pred_class = classes[pred_class_index]

            # إعادة التصنيف كاستجابة
            return jsonify({"class": pred_class})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid request method"}), 405

if __name__ == "__main__":
    app.run(debug=True)
