import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile

# ---------------- PAGE ----------------
st.set_page_config(page_title="Food Detection", layout="centered")

st.title("🍎 AI Food Detection & Calorie Estimator")
st.write("Upload an image to detect food items and estimate calories.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- CALORIES PER ITEM ----------------
calories_per_item = {
    "apple":95,
    "banana":105,
    "grape":3,
    "guava":68,
    "mango":150,
    "orange":62,
    "pineapple":82,
    "pomegranate":105,
    "watermelon":85,
    "almond":7,
    "bread":80,
    "cashew":9,
    "cucumber":16,
    "dates":66,
    "egg":78,
    "onion":44,
    "potato":163,
    "carrot":25,
    "corn":96,
    "tomato":22
}

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Prediction (same as Kaggle)
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25
    )

    # Show detection
    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_container_width=True)

    boxes = results[0].boxes

    if boxes is None:
        st.warning("No objects detected")

    else:

        names = model.names
        detected_counts = {}

        for cls in boxes.cls:
            label = names[int(cls)]

            if label not in detected_counts:
                detected_counts[label] = 1
            else:
                detected_counts[label] += 1

        # ---------------- CALORIE CALCULATION ----------------
        data = []
        total_calories = 0

        for food, count in detected_counts.items():

            cal_per_item = calories_per_item.get(food,0)
            cal = cal_per_item * count

            total_calories += cal

            data.append({
                "Food Item": food,
                "Count": count,
                "Calories (kcal)": cal
            })

        df = pd.DataFrame(data)

        st.subheader("📊 Detected Food Items")
        st.dataframe(df, use_container_width=True)

        st.subheader(f"🔥 Total Estimated Calories: {total_calories} kcal")