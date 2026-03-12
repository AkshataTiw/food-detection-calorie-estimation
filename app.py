import streamlit as st

import torch
torch.set_num_threads(1)

from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Food Detection & Calorie Estimator", layout="wide")

st.markdown("<h1 style='text-align:center;'>🍎 AI Food Detection & Calorie Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a meal image to detect food items and estimate calories.</p>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# ---------------- CALORIE DATABASE ----------------
calories = {
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

    # save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25
    )

    result_img = results[0].plot()

    # ---------------- IMAGE LAYOUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(result_img, caption="Detected Image", use_container_width=True)

    boxes = results[0].boxes

    if boxes is None:
        st.warning("No food detected.")

    else:

        names = model.names
        detected = {}
        confidences = {}

        for cls, conf in zip(boxes.cls, boxes.conf):

            label = names[int(cls)]

            if label not in detected:
                detected[label] = 1
                confidences[label] = [float(conf)]
            else:
                detected[label] += 1
                confidences[label].append(float(conf))

        # ---------------- FOOD CARDS ----------------
        st.subheader("Detected Food Items")

        card_cols = st.columns(len(detected))

        i = 0
        for food, count in detected.items():

            avg_conf = np.mean(confidences[food])

            with card_cols[i]:
                st.metric(
                    label=food.capitalize(),
                    value=f"{count} item(s)",
                    delta=f"{avg_conf*100:.1f}% confidence"
                )

            i += 1

        # ---------------- CALORIE TABLE ----------------
        data = []
        total_calories = 0

        for food, count in detected.items():

            cal = calories.get(food,0) * count
            total_calories += cal

            data.append({
                "Food Item": food,
                "Count": count,
                "Calories": cal
            })

        df = pd.DataFrame(data)

        st.subheader("Nutrition Table")
        st.dataframe(df, use_container_width=True)

        st.subheader(f"🔥 Total Estimated Calories: {total_calories} kcal")

        # ---------------- PIE CHART ----------------
        st.subheader("Calorie Distribution")

        fig, ax = plt.subplots()

        ax.pie(
            df["Calories"],
            labels=df["Food Item"],
            autopct="%1.1f%%"
        )

        ax.set_title("Calories by Food Item")

        st.pyplot(fig)
