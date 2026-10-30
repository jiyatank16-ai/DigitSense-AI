import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Import modular functions from database.py
from database import (
    clear_all_history,
    fetch_history,
    init_db,
    save_prediction,
    update_feedback,
)

# --- Page Configuration ---
st.set_page_config(page_title="AI DigitSense", page_icon="🔢", layout="wide")

# Initialize Database
init_db()


# --- Load CNN Model ---
@st.cache_resource
def load_digit_model():
    try:
        return tf.keras.models.load_model("digit_model.h5")
    except Exception:
        return None


model = load_digit_model()


# --- Image Preprocessing ---
def preprocess_image(image_data):
    img = image_data.convert("L")
    stat = np.mean(np.array(img))
    if stat > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=(0, -1))
    return img_arr


# --- Navigation Sidebar ---
st.sidebar.title("🔢 AI DigitSense")
st.sidebar.caption("BCA 5th Sem AI Minor Project")
menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard",
        "✏️ Draw on Canvas",
        "📁 Upload Image",
        "📜 History & Export",
        "📈 Analytics & Graphs",
    ],
)

# -------------------------------------------------------------
# 1. DASHBOARD
# -------------------------------------------------------------
if menu == "📊 Dashboard":
    st.title("🎯 Project Overview & Dashboard")
    st.write(
        "Welcome to **AI DigitSense**, a handwritten digit recognition system with real-time feedback & SQLite tracking."
    )

    df = fetch_history()
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", len(df))

    if not df.empty:
        col2.metric("Avg Confidence", f"{df['confidence'].mean():.1f}%")
        col3.metric("Most Frequent Digit", int(df["predicted_digit"].mode()[0]))

        correct_count = (df["feedback"] == "Correct").sum()
        total_feedback = (df["feedback"] != "Pending").sum()
        if total_feedback > 0:
            user_acc = (correct_count / total_feedback) * 100
            col4.metric("User Feedback Accuracy", f"{user_acc:.1f}%")
        else:
            col4.metric("User Feedback Accuracy", "No Feedback Yet")
    else:
        col2.metric("Avg Confidence", "N/A")
        col3.metric("Most Frequent Digit", "N/A")
        col4.metric("User Feedback Accuracy", "N/A")

    st.markdown("---")
    st.subheader("Key System Features")
    st.markdown("""
    - **Interactive Canvas & File Upload:** Multi-mode digit recognition.
    - **Feedback Loop:** Users can flag predictions as Correct/Incorrect.
    - **History Management:** Export records to CSV or clear database with one click.
    - **CNN Backbone:** Lightweight 2-layer Convolutional Neural Network.
    """)

# -------------------------------------------------------------
# 2. DRAW ON CANVAS
# -------------------------------------------------------------
elif menu == "✏️ Draw on Canvas":
    st.title("✏️ Interactive Canvas Recognition")
    st.write("Draw a single digit (0-9) inside the black box below:")

    col_canvas, col_result = st.columns([1, 1])

    with col_canvas:
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_draw",
        )

    with col_result:
        if st.button("Predict Canvas Digit"):
            if (
                canvas_result.image_data is not None
                and model is not None
                and np.sum(canvas_result.image_data[:, :, :3]) > 0
            ):
                raw_img = Image.fromarray((canvas_result.image_data).astype("uint8"))
                processed = preprocess_image(raw_img)

                preds = model.predict(processed)
                pred_digit = int(np.argmax(preds))
                confidence = float(np.max(preds) * 100)

                # Save record and get ID for feedback
                rec_id = save_prediction("Canvas", pred_digit, confidence)
                st.session_state["last_canvas_id"] = rec_id
                st.session_state["canvas_digit"] = pred_digit
                st.session_state["canvas_conf"] = confidence
                st.session_state["canvas_preds"] = preds[0]
            elif model is None:
                st.error("Model not found. Please run `train_model.py` first.")
            else:
                st.warning("Please draw a digit before predicting.")

        # Display result & Feedback buttons
        if "last_canvas_id" in st.session_state:
            st.success(f"### Predicted Digit: **{st.session_state['canvas_digit']}**")
            st.info(f"Confidence: **{st.session_state['canvas_conf']:.2f}%**")
            st.bar_chart(
                pd.DataFrame(st.session_state["canvas_preds"], columns=["Probability"])
            )

            st.write("##### Was this prediction correct?")
            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                if st.button("👍 Correct", key="canvas_corr"):
                    update_feedback(st.session_state["last_canvas_id"], "Correct")
                    st.toast("Thank you for your feedback! (Marked Correct)")
            with fb_col2:
                if st.button("👎 Incorrect", key="canvas_incorr"):
                    update_feedback(st.session_state["last_canvas_id"], "Incorrect")
                    st.toast("Feedback updated! (Marked Incorrect)")

# -------------------------------------------------------------
# 3. UPLOAD IMAGE
# -------------------------------------------------------------
elif menu == "📁 Upload Image":
    st.title("📁 Upload Digit Image")
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        col_img, col_res = st.columns(2)
        img = Image.open(uploaded_file)

        with col_img:
            st.image(img, caption="Uploaded Image", width=200)

        with col_res:
            if st.button("Predict Uploaded Image"):
                if model is not None:
                    processed = preprocess_image(img)
                    preds = model.predict(processed)
                    pred_digit = int(np.argmax(preds))
                    confidence = float(np.max(preds) * 100)

                    rec_id = save_prediction("Upload", pred_digit, confidence)
                    st.session_state["last_upload_id"] = rec_id
                    st.session_state["upload_digit"] = pred_digit
                    st.session_state["upload_conf"] = confidence
                    st.session_state["upload_preds"] = preds[0]
                else:
                    st.error("Model not found. Please run `train_model.py` first.")

            if "last_upload_id" in st.session_state:
                st.success(
                    f"### Predicted Digit: **{st.session_state['upload_digit']}**"
                )
                st.info(f"Confidence: **{st.session_state['upload_conf']:.2f}%**")
                st.bar_chart(
                    pd.DataFrame(
                        st.session_state["upload_preds"],
                        columns=["Probability"],
                    )
                )

                st.write("##### Was this prediction correct?")
                fb_u1, fb_u2 = st.columns(2)
                with fb_u1:
                    if st.button("👍 Correct", key="upload_corr"):
                        update_feedback(st.session_state["last_upload_id"], "Correct")
                        st.toast("Thank you for your feedback! (Marked Correct)")
                with fb_u2:
                    if st.button("👎 Incorrect", key="upload_incorr"):
                        update_feedback(st.session_state["last_upload_id"], "Incorrect")
                        st.toast("Feedback updated! (Marked Incorrect)")

# -------------------------------------------------------------
# 4. HISTORY & EXPORT
# -------------------------------------------------------------
elif menu == "📜 History & Export":
    st.title("📜 Prediction History & Data Export")
    df = fetch_history()

    if not df.empty:
        st.dataframe(df, use_container_width=True)

        col_act1, col_act2 = st.columns([1, 4])
        with col_act1:
            if st.button("🗑️ Clear All History"):
                clear_all_history()
                st.rerun()

        with col_act2:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name="digitsense_history.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "No predictions recorded yet. Draw or upload a digit to generate history."
        )

# -------------------------------------------------------------
# 5. ANALYTICS & GRAPHS
# -------------------------------------------------------------
# -------------------------------------------------------------
# 5. ANALYTICS & GRAPHS
# -------------------------------------------------------------
elif menu == "📈 Analytics & Graphs":
    st.title("📈 Prediction Analytics")
    df = fetch_history()

    if not df.empty:
        # Graph 1: Frequency of Each Digit
        st.subheader("1. Digit Frequency (0 to 9)")
        st.caption("How many times each digit has been predicted")
        digit_counts = df["predicted_digit"].value_counts().sort_index()
        st.bar_chart(digit_counts)

        st.markdown("---")

        # Graph 2: User Feedback Breakdown
        st.subheader("2. Model Feedback (Accuracy Check)")
        st.caption("Distribution of Correct vs. Incorrect predictions")
        feedback_counts = df["feedback"].value_counts()
        st.bar_chart(feedback_counts)

    else:
        st.info("No data available yet. Draw or upload a digit to view graphs.")
