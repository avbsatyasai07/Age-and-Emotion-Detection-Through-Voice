import streamlit as st

from gender_detection import predict_gender
from age_detection import predict_age
from emotion_detection import predict_emotion

st.title("Age and Emotion Detection Through Voice")

uploaded_file = st.file_uploader(
    "Upload Voice File",
    type=["wav"]
)

if uploaded_file is not None:

    temp_path = "temp.wav"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    gender = predict_gender(temp_path)

    st.write("Gender:", gender)

    if gender.lower() == "female":

        st.error("Upload Male Voice")

    else:

        age = predict_age(temp_path)

        st.write("Predicted Age:", age)

        if age > 60:

            st.success("Senior Citizen Detected")

            emotion = predict_emotion(temp_path)

            st.write("Emotion:", emotion)

        else:

            st.info("Age below 60")