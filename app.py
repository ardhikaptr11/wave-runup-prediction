import pickle

import numpy as np
import pandas as pd
import streamlit as st

# -*- coding: utf-8 -*-
"""
Dibuat untuk memenuhi syarat kelulusan program S1-Teknik Kelautan

(I Putu Crisna Putra Ardhika - 04311940000096)
"""

pickle_in = open("saved_model/latest_xgbmodel.pkl", "rb")
regressor = pickle.load(pickle_in)

real_data = pd.read_csv("data/data_runup.csv")

html_template = """
<div style="border:3px solid #0069cb;padding:10px">
<h2 style="color:white;text-align:center;">Prediksi Runup Gelombang dengan XGBoost 🌊🌊🌊</h2>
</div>
"""

st.markdown(html_template, unsafe_allow_html=True)

slope = st.number_input(
    "Kemiringan\n1: ",
    min_value=1.0,
    step=0.5,
    value=1.5,
    placeholder="Masukkan disini...",
    format="%.1f",
)
layer = st.number_input(
    "Jumlah Lapisan",
    min_value=1,
    value=1,
    max_value=2,
    placeholder="Masukkan disini...",
)
significant_wave_height = st.number_input(
    "Tinggi Gelombang Signifikan, Hs (m)",
    value=None,
    placeholder="Masukkan disini...",
    format="%.3f",
    max_value=0.1,
)
peak_period = st.number_input(
    "Periode Gelombang, Tp (s)",
    value=None,
    placeholder="Masukkan disini...",
    format="%.3f",
    max_value=2.0,
)


def welcome():
    return "Welcome All"


def predict_wave_runup(slope, layer, significant_wave_height, peak_period):
    """
    Predicts the wave runup using the given parameters.

    Parameters:
    slope (float): The slope of the beach.
    layer (int): The number of layers in the beach.
    significant_wave_height (float): The significant wave height.
    peak_period (float): The peak wave period.

    Returns:
    float: The predicted wave runup.
    """

    # Validasi untuk memastikan semua input terisi
    if None in [slope, layer, significant_wave_height, peak_period]:
        st.warning("Mohon lengkapi semua input untuk melakukan prediksi.")
        return None, None

    # Check if significant_wave_height exists in the CSV data
    if significant_wave_height in real_data["Hs"].values:
        real_result = real_data.loc[
            real_data["Hs"] == significant_wave_height, "runup"
        ].values[0]
    else:
        # If not found, use None for real_result
        real_result = None

    columns = ["slope", "layer", "Hs", "Tp"]
    row = np.array([slope, layer, significant_wave_height, peak_period])
    X = pd.DataFrame([row], columns=columns)
    predicted_result = regressor.predict(X)[0]

    return predicted_result, real_result


predict_trigger = st.button("Lakukan prediksi!")
reset_trigger = st.button("Reset")

if reset_trigger:
    slope = 1.5
    layer = 1
    significant_wave_height = None
    peak_period = None

if predict_trigger:
    predicted_result, real_result = predict_wave_runup(
        slope, layer, significant_wave_height, peak_period
    )

    if predicted_result is not None:
        if real_result is not None:
            difference = abs(predicted_result - real_result)
            container = st.container(border=True)
            container.write("Nilai sesungguhnya: {:.3f} m".format(real_result))
            container.write("Nilai hasil prediksi: {:.3f} m".format(predicted_result))
            container.write("Selisih: {:.3f} m".format(difference))
        else:
            st.success("Nilai hasil prediksi: {:.3f} m".format(predicted_result))
