import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
import os
import plotly.graph_objects as go

# Get list of files in data folder
data_files = [f for f in os.listdir("data") if f.endswith(".csv")]

# Dropdown to select file
# selected_file = st.selectbox("Select a data file", data_files)

periods = []
lengths = []

for selected_file in data_files:
    # Load data
    df = pd.read_csv(f"data/{selected_file}", sep=";")
    # Estrarre lunghezza dal nome del file (in cm) e convertirla in metri
    default_length_cm = float(selected_file.split("_")[-1].replace(".csv", ""))
    length = default_length_cm / 100

    # Campo di input per modificare la lunghezza
    # length = st.number_input(
    #     "Insert or confirm pendulum length (in meters)", 
    #     value=default_length_m, 
    #     min_value=0.0, 
    #     format="%.4f"
    # )

    # Remove first 29 rows
    df = df.iloc[6:]

    x = df["time"].values.astype(float)
    y = df["period"].values.astype(float)

    # Define exponential function
    def exp_model(x, A, B, C):
        return A * np.exp(B * x) + C

    # Initial guess for parameters
    initial_guess = [1, -0.1, 10]

    # Fit curve
    params, covariance = curve_fit(exp_model, x, y, p0=initial_guess, maxfev=5000)

    # Extract fitted parameters and their errors
    A_fit, B_fit, C_fit = params
    perr = np.sqrt(np.diag(covariance))  # standard deviation (errors)
    A_err, B_err, C_err = perr

    # Generate fitted values
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = exp_model(x_fit, A_fit, B_fit, C_fit)

    print(C_fit**2, length)

# # Display results in Streamlit
# st.title("Exponential Fit for Period Data")
# st.write(
#     f"Fitted equation: ({A_fit:.6f} ± {A_err:.6f}) * exp(({B_fit:.6f} ± {B_err:.6f}) * x) + ({C_fit:.6f} ± {C_err:.6f})"
# )

# # Plot using Streamlit native functions
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
# fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fit', line=dict(color='red')))

# fig.update_layout(title="Period vs Time", xaxis_title="Time", yaxis_title="Period")

# st.plotly_chart(fig)

# # Compute and display g
# g = 4 * math.pi ** 2 * length / (C_fit ** 2)
# st.write(f"Estimated g: {g:.4f} m/s²")
# st.write(length)

# # Save estimated g values
# g_file = "estimated_g_values.csv"
# if os.path.exists(g_file):
#     g_df = pd.read_csv(g_file)
# else:
#     g_df = pd.DataFrame(columns=["Length (m)", "Estimated g (m/s²)"])

# new_entry = pd.DataFrame([[length, g]], columns=["Length (m)", "Estimated g (m/s²)"])
# g_df = pd.concat([g_df, new_entry], ignore_index=True)
# g_df.to_csv(g_file, index=False)

# st.write("Updated estimated g values saved to estimated_g_values.csv")
# st.dataframe(g_df)