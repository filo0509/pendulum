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
selected_file = st.selectbox("Select a data file", data_files)

# Load data
df = pd.read_csv(f"data/{selected_file}", sep=";")
length = float(selected_file.split("_")[-1].replace(".csv", ""))

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

# Extract fitted parameters
A_fit, B_fit, C_fit = params

# Generate fitted values
x_fit = np.linspace(min(x), max(x), 200)
y_fit = exp_model(x_fit, A_fit, B_fit, C_fit)

# Display results in Streamlit
st.title("Exponential Fit for Period Data")
st.write(f"Fitted equation: {A_fit:.3f} * exp({B_fit:.3f} * x) + {C_fit:.3f}")

# Plot using Streamlit native functions
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fit', line=dict(color='red')))

fig.update_layout(title="Period vs Time", xaxis_title="Time", yaxis_title="Period")

st.plotly_chart(fig)

# Compute and display g
g = 4 * math.pi ** 2 * length / (C_fit ** 2)
st.write(f"Estimated g: {g:.4f} m/s²")
st.write(length/100)