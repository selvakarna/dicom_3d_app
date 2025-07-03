import streamlit as st
import os
import pydicom
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from zipfile import ZipFile

st.set_page_config(page_title="DICOM 3D Viewer", layout="wide")
st.title("🧠 DICOM 3D Viewer - Upload & Explore")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------- Upload DICOM Files ------------------------
uploaded = st.file_uploader("Upload DICOM ZIP file", type=["zip"])

def load_dicom_slices_from_zip(zip_bytes):
    slices = []
    with ZipFile(zip_bytes) as archive:
        dicom_files = sorted([f for f in archive.namelist() if f.endswith(".dcm")])
        for file in dicom_files:
            with archive.open(file) as f:
                ds = pydicom.dcmread(BytesIO(f.read()))
                slices.append(ds.pixel_array)
    return np.array(slices)

# ---------------------- 3D Viewer ------------------------
def show_3d_volume(volume_np):
    st.subheader("📊 3D View (scroll to zoom, drag to rotate)")
    # Use Plotly Volume rendering
    fig = go.Figure(data=go.Volume(
        x=np.linspace(0, 1, volume_np.shape[2]).repeat(volume_np.shape[1]*volume_np.shape[0]),
        y=np.tile(np.linspace(0, 1, volume_np.shape[1]), volume_np.shape[2]*volume_np.shape[0]),
        z=np.repeat(np.linspace(0, 1, volume_np.shape[0]), volume_np.shape[1]*volume_np.shape[2]),
        value=volume_np.flatten(),
        opacity=0.1,
        surface_count=20,
        colorscale='Gray'
    ))
    fig.update_layout(width=800, height=600, scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Main Flow ------------------------
if uploaded:
    with st.spinner("📦 Extracting and loading DICOM slices..."):
        try:
            vol = load_dicom_slices_from_zip(uploaded)
            st.success(f"Loaded {vol.shape[0]} slices, shape: {vol.shape}")
            show_3d_volume(vol)
        except Exception as e:
            st.error(f"❌ Error loading DICOM: {e}")
