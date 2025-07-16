import os
import io
import zipfile
import tempfile
import requests
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import pydicom
import numpy as np
import plotly.graph_objects as go
import nibabel as nib
import glob
from PIL import Image

st.set_page_config(page_title="DICOM 3D Viewer", layout="wide")
st.title("🧠 DICOM 3D Viewer - Upload, Explore & Export")

UPLOAD_DIR = "uploads"
EXPORT_DIR = "exports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------------- Upload DICOM Files ------------------------
uploaded = st.file_uploader("Upload DICOM ZIP or Single DCM file", type=["zip", "dcm"])
folder_path = st.text_input("Or enter folder path containing DICOM files")

# ---------------------- Load DICOM Volume ------------------------
def load_dicom_slices_from_zip(zip_bytes):
    slices = []
    with zipfile.ZipFile(zip_bytes) as archive:
        dicom_files = sorted([f for f in archive.namelist() if not f.endswith("/")])
        for file in dicom_files:
            try:
                with archive.open(file) as f:
                    ds = pydicom.dcmread(io.BytesIO(f.read()), force=True)
                    if hasattr(ds, "PixelData") and hasattr(ds, "pixel_array"):
                        slices.append(ds.pixel_array)
                        st.info(f"✅ Loaded: {file} - shape: {ds.pixel_array.shape}")
                    else:
                        st.warning(f"⚠️ Skipped (no pixel data): {file}")
            except Exception as e:
                st.warning(f"❌ Skipped {file}: {e}")
    if len(slices) == 0:
        raise ValueError("No valid DICOM slices with pixel data found in ZIP.")
    return np.array(slices)

def load_dicom_slices_from_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds.pixel_array)
        except Exception as e:
            st.warning(f"Skipping file {f}: {e}")
    if len(slices) == 0:
        raise ValueError("No valid DICOM slices found in folder.")
    return np.array(slices)

# ---------------------- Show 3D Volume ------------------------
def show_3d_volume(volume_np):
    if volume_np.ndim != 3 or volume_np.size == 0:
        st.warning(f"Expected a 3D volume, but got shape {volume_np.shape}. Skipping 3D view.")
        return

    st.subheader("📊 3D View (scroll to zoom, drag to rotate)")

    threshold_min = st.slider("Voxel Intensity Min Threshold", int(volume_np.min()), int(volume_np.max()), int(volume_np.min()))
    threshold_max = st.slider("Voxel Intensity Max Threshold", int(volume_np.min()), int(volume_np.max()), int(volume_np.max()))

    # Apply thresholding
    volume_thresh = np.clip(volume_np, threshold_min, threshold_max)
    values = volume_thresh.flatten()
    values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-5)

    x = np.linspace(0, 1, volume_np.shape[2]).repeat(volume_np.shape[1] * volume_np.shape[0])
    y = np.tile(np.linspace(0, 1, volume_np.shape[1]), volume_np.shape[2] * volume_np.shape[0])
    z = np.repeat(np.linspace(0, 1, volume_np.shape[0]), volume_np.shape[1] * volume_np.shape[2])

    fig = go.Figure(data=go.Volume(
        x=x,
        y=y,
        z=z,
        value=values,
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

# ---------------------- Slice Viewer ------------------------
def normalize_slice(slice_2d):
    if slice_2d.max() == slice_2d.min():
        return np.zeros_like(slice_2d, dtype=np.uint8)
    norm = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255).astype(np.uint8)
    return norm

def show_slice_views(volume_np):
    st.subheader(":camera: Slice Viewer (Axial / Coronal / Sagittal)")
    axis = st.radio("Choose orientation:", ["Axial", "Coronal", "Sagittal"])

    try:
        if axis == "Axial":
            if volume_np.shape[0] < 1:
                st.warning("No axial slices found.")
                return
            max_index = volume_np.shape[0] - 1
            index = st.slider("Slice Index", 0, max_index, max_index // 2)
            img = normalize_slice(volume_np[index, :, :])
            st.image(img, caption=f"Axial Slice {index}", use_column_width=True)

        elif axis == "Coronal":
            if volume_np.shape[1] < 1:
                st.warning("No coronal slices found.")
                return
            max_index = volume_np.shape[1] - 1
            index = st.slider("Slice Index", 0, max_index, max_index // 2)
            img = normalize_slice(volume_np[:, index, :])
            st.image(img, caption=f"Coronal Slice {index}", use_column_width=True)

        elif axis == "Sagittal":
            if volume_np.shape[2] < 1:
                st.warning("No sagittal slices found.")
                return
            max_index = volume_np.shape[2] - 1
            index = st.slider("Slice Index", 0, max_index, max_index // 2)
            img = normalize_slice(volume_np[:, :, index])
            st.image(img, caption=f"Sagittal Slice {index}", use_column_width=True)

    except Exception as e:
        st.error(f"Error showing slice: {e}")

# ---------------------- Export to NIfTI ------------------------
def export_nifti(volume_np):
    nifti_img = nib.Nifti1Image(volume_np.astype(np.int16), affine=np.eye(4))
    nifti_path = os.path.join(EXPORT_DIR, "volume.nii.gz")
    nib.save(nifti_img, nifti_path)
    return nifti_path

# ---------------------- Main Flow ------------------------
volume_data = None

if uploaded:
    if uploaded.name.endswith(".zip"):
        with st.spinner(":package: Extracting and loading DICOM slices from ZIP..."):
            try:
                volume_data = load_dicom_slices_from_zip(uploaded)
                st.success(f"Loaded {volume_data.shape[0]} slices | Shape: {volume_data.shape}")
            except Exception as e:
                st.error(f"❌ Error loading DICOM from zip: {e}")

    elif uploaded.name.endswith(".dcm"):
        with st.spinner(":package: Loading single DICOM file..."):
            try:
                ds = pydicom.dcmread(uploaded)
                volume_data = np.expand_dims(ds.pixel_array, axis=0)
                st.success(f"Loaded 1 slice | Shape: {volume_data.shape}")
            except Exception as e:
                st.error(f"❌ Error loading DICOM file: {e}")

elif folder_path:
    with st.spinner(":file_folder: Loading DICOM slices from folder..."):
        try:
            volume_data = load_dicom_slices_from_folder(folder_path)
            st.success(f"Loaded {volume_data.shape[0]} slices | Shape: {volume_data.shape}")
        except Exception as e:
            st.error(f"❌ Error loading DICOM from folder: {e}")

if volume_data is not None:
    st.write("Volume shape:", volume_data.shape)
    if volume_data.ndim == 3 and volume_data.size > 0:
        show_3d_volume(volume_data)
        show_slice_views(volume_data)

        if st.button(":arrow_down: Export as NIfTI (.nii.gz)"):
            nifti_file = export_nifti(volume_data)
            with open(nifti_file, "rb") as f:
                st.download_button("Download NIfTI", f, file_name="volume.nii.gz")
    else:
        st.error("❌ DICOM volume is not valid or empty. Please upload a valid 3D series.")
