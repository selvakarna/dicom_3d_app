import streamlit as st
import os
import pydicom
import numpy as np
from io import BytesIO
from zipfile import ZipFile
import plotly.graph_objects as go
import nibabel as nib
import glob

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
    with ZipFile(zip_bytes) as archive:
        dicom_files = sorted([f for f in archive.namelist() if f.endswith(".dcm")])
        for file in dicom_files:
            with archive.open(file) as f:
                ds = pydicom.dcmread(BytesIO(f.read()))
                slices.append(ds.pixel_array)
    return np.array(slices)

def load_dicom_slices_from_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        slices.append(ds.pixel_array)
    return np.array(slices)

# ---------------------- Show 3D Volume ------------------------
def show_3d_volume(volume_np):
    st.subheader("📊 3D View (scroll to zoom, drag to rotate)")
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

# ---------------------- Slice Viewer ------------------------
def show_slice_views(volume_np):
    st.subheader(":camera: Slice Viewer (Axial / Coronal / Sagittal)")
    axis = st.radio("Choose orientation:", ["Axial", "Coronal", "Sagittal"])

    if axis == "Axial":
        max_index = volume_np.shape[0] - 1
        index = st.slider("Slice Index", 0, max_index, max_index // 2)
        st.image(volume_np[index, :, :], caption=f"Axial Slice {index}", use_column_width=True)

    elif axis == "Coronal":
        max_index = volume_np.shape[1] - 1
        index = st.slider("Slice Index", 0, max_index, max_index // 2)
        st.image(volume_np[:, index, :], caption=f"Coronal Slice {index}", use_column_width=True)

    elif axis == "Sagittal":
        max_index = volume_np.shape[2] - 1
        index = st.slider("Slice Index", 0, max_index, max_index // 2)
        st.image(volume_np[:, :, index], caption=f"Sagittal Slice {index}", use_column_width=True)

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
                volume_data = np.expand_dims(ds.pixel_array, axis=0)  # Make it 3D-compatible
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
    show_3d_volume(volume_data)
    show_slice_views(volume_data)

    if st.button(":arrow_down: Export as NIfTI (.nii.gz)"):
        nifti_file = export_nifti(volume_data)
        with open(nifti_file, "rb") as f:
            st.download_button("Download NIfTI", f, file_name="volume.nii.gz")
