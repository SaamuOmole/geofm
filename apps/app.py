"""
Geospatial AI Water Segmentation Application
=============================================
A Streamlit web application for water body detection using Sentinel satellite imagery
and a fine-tuned geospatial foundation model.

Installation:
pip install streamlit streamlit-folium folium plotly

Run:
streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path
import json
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO

# from geofm import (
#     find_and_download_sentinel_images,
#     predict_pair, visualize_sentinel2,
#     visualize_sentinel1
# )

from geofm import (
    get_sentinel_images,
    model_inference_pair,
    visualize_images
)

# Page configuration
st.set_page_config(
    page_title="Water Segmentation AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
    }
    .status-error {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
    }
    .status-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    .status-info {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'bbox' not in st.session_state:
    st.session_state.bbox = None
if 'download_results' not in st.session_state:
    st.session_state.download_results = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'selected_preset' not in st.session_state:
    st.session_state.selected_preset = None

# Location presets
LOCATION_PRESETS = {
    "London, UK": {"bbox": [-0.15, 51.48, -0.05, 51.53], "zoom": 11},
    "Venice, Italy": {"bbox": [12.30, 45.41, 12.38, 45.46], "zoom": 12},
    "Netherlands Coast": {"bbox": [4.2, 51.9, 4.4, 52.0], "zoom": 11},
    "Bangladesh Delta": {"bbox": [90.0, 22.0, 91.0, 23.0], "zoom": 9},
    "Amazon River": {"bbox": [-60.0, -3.2, -59.0, -2.5], "zoom": 9},
    "Custom": {"bbox": None, "zoom": 10}
}

def create_map(bbox=None, center=None, zoom=10):
    """Create an interactive folium map"""
    if center is None:
        center = [51.5, -0.1]  # Default: London
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )
    
    # Add different tile layers with proper attribution
    folium.TileLayer(
        tiles='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors',
        name='Stamen Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors',
        name='Stamen Toner',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB positron',
        name='CartoDB Positron',
        overlay=False,
        control=True
    ).add_to(m)
    
    if bbox:
        # Draw bounding box
        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="#3b82f6",
            fill=True,
            fillColor="#3b82f6",
            fillOpacity=0.2,
            popup="Selected Area"
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def display_status(message: str, status: str = "info"):
    """Display a styled status message"""
    status_class = f"status-{status}"
    st.markdown(
        f'<div class="status-box {status_class}">{message}</div>',
        unsafe_allow_html=True
    )

# def image_to_base64(img_path):
#     """Convert image to base64 for display"""
#     try:
#         img = Image.open(img_path)
#         buffered = BytesIO()
#         img.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode()
#     except:
#         return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🌊 Water Segmentation AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Detect water bodies using Sentinel satellite imagery and AI-powered segmentation</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Sentinel-2.jpg/400px-Sentinel-2.jpg", 
             caption="Sentinel-2 Satellite", use_container_width=True)
    
    st.markdown("### 📋 Workflow Steps")
    steps = [
        ("1️⃣", "Select Location", st.session_state.step >= 1),
        ("2️⃣", "Configure Parameters", st.session_state.step >= 2),
        ("3️⃣", "Download Satellite Data", st.session_state.step >= 3),
        ("4️⃣", "Run AI Prediction", st.session_state.step >= 4),
        ("5️⃣", "View Results", st.session_state.step >= 5)
    ]
    
    for emoji, step_name, completed in steps:
        if completed:
            st.markdown(f"{emoji} **{step_name}** ✅")
        else:
            st.markdown(f"{emoji} {step_name}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This application uses:
    - **Sentinel-2** optical imagery
    - **Sentinel-1** SAR imagery
    - **TerraMind** geospatial foundation model
    - Fine-tuned on Sen1Floods11 dataset
    """)
    
    # Model configuration
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    
    checkpoint_path = st.text_input(
        "Model Checkpoint Path",
        value="/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt",
        help="Path to your trained model checkpoint"
    )
    
    if not Path(checkpoint_path).exists():
        st.error("⚠️ Checkpoint file not found!")
    else:
        st.success("✅ Model checkpoint found")

# ============================================================================
# STEP 1: LOCATION SELECTION
# ============================================================================

st.markdown("## 🗺️ Step 1: Select Your Area of Interest")

col1, col2 = st.columns([2, 1])

with col1:
    # Location preset selector
    preset = st.selectbox(
        "Choose a location preset or select custom:",
        list(LOCATION_PRESETS.keys()),
        key="location_preset"
    )
    
    if preset != "Custom":
        preset_data = LOCATION_PRESETS[preset]
        if st.session_state.selected_preset != preset:
            st.session_state.bbox = preset_data["bbox"]
            st.session_state.selected_preset = preset
    
    # Custom bbox input
    if preset == "Custom":
        st.markdown("### Enter Bounding Box Coordinates")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            min_lon = st.number_input("Min Longitude", value=-0.15, format="%.4f")
        with col_b:
            min_lat = st.number_input("Min Latitude", value=51.48, format="%.4f")
        with col_c:
            max_lon = st.number_input("Max Longitude", value=-0.05, format="%.4f")
        with col_d:
            max_lat = st.number_input("Max Latitude", value=51.53, format="%.4f")
        
        st.session_state.bbox = [min_lon, min_lat, max_lon, max_lat]
    
    # Display map
    if st.session_state.bbox:
        center = [
            (st.session_state.bbox[1] + st.session_state.bbox[3]) / 2,
            (st.session_state.bbox[0] + st.session_state.bbox[2]) / 2
        ]
        zoom = LOCATION_PRESETS.get(preset, {"zoom": 10})["zoom"]
        map_obj = create_map(st.session_state.bbox, center, zoom)
        st_folium(map_obj, width=700, height=400)

with col2:
    st.markdown("### 📅 Date Range")
    
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )
    with col_date2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    st.markdown("### ☁️ Quality Settings")
    max_cloud_cover = st.slider(
        "Max Cloud Cover (%)",
        min_value=0,
        max_value=100,
        value=30,
        help="Filter images by cloud cover percentage"
    )
    
    tile_size = st.selectbox(
        "Tile Size (pixels)",
        [256, 512, 1024],
        index=1,
        help="Size of the downloaded image tiles"
    )

if st.button("✅ Confirm Location & Proceed", type="primary", use_container_width=True):
    st.session_state.step = 2
    st.success("Location configured! Scroll down to download satellite data.")

# ============================================================================
# STEP 2: DOWNLOAD SATELLITE DATA
# ============================================================================

if st.session_state.step >= 2:
    st.markdown("---")
    st.markdown("## 📡 Step 2: Download Satellite Imagery")
    
    output_dir = st.text_input(
        "Output Directory for Downloaded Images",
        value="./satellite_data_download",
        help="Directory where satellite images will be saved"
    )
    
    if st.button("🚀 Download Satellite Data", type="primary", use_container_width=True):
        st.session_state.step = 3
        
        with st.spinner("🔍 Searching for satellite imagery..."):
            try:
                # Create output directory
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Download satellite data
                location_str = f"{st.session_state.bbox[0]},{st.session_state.bbox[1]},{st.session_state.bbox[2]},{st.session_state.bbox[3]}"
                
                results = find_and_download_sentinel_images(
                    location=location_str,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    output_dir=output_dir,
                    max_cloud_cover=max_cloud_cover,
                    tile_size=tile_size
                )
                
                st.session_state.download_results = results
                
                # Display results
                if results.get('sentinel2_paths') and results.get('sentinel1_paths'):
                    display_status("✅ Successfully downloaded satellite imagery!", "success")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentinel-2 Images", len(results['sentinel2_paths']))
                    with col2:
                        st.metric("Sentinel-1 Images", len(results['sentinel1_paths']))
                    
                    # Show metadata
                    with st.expander("📊 View Image Metadata"):
                        st.json(results['metadata'])
                    
                    st.session_state.step = 4
                    
                else:
                    display_status("⚠️ No images found matching your criteria. Try adjusting the date range or cloud cover threshold.", "warning")
                    
            except Exception as e:
                display_status(f"❌ Error downloading data: {str(e)}", "error")
                st.exception(e)

# ============================================================================
# STEP 3: VISUALIZE DOWNLOADED IMAGES
# ============================================================================

if st.session_state.step >= 3 and st.session_state.download_results:
    st.markdown("---")
    st.markdown("## 🖼️ Step 3: Preview Downloaded Images")
    
    results = st.session_state.download_results
    
    if results.get('sentinel2_paths'):
        st.markdown("### Sentinel-2 Optical Imagery")
        s2_path = results['sentinel2_paths'][0]
        
        with st.spinner("Rendering Sentinel-2 visualization..."):
            try:
                viz_path = Path(output_dir) / "s2_preview.png"
                visualize_sentinel2(s2_path, save_path=str(viz_path))
                st.image(str(viz_path), use_container_width=True)
            except Exception as e:
                st.error(f"Could not visualize S2: {e}")
    
    if results.get('sentinel1_paths'):
        st.markdown("### Sentinel-1 SAR Imagery")
        s1_path = results['sentinel1_paths'][0]
        
        with st.spinner("Rendering Sentinel-1 visualization..."):
            try:
                viz_path = Path(output_dir) / "s1_preview.png"
                visualize_sentinel1(s1_path, save_path=str(viz_path))
                st.image(str(viz_path), use_container_width=True)
            except Exception as e:
                st.error(f"Could not visualize S1: {e}")

# ============================================================================
# STEP 4: RUN AI PREDICTION
# ============================================================================

if st.session_state.step >= 4 and st.session_state.download_results:
    st.markdown("---")
    st.markdown("## 🤖 Step 4: Run AI Water Segmentation")
    
    prediction_output_dir = st.text_input(
        "Prediction Output Directory",
        value="./predictions_output",
        help="Directory where prediction results will be saved"
    )
    
    if st.button("🎯 Run Prediction Model", type="primary", use_container_width=True):
        st.session_state.step = 5
        
        with st.spinner("🧠 Running AI model inference... This may take a minute."):
            try:
                # Create output directory
                Path(prediction_output_dir).mkdir(parents=True, exist_ok=True)
                
                results = st.session_state.download_results
                s2_path = results['sentinel2_paths'][0]
                s1_path = results['sentinel1_paths'][0]
                
                # Run prediction
                prediction_results = predict_pair(
                    checkpoint_path=checkpoint_path,
                    s2_path=s2_path,
                    s1_path=s1_path,
                    out_dir=prediction_output_dir,
                    show_plot=False
                )
                
                st.session_state.prediction_results = prediction_results
                
                display_status("✅ Prediction completed successfully!", "success")
                
                # Calculate statistics
                raw_pred = prediction_results['raw_pred']
                water_pixels = (raw_pred == 1).sum()
                total_pixels = raw_pred.size
                water_percentage = (water_pixels / total_pixels) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Water Pixels", f"{water_pixels:,}")
                with col2:
                    st.metric("Total Pixels", f"{total_pixels:,}")
                with col3:
                    st.metric("Water Coverage", f"{water_percentage:.2f}%")
                
            except Exception as e:
                display_status(f"❌ Error running prediction: {str(e)}", "error")
                st.exception(e)

# ============================================================================
# STEP 5: DISPLAY RESULTS
# ============================================================================

if st.session_state.step >= 5 and st.session_state.prediction_results:
    st.markdown("---")
    st.markdown("## 📊 Step 5: View Prediction Results")
    
    pred_results = st.session_state.prediction_results
    
    tab1, tab2, tab3 = st.tabs(["🖼️ Visualizations", "📈 Analytics", "💾 Downloads"])
    
    with tab1:
        st.markdown("### Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if pred_results.get('figure_path') and Path(pred_results['figure_path']).exists():
                st.image(pred_results['figure_path'], caption="Side-by-side comparison", use_container_width=True)
        
        with col2:
            if pred_results.get('overlay_path') and Path(pred_results['overlay_path']).exists():
                st.image(pred_results['overlay_path'], caption="Water overlay on RGB", use_container_width=True)
    
    with tab2:
        st.markdown("### Water Detection Analytics")
        
        raw_pred = pred_results['raw_pred']
        water_mask = (raw_pred == 1)
        
        # Create histogram
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Non-Water', 'Water'],
            y=[(~water_mask).sum(), water_mask.sum()],
            marker_color=['#94a3b8', '#3b82f6']
        ))
        fig.update_layout(
            title="Pixel Classification Distribution",
            xaxis_title="Class",
            yaxis_title="Number of Pixels",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show confidence metrics (if available)
        st.markdown("#### Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Model**: TerraMind Small")
        with col2:
            st.info(f"**Dataset**: Sen1Floods11")
        with col3:
            st.info(f"**Classes**: Water / Non-Water")
    
    with tab3:
        st.markdown("### Download Results")
        
        if pred_results.get('overlay_path'):
            with open(pred_results['overlay_path'], 'rb') as f:
                st.download_button(
                    label="📥 Download Overlay Image",
                    data=f,
                    file_name="water_overlay.png",
                    mime="image/png"
                )
        
        if pred_results.get('figure_path'):
            with open(pred_results['figure_path'], 'rb') as f:
                st.download_button(
                    label="📥 Download Comparison Figure",
                    data=f,
                    file_name="comparison.png",
                    mime="image/png"
                )
        
        # Export metadata
        metadata = {
            "location": st.session_state.bbox,
            "date_range": [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
            "water_coverage": f"{(water_mask.sum() / water_mask.size * 100):.2f}%",
            "timestamp": datetime.now().isoformat()
        }
        
        st.download_button(
            label="📥 Download Metadata (JSON)",
            data=json.dumps(metadata, indent=2),
            file_name="analysis_metadata.json",
            mime="application/json"
        )

# Reset button
st.markdown("---")
if st.button("🔄 Start New Analysis", use_container_width=True):
    st.session_state.step = 1
    st.session_state.bbox = None
    st.session_state.download_results = None
    st.session_state.prediction_results = None
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p>Built using Streamlit</p>
    <p>Powered by Microsoft Planetary Computer STAC API</p>
</div>
""", unsafe_allow_html=True)