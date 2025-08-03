#dashboard_app.py
import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import sys
from ultralytics import YOLO
import tempfile
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Office Detection Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Minimalistic UI ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-1px);
    }
    
    .confidence-high { color: #28a745; font-weight: 600; }
    .confidence-medium { color: #ffc107; font-weight: 600; }
    .confidence-low { color: #dc3545; font-weight: 600; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
@st.cache_data
def load_config():
    """Load configuration and categories"""
    PROJECT_ROOT = "/home/noor/A/projects/smart_office"
    CUSTOM_MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "smart_office_prompttuned.pt")
    
    # Default categories if categories.py is not available
    default_categories = {
        "person": ["person", "employee", "worker", "staff"],
        "chair": ["office_chair", "chair", "desk_chair", "ergonomic_chair"],
        "monitor": ["monitor", "screen", "display", "computer_monitor"],
        "keyboard": ["keyboard", "wireless_keyboard", "mechanical_keyboard"],
        "laptop": ["laptop", "notebook", "macbook", "computer"],
        "phone": ["phone", "mobile", "smartphone", "cell_phone"]
    }
    
    try:
        # Try to import categories from your existing file
        sys.path.append(os.path.join(PROJECT_ROOT, "src"))
        from categories import categories_data
        return CUSTOM_MODEL_PATH, categories_data
    except:
        return CUSTOM_MODEL_PATH, default_categories

@st.cache_resource
def load_model(model_path):
    """Load the YOLO model with caching"""
    try:
        if os.path.exists(model_path):
            return YOLO(model_path)
        else:
            st.warning("Custom model not found. Loading base YOLO-World model...")
            return YOLO("yolov8x-worldv2.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_category_mappings(categories_data):
    """Create mappings between subcategories and parent categories"""
    detectable_classes = []
    subcategory_to_parent = {}
    
    for parent_category, sub_categories in categories_data.items():
        detectable_classes.extend(sub_categories)
        for sub_category in sub_categories:
            subcategory_to_parent[sub_category] = parent_category
    
    # Color mapping for parent categories - more saturated and distinct
    parent_category_colors = {
        "person": "#FF0040",      # Bright Red
        "chair": "#018c26",       # Bright Green
        "monitor": "#0080FF",     # Bright Blue
        "keyboard": "#FF8000",    # Bright Orange
        "laptop": "#8000FF",      # Bright Purple
        "phone": "#FFFF00"        # Bright Yellow
    }
    
    return detectable_classes, subcategory_to_parent, parent_category_colors

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def draw_predictions_pil(image, results, subcategory_to_parent, parent_category_colors):
    """Draw bounding boxes and labels on image using PIL"""
    draw = ImageDraw.Draw(image)
    
    # Try to use a better font
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    detections = []
    
    if results and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls)
            confidence = float(box.conf.item())  # Convert to Python float
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class name and parent category
            detected_subcategory = results[0].names[class_id]
            parent_category = subcategory_to_parent.get(detected_subcategory, "unknown")
            
            # Get color
            color = parent_category_colors.get(parent_category, "#FF00FF")  # Bright magenta for unknown
            
            # Draw bounding box with thinner lines
            x1, y1, x2, y2 = xyxy
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)  # Reduced from 3 to 2
            
            # Draw label background
            label = f"{parent_category} {confidence:.2f}"
            bbox = draw.textbbox((x1, y1-22), label, font=font_small)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1-22), label, fill="white", font=font_small)
            
            detections.append({
                "parent_category": parent_category,
                "subcategory": detected_subcategory,
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]  # Convert to Python int
            })
    
    return image, detections

def create_detection_charts(detections):
    """Create visualization chart for detections"""
    if not detections:
        return None
    
    df = pd.DataFrame(detections)
    
    # Category count chart only
    category_counts = df['parent_category'].value_counts()
    fig_count = px.bar(
        x=category_counts.index, 
        y=category_counts.values,
        title="Detected Objects by Category",
        color=category_counts.values,
        color_continuous_scale="viridis"
    )
    fig_count.update_layout(
        xaxis_title="Category",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig_count

def prepare_json_data(detections, inference_time, confidence_threshold):
    """Prepare detection data for JSON serialization"""
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": float(inference_time),  # Ensure it's a Python float
        "confidence_threshold": float(confidence_threshold),  # Ensure it's a Python float
        "total_detections": len(detections),
        "detections": []
    }
    
    for i, detection in enumerate(detections):
        detection_json = {
            "id": i + 1,
            "class": detection['parent_category'],
            "subcategory": detection['subcategory'],
            "confidence": float(detection['confidence']),  # Ensure it's a Python float
            "bbox": [int(x) for x in detection['bbox']]  # Ensure all bbox values are Python int
        }
        json_data["detections"].append(detection_json)
    
    return json_data

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Smart Office Detection Dashboard</h1>
        <p>AI-powered object detection for office environments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration and model
    model_path, categories_data = load_config()
    model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check your configuration.")
        return
    
    # Create mappings
    detectable_classes, subcategory_to_parent, parent_category_colors = create_category_mappings(categories_data)
    
    # Set model classes
    model.set_classes(detectable_classes)
    
    # Sidebar for settings (minimal)
    with st.sidebar:
        st.header("Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.4, 
            step=0.05
        )
        
        show_subcategories = st.checkbox("Show Subcategories", value=False)
        
        show_json_data = st.checkbox("Show JSON Data", value=False)
    
    # Image Upload Section
    st.header("Image Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an office image for object detection"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Detection button
        if st.button("Run Detection", use_container_width=True):
            with st.spinner("Processing image..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    # Convert image to RGB if it has transparency (RGBA)
                    if image.mode in ('RGBA', 'LA'):
                        # Create white background
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'RGBA':
                            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                        else:
                            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                        image_to_save = background
                    else:
                        image_to_save = image.convert('RGB')
                    
                    image_to_save.save(tmp_file.name, 'PNG')
                    
                    # Run inference
                    start_time = time.time()
                    results = model.predict(
                        tmp_file.name, 
                        conf=confidence_threshold, 
                        verbose=False
                    )
                    inference_time = time.time() - start_time
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
            
            # Process results
            annotated_image, detections = draw_predictions_pil(
                image.copy(), results, subcategory_to_parent, parent_category_colors
            )
            
            # Store results in session state
            st.session_state.annotated_image = annotated_image
            st.session_state.detections = detections
            st.session_state.inference_time = inference_time
            st.session_state.confidence_threshold = confidence_threshold
    
    # Detection Results Section
    st.header("Detection Results")
    
    if hasattr(st.session_state, 'annotated_image'):
        # Display annotated image
        st.image(
            st.session_state.annotated_image, 
            caption="Detected Objects", 
            use_column_width=True
        )
        
        # Performance metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric(
                "Processing Time", 
                f"{st.session_state.inference_time:.2f}s"
            )
        
        with col_metric2:
            st.metric(
                "Objects Found", 
                len(st.session_state.detections)
            )
        
        with col_metric3:
            if st.session_state.detections:
                # Count unique categories detected
                unique_categories = len(set([d['parent_category'] for d in st.session_state.detections]))
                st.metric("Categories", unique_categories)
            else:
                st.metric("Categories", "0")
    else:
        st.info("Upload an image and run detection to see results")
    
    # Detailed results section
    if hasattr(st.session_state, 'detections') and st.session_state.detections:
        st.header("Detection Details")
        
        # Create detection table
        detection_data = []
        for i, detection in enumerate(st.session_state.detections):
            detection_data.append({
                "ID": i + 1,
                "Category": detection['parent_category'],
                "Subcategory": detection['subcategory'] if show_subcategories else "Hidden",
                "Confidence": f"{detection['confidence']:.3f}"
            })
        
        df_results = pd.DataFrame(detection_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Simple chart
        fig_count = create_detection_charts(st.session_state.detections)
        
        if fig_count:
            st.plotly_chart(fig_count, use_container_width=True)
        
        # Optional JSON data display
        if show_json_data:
            st.subheader("Detection Data (JSON Format)")
            
            # Create JSON structure for all detections using the helper function
            json_data = prepare_json_data(
                st.session_state.detections,
                st.session_state.inference_time,
                st.session_state.confidence_threshold
            )
            
            # Display JSON in expandable code block
            st.json(json_data)
        
        # Download section
        st.subheader("Export Data")
        
        col_download1, col_download2, col_download3 = st.columns(3)
        
        with col_download1:
            # Download annotated image
            img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            st.session_state.annotated_image.save(img_buffer.name)
            
            with open(img_buffer.name, 'rb') as f:
                st.download_button(
                    "Download Annotated Image",
                    f.read(),
                    file_name=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            os.unlink(img_buffer.name)
        
        with col_download2:
            # Download detection data as CSV
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                "Download CSV Statistics ",
                csv_data,
                file_name=f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_download3:
            # Download JSON data using the helper function
            json_data = prepare_json_data(
                st.session_state.detections,
                st.session_state.inference_time,
                st.session_state.confidence_threshold
            )
            
            import json
            json_string = json.dumps(json_data, indent=2)
            
            st.download_button(
                "Download JSON Boxes",
                json_string,
                file_name=f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>Smart Office Detection Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
