#evaluate.py
import os
import sys
from ultralytics import YOLO
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- Configuration Paths ---
# Define the base directory for your project. Adjust if your structure is different.
# This script assumes it's run from within the 'src' directory or similar,
# and that 'categories.py' is accessible.
PROJECT_ROOT = "/home/noor/A/projects/smart_office"
CATEGORIES_FILE_PATH = os.path.join(os.path.dirname(__file__), 'categories.py') # Assumes categories.py is in the same directory

# Define the path for the custom model that we will evaluate
CUSTOM_MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models")
CUSTOM_MODEL_NAME = "smart_office_prompttuned.pt"
CUSTOM_MODEL_PATH = os.path.join(CUSTOM_MODEL_DIR, CUSTOM_MODEL_NAME)

# Define the directory containing images for inference
IMAGE_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Define the directory for saving evaluation results
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "results")


# --- Import Categories Data ---
# Assume categories.py exists and contains a dictionary named categories_data
try:
    # Dynamically add the directory of categories.py to sys.path if it's not current
    if os.path.dirname(CATEGORIES_FILE_PATH) not in sys.path:
        sys.path.append(os.path.dirname(CATEGORIES_FILE_PATH))
    from categories import categories_data
    print("Successfully imported categories from 'categories.py'.")
except ImportError:
    print(f"Error: '{CATEGORIES_FILE_PATH}' not found or 'categories_data' dictionary missing.")
    print("Please ensure 'categories.py' exists and contains the 'categories_data' dictionary.")
    sys.exit(1)

# --- Prepare Class Lists and Mappings ---
# Prepare a flat list of all sub-categories that the YOLO-World model was tuned to detect.
detectable_classes = []
# Create a reverse mapping from subcategory to parent category.
# This is crucial for displaying only parent categories in the output and annotations.
subcategory_to_parent = {}

for parent_category, sub_categories in categories_data.items():
    detectable_classes.extend(sub_categories)
    for sub_category in sub_categories:
        subcategory_to_parent[sub_category] = parent_category

# --- Define Colors for Parent Categories ---
# Assign different colors to each parent category for visual distinction.
# Colors are in BGR format for OpenCV (Blue, Green, Red).
parent_category_colors = {
    "person": (255, 0, 0),    # Blue
    "chair": (0, 255, 0),     # Green
    "monitor": (0, 0, 255),   # Red
    "keyboard": (255, 255, 0),# Cyan
    "laptop": (0, 255, 255),  # Yellow
    "phone": (255, 0, 255),   # Magenta
    "unknown": (128, 128, 128) # Grey for unknown categories
}


def create_detection_heatmap(detections_data, output_path):
    """Create a heatmap showing detection confidence distribution"""
    if not detections_data["detections"]:
        print(f"No detections data for heatmap, skipping: {output_path}")
        return
    
    # Extract class names and confidences (these will now be parent categories)
    classes = [det["class"] for det in detections_data["detections"]]
    confidences = [det["confidence"] for det in detections_data["detections"]]
    
    # Create confidence bins
    bins = np.arange(0, 1.1, 0.1)
    class_names = list(set(classes)) # Unique parent class names
    
    # Create matrix for heatmap
    heatmap_data = np.zeros((len(class_names), len(bins)-1))
    
    for i, class_name in enumerate(class_names):
        class_confidences = [conf for cls, conf in zip(classes, confidences) if cls == class_name]
        hist, _ = np.histogram(class_confidences, bins=bins)
        heatmap_data[i] = hist
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data,
                xticklabels=[f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
                yticklabels=class_names,
                annot=True,
                fmt='.0f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Detections'})
    
    plt.title('Detection Confidence Distribution Heatmap (Parent Categories)', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Range', fontsize=12)
    plt.ylabel('Object Classes', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_batch_analysis_charts(batch_results, output_path):
    """Create comprehensive batch processing analysis charts"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    images = [r["image"] for r in batch_results]
    detections = [r["detections"] for r in batch_results]
    times = [r["time"] for r in batch_results]
    
    # 1. Detections per Image
    ax1.plot(range(len(images)), detections, marker='o', linewidth=2, markersize=6)
    ax1.set_title('Detections per Image', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Number of Detections')
    ax1.grid(True, alpha=0.3)
    
    # 2. Processing Time Distribution
    ax2.hist(times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.3f}s')
    ax2.set_title('Processing Time Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Processing Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter: Time vs Detections
    ax3.scatter(times, detections, alpha=0.7, s=50, color='orange')
    ax3.set_title('Processing Time vs Number of Detections', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Processing Time (seconds)')
    ax3.set_ylabel('Number of Detections')
    
    # Add correlation coefficient
    correlation = np.corrcoef(times, detections)[0, 1] if len(times) > 1 else 0
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Performance
    cumulative_detections = np.cumsum(detections)
    cumulative_time = np.cumsum(times)
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(range(len(images)), cumulative_detections, 'b-', linewidth=2, label='Cumulative Detections')
    line2 = ax4_twin.plot(range(len(images)), cumulative_time, 'r-', linewidth=2, label='Cumulative Time')
    
    ax4.set_title('Cumulative Performance', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Image Index')
    ax4.set_ylabel('Cumulative Detections', color='b')
    ax4_twin.set_ylabel('Cumulative Time (s)', color='r')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right')
    
    plt.suptitle('Batch Processing Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_detection_patterns_visualization(detection_summary, output_path):
    """Create a visualization of detection patterns"""
    if not detection_summary:
        print(f"No detection summary for patterns visualization, skipping: {output_path}")
        return
    
    # Count class occurrences
    class_counts = Counter()
    confidence_ranges = defaultdict(list)
    
    for data in detection_summary.values():
        for det in data.get("detections", []):
            class_name = det["class"] # This will be the parent category
            confidence = det["confidence"]
            class_counts[class_name] += 1
            confidence_ranges[class_name].append(confidence)
    
    if not class_counts:
        print(f"No class counts for patterns visualization, skipping: {output_path}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Class Distribution Pie Chart
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = sns.color_palette("Set3", len(classes))
    
    wedges, texts, autotexts = ax1.pie(counts, labels=classes, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax1.set_title('Class Distribution (Parent Categories)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_weight('bold')
    
    # 2. Box Plot of Confidence by Class
    confidence_data = []
    class_labels = []
    
    for class_name in classes:
        confidence_data.extend(confidence_ranges[class_name])
        class_labels.extend([class_name] * len(confidence_ranges[class_name]))
    
    df = pd.DataFrame({'Class': class_labels, 'Confidence': confidence_data})
    sns.boxplot(data=df, x='Class', y='Confidence', ax=ax2)
    ax2.set_title('Confidence Distribution by Class (Parent Categories)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_single_image_detection():
    """Test detection on a single image and annotate with parent categories"""
    print("🔍 Single Image Detection Test")
    print("="*50)
    
    # Paths from configuration
    model_path = CUSTOM_MODEL_PATH
    datasets_dir = Path(IMAGE_DATASET_DIR)
    results_dir = Path(RESULTS_BASE_DIR)
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Loaded model: {Path(model_path).name}")
    
    # Find test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(datasets_dir.rglob(ext)))
    
    if not test_images:
        print(f"❌ No images found in {datasets_dir}")
        return
    
    # Use first image for single test
    test_image_path = test_images[0]
    print(f"🖼️ Testing with: {test_image_path}")
    
    # Run detection
    start_time = time.time()
    # verbose=False to suppress detailed YOLO output during prediction
    results = model.predict(str(test_image_path), conf=0.3, verbose=False) 
    inference_time = time.time() - start_time
    
    print(f"⚡ Inference completed in {inference_time:.3f} seconds")
    
    if results and len(results) > 0:
        # Load the image using OpenCV for drawing annotations.
        img = cv2.imread(str(test_image_path))
        if img is None:
            print(f"Warning: Could not load image {test_image_path}. Skipping annotation.")
            return

        # Prepare detection data
        detections_data = {
            "image_path": str(test_image_path),
            "model_used": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "inference_time": inference_time,
            "total_detections": 0, # Will be updated after processing boxes
            "detections": []
        }
        
        if results[0].boxes is not None:
            detections_count = 0
            print(f"🎯 Detected objects (Parent Categories):")
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf.item())
                # Get the class name (which will be a subcategory) from the model's names attribute
                detected_subcategory_name = model.names[class_id]
                
                # Map the detected subcategory name to its parent category name
                parent_category_name = subcategory_to_parent.get(detected_subcategory_name, "unknown")

                # Get the color for the parent category
                color = parent_category_colors.get(parent_category_name, (128, 128, 128)) # Default to grey if not found

                bbox = box.xyxy[0].tolist()
                
                print(f"  • {parent_category_name} (Sub: {detected_subcategory_name}): {confidence:.3f}")
                
                detections_data["detections"].append({
                    "class": parent_category_name, # Store parent category for analysis
                    "confidence": confidence,
                    "bbox": bbox,
                    "original_subcategory": detected_subcategory_name # Keep original for reference
                })
                detections_count += 1

                # --- Draw bounding box and parent category label on the image ---
                x1, y1, x2, y2 = map(int, bbox)
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2

                # Draw the rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                # Put the text label (parent category name) above the bounding box
                cv2.putText(img, parent_category_name, (x1, y1 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)
            
            detections_data["total_detections"] = detections_count
            print(f"Total detected objects: {detections_count}")

        # Save annotated image
        annotated_path = results_dir / f"annotated_{test_image_path.name}"
        cv2.imwrite(str(annotated_path), img) # Save the manually annotated image
        print(f"💾 Saved annotated image: {annotated_path}")
        
        # Save JSON
        json_path = results_dir / "single_detection_results.json"
        with open(json_path, 'w') as f:
            json.dump(detections_data, f, indent=2)
        print(f"💾 Saved detection data: {json_path}")
        
        # Create heatmap if detections exist
        if detections_data["detections"]:
            heatmap_path = results_dir / "confidence_heatmap.png"
            create_detection_heatmap(detections_data, heatmap_path)
            print(f"📊 Saved confidence heatmap: {heatmap_path}")
        
    else:
        print("❌ No objects detected")

def run_batch_processing():
    """Run batch processing on all images in dataset and annotate with parent categories"""
    print("\n📁 Batch Processing All Images")
    print("="*50)
    
    # Paths from configuration
    model_path = CUSTOM_MODEL_PATH
    datasets_dir = Path(IMAGE_DATASET_DIR)
    results_dir = Path(RESULTS_BASE_DIR) / "batch_processing"
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Loaded model: {Path(model_path).name}")
    
    # Find all test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(datasets_dir.rglob(ext)))
    
    if not test_images:
        print(f"❌ No images found in {datasets_dir}")
        return
    
    print(f"📸 Found {len(test_images)} images to process...")
    
    batch_results = []
    detection_summary = {}
    total_detections = 0
    total_time = 0
    
    # Create annotated images directory
    annotated_dir = results_dir / "annotated_images"
    annotated_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(test_images, 1):
        print(f"🔍 ({i}/{len(test_images)}) Processing: {img_path.name}")
        
        start_time = time.time()
        results = model.predict(str(img_path), conf=0.3, verbose=False)
        inference_time = time.time() - start_time
        
        detections_count_in_image = 0
        detections_list_for_image = [] # This will store parent categories for plots
        
        # Load the image for manual annotation
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image {img_path.name}. Skipping annotation for this image.")
            continue

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf.item())
                detected_subcategory_name = model.names[class_id]
                parent_category_name = subcategory_to_parent.get(detected_subcategory_name, "unknown")
                bbox = box.xyxy[0].tolist()

                # Store parent category for analysis and plotting
                detections_list_for_image.append({
                    "class": parent_category_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "original_subcategory": detected_subcategory_name
                })
                detections_count_in_image += 1
                
                # Get color for annotation
                color = parent_category_colors.get(parent_category_name, (128, 128, 128))
                
                # --- Draw bounding box and parent category label on the image ---
                x1, y1, x2, y2 = map(int, bbox)
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2

                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img, parent_category_name, (x1, y1 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)
        
        # Save annotated image
        annotated_path = annotated_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(annotated_path), img)
        
        batch_results.append({
            "image": img_path.name,
            "detections": detections_count_in_image, # Number of detections in this image
            "time": inference_time,
            "detections_list": detections_list_for_image # List of detections (parent categories)
        })
        
        detection_summary[img_path.name] = {
            "detections": detections_list_for_image,
            "total_detections": detections_count_in_image,
            "processing_time": inference_time
        }
        
        total_detections += detections_count_in_image
        total_time += inference_time
        
        print(f"  • {detections_count_in_image} objects detected in {inference_time:.3f}s")
        
    # Calculate statistics
    avg_time = total_time / len(test_images) if len(test_images) > 0 else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    # Save comprehensive results
    detailed_batch_results = {
        "summary": {
            "total_images": len(test_images),
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / len(test_images) if len(test_images) > 0 else 0,
            "avg_processing_time": avg_time,
            "avg_fps": avg_fps,
            "total_processing_time": total_time
        },
        "detailed_results": batch_results,
        "detection_summary": detection_summary
    }
    
    # Save JSON results
    json_path = results_dir / "batch_processing_results.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_batch_results, f, indent=2)
    print(f"\n💾 Saved batch results: {json_path}")
    
    # Create visualizations
    charts_path = results_dir / "batch_analysis_charts.png"
    create_batch_analysis_charts(batch_results, charts_path)
    print(f"📊 Saved batch analysis charts: {charts_path}")
    
    patterns_path = results_dir / "detection_patterns.png"
    create_detection_patterns_visualization(detection_summary, patterns_path)
    print(f"📊 Saved detection patterns: {patterns_path}")
    
    # Print summary
    print(f"\n📊 BATCH PROCESSING SUMMARY:")
    print(f"  Total images: {len(test_images)}")
    print(f"  Total objects detected: {total_detections}")
    print(f"  Average objects per image: {total_detections/len(test_images):.1f}")
    print(f"  Average processing time: {avg_time:.3f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Annotated images saved in: {annotated_dir}")

def main():
    """Run smart office detection evaluation with updated paths and parent category focus"""
    print("🏢 Smart Office Detection Evaluation - Custom Model")
    print("Humblebee AI Hackathon 2025")
    print("=" * 70)
    
    # Create main results directory
    Path(RESULTS_BASE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run tests
    run_single_image_detection()
    run_batch_processing()
    
    print("\n✅ All evaluation tests completed!")
    print("\n📊 Generated outputs:")
    print("  • Annotated images (single + batch) with parent categories and distinct colors")
    print("  • JSON detection results (including parent categories)")
    print("  • Confidence heatmaps (based on parent categories)")
    print("  • Batch analysis charts")
    print("  • Detection pattern visualizations (based on parent categories)")
    print(f"\n📁 All results saved in: {RESULTS_BASE_DIR}")

if __name__ == "__main__":
    main()

