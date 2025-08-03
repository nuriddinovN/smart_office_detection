#prompt_rune.py
import os
import sys
from ultralytics import YOLO
import cv2 # Used for image manipulation and display
import numpy as np # Used for array operations with images

# --- Configuration Paths ---
# Define the base directory for your project. Adjust if your structure is different.
# This script assumes it's run from within the 'src' directory or similar,
# and that 'categories.py' is accessible.
PROJECT_ROOT = "/home/noor/A/projects/smart_office"
CATEGORIES_FILE_PATH = os.path.join(os.path.dirname(__file__), 'categories.py') # Assumes categories.py is in the same directory

# Define the path for the custom model that we will create and save
CUSTOM_MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models")
CUSTOM_MODEL_NAME = "smart_office_prompttuned.pt"
CUSTOM_MODEL_PATH = os.path.join(CUSTOM_MODEL_DIR, CUSTOM_MODEL_NAME)

# Define the directory containing images for inference
IMAGE_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")

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
# Prepare a flat list of all sub-categories to be detectable by YOLO-World.
# The model will be trained/tuned to recognize these specific subcategories.
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


# --- Part 1: Save a custom model with the defined vocabulary ---
print("\n--- Part 1: Saving a custom model with your vocabulary ---")
try:
    # Ensure the directory for saving the model exists
    os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)

    # Initialize a base YOLO-World model.
    # Using yolov8x-worldv2.pt as specified.
    print("Loading base model: yolov8x-worldv2.pt")
    model = YOLO("models/yolov8x-worldv2.pt")

    # Set the custom classes from the categories_data dictionary.
    # We set ALL subcategories here so the model can distinguish them internally.
    print(f"Setting {len(detectable_classes)} custom classes (including subcategories) for the model...")
    model.set_classes(detectable_classes)

    # Save the model with the defined custom vocabulary.
    # This model will now be specialized for detecting your specified subcategories.
    print(f"Saving the custom model to '{CUSTOM_MODEL_PATH}'")
    model.save(CUSTOM_MODEL_PATH)
    print("Model saved successfully. This model is now specialized for your classes.")

except Exception as e:
    print(f"An error occurred during model saving: {e}")
    sys.exit(1)

# --- Part 2: Load the custom model and run inference ---
print("\n--- Part 2: Loading the custom model and running inference ---")
try:
    # Load your newly created custom model.
    print(f"Loading the custom model from '{CUSTOM_MODEL_PATH}'")
    custom_model = YOLO(CUSTOM_MODEL_PATH)
    print("Custom model loaded successfully.")

    # Check if the image dataset directory exists
    if not os.path.isdir(IMAGE_DATASET_DIR):
        print(f"Error: Image dataset directory not found at '{IMAGE_DATASET_DIR}'. Please check the path.")
        sys.exit(1)

    # Get all image files from the specified dataset directory
    image_files = [f for f in os.listdir(IMAGE_DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in '{IMAGE_DATASET_DIR}'. Skipping inference step.")
    else:
        print(f"Found {len(image_files)} images in '{IMAGE_DATASET_DIR}'.")
        for image_file in image_files:
            image_path = os.path.join(IMAGE_DATASET_DIR, image_file)
            print(f"\n--- Running inference on image: {image_path} ---")

            # Run inference to detect all subcategories.
            # verbose=False suppresses detailed output from the predict function.
            results = custom_model.predict(image_path, conf=0.4, verbose=False) # Using a confidence threshold of 0.4

            # Process results for the current image
            if results and results[0].boxes:
                print(f"--- Detected Objects from Custom Model in {image_file} ---")
                
                # Load the image using OpenCV for drawing annotations.
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not load image {image_file}. Skipping annotation for this image.")
                    continue

                # Iterate through each detected bounding box
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    confidence = box.conf.item()
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    xyxy = box.xyxy[0].cpu().numpy().astype(int) 

                    # Get the class name (which will be a subcategory) from the model's names attribute
                    detected_subcategory_name = custom_model.names[class_id]
                    
                    # Map the detected subcategory name to its parent category name
                    parent_category_name = subcategory_to_parent.get(detected_subcategory_name, "unknown")

                    # Get the color for the parent category
                    color = parent_category_colors.get(parent_category_name, (128, 128, 128)) # Default to grey if not found

                    # Print the parent category and confidence to the terminal
                    print(f"  Object: '{parent_category_name}' (Subcategory: '{detected_subcategory_name}', Confidence: {confidence:.2f})")

                    # --- Draw bounding box and parent category label on the image ---
                    x1, y1, x2, y2 = xyxy
                    thickness = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2

                    # Draw the rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    # Put the text label (parent category name) above the bounding box
                    cv2.putText(img, parent_category_name, (x1, y1 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)
                
                # Display the annotated image.
                # The window will remain open until a key is pressed.
                print(f"Showing annotated image for {image_file}. Close the window to continue to the next image.")
                cv2.imshow(f"Annotated Image: {image_file}", img)
                cv2.waitKey(0) # Wait indefinitely until a key is pressed
                cv2.destroyAllWindows() # Close all OpenCV windows after a key is pressed

            else:
                print(f"No objects detected in {image_file} for the custom classes.")

except Exception as e:
    print(f"An error occurred during inference: {e}")
    sys.exit(1)


