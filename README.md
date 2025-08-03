# 🏢 Smart Office Detection Dashboard

An AI-powered object detection system specifically designed for office environments using YOLO-World model. This project provides real-time detection of office objects including people, chairs, monitors, keyboards, laptops, and phones through an intuitive Streamlit web interface.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-World-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Custom YOLO-World Model**: Prompt-tuned for office object detection
- **Interactive Web Dashboard**: Easy-to-use Streamlit interface
- **Real-time Detection**: Upload images and get instant results
- **Multiple Export Formats**: JSON, CSV, and annotated images
- **Comprehensive Analytics**: Detection confidence analysis and visualization
- **Batch Processing**: Process multiple images at once
- **Performance Metrics**: Processing time and detection statistics

## 🎯 Detected Object Categories

| Category | Subcategories |
|----------|---------------|
| **Person** | person, people, human, worker, employee |
| **Chair** | chair, office chair, seat, desk chair, swivel chair |
| **Monitor** | computer screen, display, LCD, LED, OLED, gaming monitor, 4K monitor |
| **Keyboard** | keyboard, typing keyboard, keypad, computer keyboard |
| **Laptop** | laptop, notebook, portable computer, notebook computer |
| **Phone** | smartphone, mobile phone, cell phone, telephone |

## 🛠️ Project Structure

```
smart_office/
├── src/
│   ├── dashboard_app.py      # Main Streamlit dashboard
│   ├── evaluate.py           # Model evaluation and batch processing
│   ├── prompt_tune.py        # Model fine-tuning script
│   ├── run.py               # Dashboard launcher
│   ├── categories.py        # Object categories configuration
│   └── models/
│       ├── smart_office_prompttuned.pt    # Custom trained model
│       ├── yolov8x-world.pt              # Base YOLO-World model
│       └── yolov8x-worldv2.pt            # Updated YOLO-World model
├── datasets/                # Training/testing images
├── results/                 # Output directory for results
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository)
- At least 4GB RAM (8GB recommended)
- NVIDIA GPU (optional, for faster processing)

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/nuriddinovN/smart_office_detection.git
cd smart_office_detection
```

### Step 2: Set Up Virtual Environment

```bash
# Create a virtual environment (recommended)
python3.10 -m venv smart_office

# Activate the virtual environment
# On Linux/macOS:
source smart_office/bin/activate

# On Windows:
# smart_office\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.10.x
```

### Step 3: Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify ultralytics installation
python -c "from ultralytics import YOLO; print('✅ Ultralytics installed successfully')"
```

### Step 4: **✅Verify Model Installation:**

```bash
# Check if models are properly downloaded
python -c "
import os
models_dir = 'src/models'
required_models = ['smart_office_prompttuned.pt', 'yolov8x-worldv2.pt']
for model in required_models:
    path = os.path.join(models_dir, model)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)  # MB
        print(f'✅ {model}: {size:.1f} MB')
    else:
        print(f'❌ {model}: NOT FOUND')
"
```


## ⚠️ **CRITICAL WARNING: Model Files Required**

**🚨 IMPORTANT: Due to GitHub file size limitations (>100MB), the trained model files MIGHT NOT be included in the repository clone. The cloned repository MIGHT contains only placeholder files that MUST be replaced with actual model files.**

**❌ The project WILL NOT WORK without downloading the actual model files!**

**⚠️If you face such truble pls follow instructions, otherwise just skip this step**

### Why This Step is Critical:
- Model files are 140MB+ each (exceeds GitHub's file size limit)
- Placeholder files in the clone are empty and non-functional
- You must manually download the real model files from GitHub's raw file links

### Required Models (MUST DOWNLOAD):

1. **smart_office_prompttuned.pt** - Custom trained model (Required for office detection)
2. **yolov8x-worldv2.pt** - Base YOLO-World model (Required for inference)

### Download Instructions (MANDATORY):

```bash
# STEP 1: Clean the models directory first (remove placeholder files)
cd src/models
rm smart_office_prompttuned.pt yolov8x-world.pt yolov8x-worldv2.pt

# STEP 2: Download the actual model files from GitHub raw links
# Method 1: Using wget (Linux/macOS)
wget https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/smart_office_prompttuned.pt
wget https://github.com/nuriddianovN/smart_office_detection/raw/main/src/models/yolov8x-worldv2.pt

# Method 2: Using curl (if wget not available)
curl -L -o smart_office_prompttuned.pt https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/smart_office_prompttuned.pt
curl -L -o yolov8x-worldv2.pt https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/yolov8x-worldv2.pt

# STEP 3: Verify downloads (files should be 100MB+ each)
ls -lh *.pt
cd ../..
```

### Manual Download (Windows or if command-line fails):

**🔗 Direct Download Links:**

1. **Custom Model:** [smart_office_prompttuned.pt](https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/smart_office_prompttuned.pt)
2. **Base Model:** [yolov8x-worldv2.pt](https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/yolov8x-worldv2.pt)

**Steps:**
1. Right-click each link above → "Save link as..."
2. Save both files to your `src/models/` directory
3. Ensure file names are exactly: `smart_office_prompttuned.pt` and `yolov8x-worldv2.pt`

**⚠️ DO NOT rename the files - they must have exact names as shown above**


## 🎮 Usage Guide

### Method 1: Quick Start with Dashboard Launcher

```bash
# Navigate to the project directory
cd smart_office_detection
cd src

# make sure Activate virtual environment

# Run the dashboard launcher
python run.py
```

The launcher will:
- ✅ Check all dependencies
- 🔍 Find the dashboard file
- 🚀 Start Streamlit server
- 🌐 Open your browser automatically

### Method 2: Direct Streamlit Launch

```bash
# Navigate to src directory
cd src

# Run Streamlit directly
streamlit run dashboard_app.py
```

### Method 3: Custom Port

```bash
# Run on specific port
python run.py 8502

# Run without auto-opening browser
python run.py --no-browser
```

## 📱 Using the Dashboard

### 1. Access the Dashboard
- Open your browser and go to `http://localhost:8501`
- You'll see the Smart Office Detection Dashboard

### 2. Upload an Image
- Click "Choose an image..." button
- Select a `.jpg`, `.png`, or `.jpeg` file
- The original image will be displayed

### 3. Configure Settings (Optional)
- Open sidebar for settings
- Adjust **Confidence Threshold** (0.1 - 1.0)
- Toggle **Show Subcategories** for detailed view
- Enable **Show JSON Data** for technical details

### 4. Run Detection
- Click the **"Run Detection"** button
- Wait for processing (usually 1-3 seconds)
- View results with bounding boxes and labels

### 5. Analyze Results
- Check **Performance Metrics**: processing time, objects found, categories
- View **Detection Details** table
- Examine **Detection Charts** for visual analysis

### 6. Export Results
- **Download Annotated Image**: Image with bounding boxes
- **Download CSV Statistics**: Detection data in spreadsheet format
- **Download JSON Boxes**: Detailed bounding box coordinates

## 🎬 Live Demo & Examples

### 🖼️ Sample Detection Results

**Input Image:** Office environment with multiple workstations
```
📸 Processing office_scene.jpg (1920x1080)
⏱️ Processing time: 2.3 seconds
```

**Detection Output:**
```
🔍 DETECTED OBJECTS (Total: 12)

👥 PEOPLE (2 detected)
├── Person 1: Confidence 92% | Position: (245, 156, 398, 567)
└── Person 2: Confidence 87% | Position: (892, 203, 1045, 623)

🪑 CHAIRS (4 detected)  
├── Office Chair 1: Confidence 94% | Position: (123, 345, 267, 678)
├── Office Chair 2: Confidence 89% | Position: (445, 298, 589, 634)
├── Desk Chair 3: Confidence 91% | Position: (756, 367, 889, 701)
└── Swivel Chair 4: Confidence 88% | Position: (1124, 334, 1245, 656)

🖥️ MONITORS (3 detected)
├── Computer Screen 1: Confidence 91% | Position: (334, 123, 678, 445)
├── LCD Monitor 2: Confidence 85% | Position: (789, 134, 1123, 456)
└── Gaming Monitor 3: Confidence 90% | Position: (1234, 145, 1567, 467)

⌨️ KEYBOARDS (2 detected)
├── Keyboard 1: Confidence 88% | Position: (356, 567, 543, 623)
└── Computer Keyboard 2: Confidence 90% | Position: (823, 578, 1012, 634)

💻 LAPTOPS (1 detected)
└── Laptop 1: Confidence 93% | Position: (456, 234, 678, 445)

📱 PHONES (1 detected)
└── Smartphone 1: Confidence 86% | Position: (567, 345, 612, 456)
```

### 📊 Dashboard Preview

When you run the dashboard, you'll see:

**1. Upload Section**
```
┌─────────────────────────────────────────┐
│  📁 Choose an image file...              │
│  Drag and drop or click to browse       │
│  Supported: JPG, PNG, JPEG, BMP         │
└─────────────────────────────────────────┘
```

**2. Settings Sidebar**
```
⚙️ DETECTION SETTINGS
├── 🎯 Confidence Threshold: 0.5 ████████░░
├── 📋 Show Subcategories: ☑️ Yes
├── 📄 Show JSON Data: ☐ No  
└── 🎨 Box Thickness: Medium
```

**3. Results Display**
```
✅ DETECTION COMPLETE!

📊 PERFORMANCE METRICS
├── ⏱️ Processing Time: 2.3s
├── 🔍 Objects Found: 12
├── 📂 Categories: 6
└── 🎯 Avg Confidence: 89.2%

📈 DETECTION DISTRIBUTION
People     ████████░░ 17% (2)
Chairs     ████████████ 33% (4)  
Monitors   ████████░░░░ 25% (3)
Keyboards  ████░░░░░░░░ 17% (2)
Laptops    ██░░░░░░░░░░ 8% (1)
Phones     ██░░░░░░░░░░ 8% (1)
```

**4. Export Options**
```
💾 DOWNLOAD RESULTS
├── 🖼️ [Download Annotated Image] (office_scene_annotated.jpg)
├── 📊 [Download CSV Report] (detection_results.csv)  
└── 📄 [Download JSON Data] (bounding_boxes.json)
```

### 🎥 Interactive Features Demo

**Real-time Confidence Adjustment:**
- Move slider: `0.1 ←────●────→ 1.0`
- Live update: Objects appear/disappear based on threshold
- Visual feedback: Box colors change with confidence levels

**Hover Effects:**
- Hover over detected objects → Show detailed info popup
- Click on category labels → Highlight all objects of that type
- Zoom functionality → Click and drag to zoom into specific areas

### 📱 Mobile-Responsive Design

The dashboard works on all devices:
```
💻 Desktop: Full feature set with sidebar
📱 Mobile: Collapsible menu, touch-friendly controls  
📟 Tablet: Optimized layout for medium screens
```

## 🔧 Advanced Usage

### Model Evaluation

```bash
# Run comprehensive model evaluation
cd src
python evaluate.py
```

This will:
- Test detection on sample images
- Generate confidence heatmaps
- Create batch processing analysis
- Save results in `results/` directory

### Model Fine-tuning

```bash
# Fine-tune the model with your custom categories
cd src
python prompt_tune.py
```

This will:
- Create a custom model with your vocabulary
- Test inference on your dataset
- Display annotated results

### Adding Custom Images

```bash
# Add your images to the datasets folder
cp /path/to/your/images/* datasets/

# Run batch evaluation
python src/evaluate.py
```

## 🛡️ Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Found Error (MOST COMMON)
```bash
# ❌ Error: FileNotFoundError: [Errno 2] No such file or directory: 'src/models/smart_office_prompttuned.pt'
# ❌ Error: Model file not found or corrupted

# ✅ SOLUTION: Download the actual model files (not placeholders)
cd src/models

# Remove placeholder files first
rm -f smart_office_prompttuned.pt yolov8x-world.pt yolov8x-worldv2.pt

# Download real model files
wget https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/smart_office_prompttuned.pt
wget https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/yolov8x-worldv2.pt

# Verify file sizes (should be 100MB+ each)
ls -lh *.pt
```

#### 2. ModuleNotFoundError
```bash
# Make sure virtual environment is activated
source smart_office/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

#### 3. CUDA/GPU Issues
```bash
# Install CPU-only PyTorch if you don't have GPU
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Port Already in Use
```bash
# Use different port
python src/run.py 8502
```

#### 5. Model Download Issues
```bash
# ❌ Error: Connection failed, file corrupted, or wrong file size

# ✅ SOLUTION 1: Use alternative download method
curl -L -o src/models/smart_office_prompttuned.pt https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/smart_office_prompttuned.pt
curl -L -o src/models/yolov8x-worldv2.pt https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/yolov8x-worldv2.pt

# ✅ SOLUTION 2: Manual download from browser
# Visit: https://github.com/nuriddinovN/smart_office_detection/raw/main/src/models/smart_office_prompttuned.pt
# Right-click → Save As → src/models/smart_office_prompttuned.pt

# ✅ SOLUTION 3: Verify file integrity
cd src/models
file *.pt  # Should show "PyTorch model" or similar
ls -lh *.pt  # Should show file sizes 100MB+
```

#### 6. Permission Errors
```bash
# Fix file permissions
chmod +x src/run.py
```

### Getting Help

If you encounter issues:

1. **Check Python Version**: Ensure you're using Python 3.10+
2. **Virtual Environment**: Always activate your virtual environment
3. **Model Files**: Verify both required models are downloaded
4. **Dependencies**: Verify all packages are installed correctly
5. **GPU Memory**: If using GPU, ensure sufficient VRAM (4GB+)
6. **File Paths**: Check all file paths are correct

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Average Processing Time | ~1-3 seconds per image |
| Supported Image Formats | JPG, PNG, JPEG, BMP |
| Maximum Image Size | 4K (4096x4096) |
| Confidence Threshold Range | 0.1 - 1.0 |
| Supported Categories | 6 main categories, 25+ subcategories |
| Model Size | ~140MB (smart_office_prompttuned.pt) |
| Base Model Size | ~280MB (yolov8x-worldv2.pt) |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO-World model
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Plotly](https://plotly.com/) for interactive visualizations

## 📞 Support

If you have questions or need help:

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/nuriddinovN/smart_office_detection/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/nuriddinovN/smart_office_detection/discussions)

---

<div align="center">
  <p>Made with ❤️ for Humblebee AI Hackathon 2025</p>
  <p>⭐ Star this repo if you found it helpful!</p>
</div>
