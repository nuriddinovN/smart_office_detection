# 🏢 Smart Office Detection Dashboard

An AI-powered object detection system specifically designed for office environments using YOLO-World model. This project provides real-time detection of office objects including people, chairs, monitors, keyboards, laptops, and phones through an intuitive Streamlit web interface.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-World-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Custom YOLO-World Model**: Fine-tuned for office object detection
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
git clone https://github.com/yourusername/smart-office-detection.git
cd smart-office-detection
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

### Step 4: Download Pre-trained Models

```bash
# Create models directory
mkdir -p src/models

# Download base YOLO-World models (these will be downloaded automatically on first run)
# The custom model will be created during the first run
```

## 🎮 Usage Guide

### Method 1: Quick Start with Dashboard Launcher

```bash
# Navigate to the project directory
cd smart-office-detection

# Activate virtual environment
source smart_office/bin/activate

# Run the dashboard launcher
python src/run.py
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
python src/run.py 8502

# Run without auto-opening browser
python src/run.py --no-browser
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

#### 1. ModuleNotFoundError
```bash
# Make sure virtual environment is activated
source smart_office/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. CUDA/GPU Issues
```bash
# Install CPU-only PyTorch if you don't have GPU
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Port Already in Use
```bash
# Use different port
python src/run.py 8502
```

#### 4. Model Download Issues
```bash
# Manually download base models
python -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"
```

#### 5. Permission Errors
```bash
# Fix file permissions
chmod +x src/run.py
```

### Getting Help

If you encounter issues:

1. **Check Python Version**: Ensure you're using Python 3.10+
2. **Virtual Environment**: Always activate your virtual environment
3. **Dependencies**: Verify all packages are installed correctly
4. **GPU Memory**: If using GPU, ensure sufficient VRAM (4GB+)
5. **File Paths**: Check all file paths are correct

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Average Processing Time | ~1-3 seconds per image |
| Supported Image Formats | JPG, PNG, JPEG, BMP |
| Maximum Image Size | 4K (4096x4096) |
| Confidence Threshold Range | 0.1 - 1.0 |
| Supported Categories | 6 main categories, 25+ subcategories |

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
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/smart-office-detection/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/smart-office-detection/discussions)

---

<div align="center">
  <p>Made with ❤️ for Humblebee AI Hackathon 2025</p>
  <p>⭐ Star this repo if you found it helpful!</p>
</div>
