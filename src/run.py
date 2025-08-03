#!run.py
"""
Smart Office Detection Dashboard Runner
Run this script to start the Streamlit dashboard application.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_imports = {
        'streamlit': 'streamlit',
        'ultralytics': 'ultralytics', 
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def find_dashboard_file():
    """Find the dashboard file in current directory"""
    current_dir = Path.cwd()
    
    # Look specifically for dashboard_app.py first, then fallbacks
    dashboard_file = current_dir / 'dashboard_app.py'
    if dashboard_file.exists():
        return str(dashboard_file)
    
    # Fallback to other possible names
    possible_names = [
        'dashboard.py',
        'smart_office_dashboard.py',
        'main_dashboard.py'
    ]
    
    for name in possible_names:
        dashboard_path = current_dir / name
        if dashboard_path.exists():
            return str(dashboard_path)
    
    return None

def run_dashboard(dashboard_file, port=8501, open_browser=True):
    """Run the Streamlit dashboard"""
    try:
        print(f"Starting Smart Office Detection Dashboard...")
        print(f"Running: streamlit run {os.path.basename(dashboard_file)} --server.port {port}")
        print(f"URL: http://localhost:{port}")
        print("-" * 50)
        
        # Change to the directory containing the dashboard file
        dashboard_dir = os.path.dirname(os.path.abspath(dashboard_file))
        dashboard_filename = os.path.basename(dashboard_file)
        
        # Build streamlit command with proper working directory
        cmd = [
            "streamlit", 
            "run", 
            dashboard_filename,
            "--server.port", 
            str(port),
            "--server.headless", 
            "false",
            "--server.address",
            "localhost"
        ]
        
        print(f"Working directory: {dashboard_dir}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 50)
        
        # Start streamlit process with correct working directory
        process = subprocess.Popen(
            cmd, 
            cwd=dashboard_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor output and wait for "Local URL" or "Network URL"
        streamlit_ready = False
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            if "Local URL:" in line or "You can now view your Streamlit app" in line:
                streamlit_ready = True
                if open_browser and not streamlit_ready:
                    time.sleep(1)
                    print("Opening browser...")
                    webbrowser.open(f"http://localhost:{port}")
        
        # If process ends, wait for it
        process.wait()
            
    except FileNotFoundError:
        print("Error: 'streamlit' command not found.")
        print("Try installing streamlit in your virtual environment:")
        print("pip install streamlit")
        print("\nOr try running directly:")
        print(f"streamlit run {os.path.basename(dashboard_file)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        if 'process' in locals():
            process.terminate()
            process.wait()
    except Exception as e:
        print(f"Error running dashboard: {e}")
        print(f"\nTry running directly: streamlit run {os.path.basename(dashboard_file)}")
        sys.exit(1)

def main():
    """Main function"""
    print("=" * 60)
    print("Smart Office Detection Dashboard Runner")
    print("=" * 60)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    print("✓ All dependencies found")
    
    # Find dashboard file
    print("Looking for dashboard file...")
    dashboard_file = find_dashboard_file()
    
    if not dashboard_file:
        print("Error: dashboard_app.py not found!")
        print("Please ensure 'dashboard_app.py' exists in the current directory.")
        sys.exit(1)
    
    print(f"✓ Found dashboard file: {os.path.basename(dashboard_file)}")
    
    # Get port from command line args or use default
    port = 8501
    open_browser = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\nUsage:")
            print("  python run.py [port] [--no-browser]")
            print("\nOptions:")
            print("  port         Port number (default: 8501)")
            print("  --no-browser Don't open browser automatically")
            print("\nExamples:")
            print("  python run.py")
            print("  python run.py 8502")
            print("  python run.py 8501 --no-browser")
            return
        
        try:
            port = int(sys.argv[1])
        except ValueError:
            if sys.argv[1] == "--no-browser":
                open_browser = False
            else:
                print(f"Invalid port number: {sys.argv[1]}")
                sys.exit(1)
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "--no-browser":
            open_browser = False
    
    # Run dashboard
    print(f"✓ Starting dashboard on port {port}")
    if not open_browser:
        print("✓ Browser auto-open disabled")
    
    print("\nPress Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    run_dashboard(dashboard_file, port, open_browser)

if __name__ == "__main__":
    main()
