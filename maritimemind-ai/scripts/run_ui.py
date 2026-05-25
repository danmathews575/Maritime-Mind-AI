#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(root_dir, "app", "ui", "streamlit_app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find Streamlit app at {app_path}")
        sys.exit(1)
        
    print("Starting MaritimeMind AI Streamlit interface...")
    print("Ensure the FastAPI backend is running on http://localhost:8000")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    try:
        subprocess.run(cmd, cwd=root_dir)
    except KeyboardInterrupt:
        print("\nStopping Streamlit UI...")
        sys.exit(0)

if __name__ == "__main__":
    main()
