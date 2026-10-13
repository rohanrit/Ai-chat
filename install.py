import subprocess
import sys

def main():
    print("Starting Local AI App Setup...")
    
    # Core packages that install without C++ compiler
    core_packages = [
        "customtkinter>=5.2.2",
        "huggingface_hub>=0.23.0",
        "python-dotenv>=1.0.1",
        "requests>=2.31.0",
        "tqdm>=4.42.1",
        "pypdf>=4.0.0",
        "python-docx>=1.1.0",
        "Pillow>=10.0.0",
        "rapidocr-onnxruntime>=1.4.0",
        "onnxruntime>=1.16.0",
        "PyMuPDF>=1.24.0"
    ]
    
    print("Installing core GUI and Hub dependencies...")
    for pkg in core_packages:
        print(f"Installing {pkg}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except Exception as e:
            print(f"Error installing {pkg}: {e}")
        
    print("\nInstalling prebuilt inference engine (llama-cpp-python)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "llama-cpp-python",
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"
        ])
        print("llama-cpp-python installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n[NOTE]: Could not install prebuilt llama-cpp-python wheel: {e}")

    print("\nSetup complete! You can now run: python main.py")

if __name__ == "__main__":
    main()
