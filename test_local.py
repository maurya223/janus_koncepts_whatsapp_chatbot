import subprocess
import sys

def check_package(pkg):
    try:
        __import__(pkg)
        print(f"[✓] {pkg} is installed")
    except ImportError:
        print(f"[✗] {pkg} is NOT installed")

# 1. Check core packages
packages = ["flask", "requests", "dotenv", "faiss", "langchain", "gunicorn"]
for pkg in packages:
    check_package(pkg)

# 2. Run Flask app in test mode
print("\n[→] Starting Flask test server...")
try:
    subprocess.run([sys.executable, "app.py"], check=True)
except KeyboardInterrupt:
    print("\n[!] Flask test server stopped.")
