import subprocess
import time
from server import start_server, wait_for_server
import shutil, os

# 1. Copy DB
SRC = "/kaggle/input/datasets/mkhafid99/database-for-gui"
DST = "/kaggle/working"

for db in os.listdir(SRC):
    dst_path = f"{DST}/{db}"
    if not os.path.exists(dst_path):
        print(f"Copying {db}...")
        shutil.copytree(f"{SRC}/{db}", dst_path)
    else:
        print(f"✅ {db} sudah ada, skip.")

print("✅ Semua DB siap!")

# 2. vLLM server
proc = start_server()
wait_for_server()

# 3. Streamlit
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "app.py",
     "--server.port", "8052",
     "--server.address", "0.0.0.0",
     "--server.headless", "true"],
)

time.sleep(8)
print("🚀 App ready! Buka di browser: http://localhost:8052")