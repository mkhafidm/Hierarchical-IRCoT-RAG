import subprocess
import time
from server import start_server, wait_for_server
import shutil, os


SRC_BASE = "/kaggle/input/datasets/mkhafid99/database-for-gui"
DST = "/kaggle/working"

datasets = ["narrativeqa_qdrant_db_new", "qasper_qdrant_db_new", "quality_qdrant_db_new", "tydiqa_qdrant_db_new"]

for db in datasets:
    src_path = f"{SRC_BASE}/{db}/kaggle/working/{db}"
    dst_path = f"{DST}/{db}"
    if not os.path.exists(dst_path):
        print(f"Copying {db}...")
        shutil.copytree(src_path, dst_path)
        print(f"✅ {db} done.")
    else:
        print(f"✅ {db} sudah ada, skip.")


print("✅ Semua DB siap!")

proc = start_server()
wait_for_server()

streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "app.py",
     "--server.port", "8052",
     "--server.address", "0.0.0.0",
     "--server.headless", "true"],
)