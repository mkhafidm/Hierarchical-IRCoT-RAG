import subprocess
import time
import urllib.request
import threading
import requests

MODEL_PATH = "/kaggle/input/models/mkhafid99/qwen2-5-7b-instruct-awq/transformers/default/1"

# ── Server ──────────────────────────────────────────────
def start_server():
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
    time.sleep(3)
    proc = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--quantization", "awq",
        "--dtype", "float16",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.75",
        "--max-model-len", "6144",
        "--port", "8000",
        "--host", "0.0.0.0",
        "--trust-remote-code",
    ], stdout=open("/kaggle/working/vllm.log", "w"), stderr=subprocess.STDOUT)
    return proc

def stream_logs(proc):
    with open("/kaggle/working/vllm.log", "r") as f:
        while proc.poll() is None:
            line = f.readline()
            if line:
                print(f"[vllm] {line}", end="")
            else:
                time.sleep(0.5)

def wait_for_server(port=8000, max_wait=600, interval=5):
    elapsed = 0
    while elapsed < max_wait:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            print(f"✅ Server ready dalam {elapsed} detik")
            return True
        except:
            print(f"⏳ {elapsed}s / {max_wait}s", end="\r", flush=True)
            time.sleep(interval)
            elapsed += interval
    print(f"\n❌ Timeout setelah {max_wait}s")
    return False

# ── Monitoring ───────────────────────────────────────────
def check_gpu():
    result = subprocess.run([
        "nvidia-smi", 
        "--query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu",
        "--format=csv,noheader"
    ], capture_output=True, text=True)
    print(f"[GPU] {result.stdout.strip()}")

def monitor_vllm(interval=10):
    """Jalanin di background thread, print metrics tiap `interval` detik"""
    keys = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
    ]
    while True:
        try:
            r = requests.get("http://localhost:8000/metrics", timeout=5)
            lines = [l.strip() for l in r.text.split("\n") 
                     if any(k in l for k in keys) and not l.startswith("#")]
            if lines:
                print(f"\n[MONITOR {time.strftime('%H:%M:%S')}]")
                for l in lines: print(f"  {l}")
            check_gpu()
        except:
            pass
        time.sleep(interval)
