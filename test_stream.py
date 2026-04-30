#!/usr/bin/env python3
import subprocess, time, sys, os, signal, socket, json, urllib.request

def wait_for_port(host, port, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

# Kill existing
subprocess.run(["pkill", "-9", "-f", "uvicorn"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "-f", "python3 main.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

# Start server
proc = subprocess.Popen(
    ["python3", "main.py"],
    cwd="/Users/manideekshith/Desktop/nvidia/backend",
    stdout=open("/tmp/start_stream.log","w"),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setsid
)
print("Server starting...")

# Wait for port 8001
if not wait_for_port("localhost", 8001, 20):
    print("ERROR: Server did not start")
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    sys.exit(1)

print("✓ Server ready on port 8001")

# Test health
try:
    health = urllib.request.urlopen("http://localhost:8001/api/health", timeout=3).read().decode()
    print("Health:", health)
except Exception as e:
    print("Health error:", e)

# Test streaming
print("\nTesting /api/chat/stream...")
try:
    req = urllib.request.Request(
        "http://localhost:8001/api/chat/stream",
        data=bytes('{"message":"hi","session_id":"test"}', "utf-8"),
        headers={"Content-Type":"application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        chunks = []
        for line in raw.split("\n\n"):
            if line.startswith("data: "):
                try:
                    d = json.loads(line[6:])
                    chunks.append(d)
                    if d.get("type") == "done":
                        break
                except:
                    pass
        print(f"Chunks received: {len(chunks)}")
        for c in chunks:
            print(f"  [{c.get('type')}] {str(c.get('content',''))[:100]}")
except Exception as e:
    print("Stream error:", e)

# Cleanup
os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
print("\n✓ Done")
