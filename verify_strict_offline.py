import os
import socket
from datasets import load_from_disk, load_dataset

# 1. Simulate NO INTERNET by disabling sockets
def guard(*args, **kwargs):
    raise Exception("I told you NO INTERNET! Access denied.")
socket.socket = guard

print("🛑 INTERNET HAS BEEN DISABLED (Socket Disabled) 🛑")

# Check environment variable
if os.environ.get("HF_DATASETS_OFFLINE") != "1":
    print("Warning: HF_DATASETS_OFFLINE is not set to 1. But sockets are killed anyway.")

print("\nAttempting to load dataset...")

try:
    # Try loading from cache (standard way lm-eval works offline)
    # We use "load_dataset" here because that's what lm-harness calls internally.
    # It should find it in cache if offline mode works.
    dataset = load_dataset("mkonomi/GreekMMLU-Public", "All")
    
    print("\n✅ SUCCESS! Dataset loaded without internet access.")
    print(f"Dataset has {len(dataset['test'])} test examples.")
    print(f"First question: {dataset['test'][0]['question']}")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    print("This means it tried to connect to the internet and got blocked, or couldn't find the cache.")
    exit(1)
