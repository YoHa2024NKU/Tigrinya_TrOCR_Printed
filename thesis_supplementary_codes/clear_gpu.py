import torch
import gc
import sys

print("🧹 Cleaning GPU Cache...")

if not torch.cuda.is_available():
    print("❌ No GPU detected. Nothing to clear.")
    sys.exit()

# 1. Force Python's Garbage Collector to release unreferenced memory
gc.collect()

# 2. Clear PyTorch's internal cache
torch.cuda.empty_cache()

# 3. Reset peak memory stats
torch.cuda.reset_peak_memory_stats()

# 4. Verification
print(f"✅ Cache Cleared.")
print("-" * 30)
print(f"GPU:       {torch.cuda.get_device_name(0)}")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
print("-" * 30)
print("You are ready to train.")