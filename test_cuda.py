import torch, os
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

i = 1
props = torch.cuda.get_device_properties(i)
print(f"[logical {i}] name={props.name}, total_memory={props.total_memory/1e9:.1f} GB, uuid={props.uuid}")