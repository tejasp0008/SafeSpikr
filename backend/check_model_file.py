# backend/check_model_file.py
import torch, os
p = r"C:\Users\TEJAS PAL\Projects\SafeSpikr\outputs\unified_run1\snn_model_best.pth"
print("Looking for:", p)
obj = torch.load(p, map_location='cpu')
print("TYPE:", type(obj))
if isinstance(obj, dict):
    print("State-dict keys (sample 60):")
    for k in list(obj.keys())[:60]:
        print(" -", k)
else:
    print("Object repr (short):", repr(obj)[:1000])
