import json
from pathlib import Path
from PIL import Image

class Predictor:
    def __init__(self, checkpoint, mapping=None, device="auto"):
        self.available=False; self.error=None
        try:
            import torch
            from .model import build_model
            from .preprocessing import inference_transforms
            if device=="auto": device="cuda" if torch.cuda.is_available() else "cpu"
            self.device=device; self.checkpoint=Path(checkpoint)
            if not self.checkpoint.exists(): self.error=f"Trained model not found at {self.checkpoint}"; return
            ckpt=torch.load(self.checkpoint,map_location=device,weights_only=False); self.classes=ckpt["classes"]; self.transform=inference_transforms(ckpt.get("image_size",224))
            self.model=build_model(len(self.classes),False).to(device); self.model.load_state_dict(ckpt["state_dict"]); self.model.eval(); self.available=True
        except Exception as exc: self.error=f"Could not load model: {exc}"
    def predict(self, image: Image.Image, top_k=3):
        if not self.available: raise RuntimeError(self.error or "Model unavailable")
        x=self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad(): probs=self.model(x).softmax(1)[0]; scores,idx=probs.topk(min(top_k,len(self.classes)))
        return [{"class_name":self.classes[i],"confidence":float(s)} for s,i in zip(scores.cpu(),idx.cpu())]
