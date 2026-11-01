import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import TrainConfig
from .dataset import load_datasets
from .model import build_model
from .utils import device_for, set_seed

def run(cfg: TrainConfig, device="auto"):
    set_seed(cfg.seed); dev = device_for(device)
    train, val, _, classes = load_datasets(cfg.data_dir, cfg.image_size, cfg.seed)
    tr = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    va = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    model = build_model(len(classes)).to(dev); criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cfg.output_dir.mkdir(parents=True, exist_ok=True); best = 0.0
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(cfg.epochs):
        model.train(); loss_sum=correct=count=0
        for x,y in tr:
            x,y=x.to(dev),y.to(dev); optimizer.zero_grad(set_to_none=True); out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(y); correct += (out.argmax(1)==y).sum().item(); count += len(y)
        history["train_loss"].append(loss_sum/count); history["train_accuracy"].append(correct/count)
        model.eval(); vl=vc=vn=0
        with torch.no_grad():
            for x,y in va:
                x,y=x.to(dev),y.to(dev); out=model(x); vl+=criterion(out,y).item()*len(y); vc+=(out.argmax(1)==y).sum().item(); vn+=len(y)
        vloss, vacc=vl/vn,vc/vn; history["val_loss"].append(vloss); history["val_accuracy"].append(vacc)
        print(f"epoch {epoch+1}/{cfg.epochs}: val_accuracy={vacc:.4f}")
        if vacc > best:
            best=vacc; torch.save({"state_dict":model.state_dict(),"classes":classes,"image_size":cfg.image_size},cfg.output_dir/"best_model.pt")
            (cfg.output_dir/"class_to_idx.json").write_text(json.dumps({c:i for i,c in enumerate(classes)},indent=2))
    (cfg.output_dir/"history.json").write_text(json.dumps(history,indent=2))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="validation"); axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="train"); axes[1].plot(history["val_accuracy"], label="validation"); axes[1].set_title("Accuracy"); axes[1].legend()
    fig.tight_layout(); fig.savefig(cfg.output_dir/"training_curves.png", dpi=150); plt.close(fig)
    return history

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",default="ml/data/PlantVillage"); p.add_argument("--epochs",type=int,default=15); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--device",default="auto"); a=p.parse_args()
    run(TrainConfig(data_dir=Path(a.data),epochs=a.epochs,batch_size=a.batch_size,learning_rate=a.lr),a.device)
