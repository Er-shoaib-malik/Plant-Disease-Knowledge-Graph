import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .dataset import load_datasets
from .model import build_model
from .utils import device_for

def evaluate(checkpoint, data, output="ml/artifacts", device="auto"):
    dev=device_for(device); _,_,test,classes=load_datasets(data)
    ckpt=torch.load(checkpoint,map_location=dev,weights_only=False); model=build_model(len(ckpt["classes"]),False).to(dev); model.load_state_dict(ckpt["state_dict"]); model.eval()
    y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in DataLoader(test,batch_size=32): y_pred.extend(model(x.to(dev)).argmax(1).cpu().tolist()); y_true.extend(y.tolist())
    metrics={"accuracy":accuracy_score(y_true,y_pred),"report":classification_report(y_true,y_pred,target_names=classes,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(y_true,y_pred).tolist()}
    Path(output).mkdir(parents=True,exist_ok=True); Path(output,"evaluation.json").write_text(json.dumps(metrics,indent=2)); return metrics

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",default="ml/data/PlantVillage");p.add_argument("--checkpoint",default="ml/artifacts/best_model.pt");p.add_argument("--device",default="auto");a=p.parse_args();print(json.dumps(evaluate(a.checkpoint,a.data,device=a.device)["accuracy"]))
