import argparse
from PIL import Image
from .inference import Predictor

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("image");p.add_argument("--checkpoint",default="ml/artifacts/best_model.pt");a=p.parse_args();print(Predictor(a.checkpoint).predict(Image.open(a.image)))
