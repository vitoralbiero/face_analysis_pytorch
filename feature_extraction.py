import argparse
from os import makedirs, path

import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

from data.data_loader_test import TestDataLoader
from model.resnet import ResNet
from utils.model_loader import load_state


class Extractor:
    def __init__(
        self,
        model_path,
        source,
        image_list,
        dest,
        net_mode,
        depth,
        batch_size,
        workers,
        drop_ratio,
        device,
    ):

        self.loader = TestDataLoader(batch_size, workers, source, image_list)
        self.batch_size = batch_size
        self.image_paths = np.asarray(self.loader.dataset.samples)
        self.model = None
        self.device = device
        self.destination = dest

        self.model = self.create_model(depth, drop_ratio, net_mode, model_path)
        self.model.eval()

    def create_model(self, depth, drop_ratio, net_mode, model_path):
        model = DataParallel(ResNet(depth, drop_ratio, net_mode)).to(self.device)
        load_state(
            model=model, path_to_model=model_path, model_only=True, load_head=False
        )

        model.eval()

        return model

    def run(self):
        idx = 0
        with torch.no_grad():
            for imgs in tqdm(iter(self.loader)):
                imgs = imgs.to(device)

                embeddings = self.model(imgs)
                embeddings = embeddings.cpu().numpy()

                image_paths = self.image_paths[idx : idx + self.batch_size]
                self.save_features(image_paths, embeddings)
                idx += self.batch_size

    def save_features(self, image_paths, embeddings):
        for i in range(0, len(embeddings)):
            image_name = path.split(image_paths[i])[1]
            sub_folder = path.basename(path.normpath(path.split(image_paths[i])[0]))
            dest_path = path.join(self.destination, sub_folder)

            if not path.exists(dest_path):
                makedirs(dest_path)

            features_name = path.join(dest_path, image_name[:-3] + "npy")
            np.save(features_name, embeddings[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features.")
    parser.add_argument("--source", "-s", help="Path to the images.")
    parser.add_argument("--image_list", "-i", help="File with images names.")
    parser.add_argument("--dest", "-d", help="Path to save the predictions.")
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=250, type=int)
    parser.add_argument("--model", "-m", help="Path to model.")
    parser.add_argument(
        "--net_mode", "-n", help="Residual type [ir, ir_se].", default="ir_se", type=str
    )
    parser.add_argument(
        "--depth", "-dp", help="Number of layers [50, 100, 152].", default=50, type=int
    )
    parser.add_argument("--workers", "-w", help="Workers number.", default=4, type=int)

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    drop_ratio = 0.4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extractor = Extractor(
        args.model,
        args.source,
        args.image_list,
        args.dest,
        args.net_mode,
        args.depth,
        args.batch_size,
        args.workers,
        drop_ratio,
        device,
    )
    extractor.run()
