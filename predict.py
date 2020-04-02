import numpy as np
import argparse
from os import makedirs, path
from model.race_head import RaceHead
from model.gender_head import GenderHead
from model.age_head import AgeHead
from model.resnet import ResNet
from utils.model_loader import load_state
from data.data_loader_predict import PredictionDataLoader
import torch
from tqdm import tqdm
from torch.nn import DataParallel


def create_model(depth, drop_ratio, net_mode, model_path, head, device):
    model = DataParallel(ResNet(depth, drop_ratio, net_mode)).to(device)
    head = DataParallel(head()).to(device)

    load_state(model, head, None, model_path, True)

    model.eval()
    head.eval()

    return model, head


def predict(loader, device, model, head, attribute):
    model.eval()
    head.eval()

    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for imgs in tqdm(iter(loader)):
            imgs = imgs.to(device)

            embeddings = model(imgs)
            outputs = head(embeddings)

            all_outputs = torch.cat((all_outputs, outputs), 0)

    if attribute == 'age':
        y_pred = all_outputs.cpu().numpy()
        y_pred = np.round(y_pred, 0)
        y_pred = np.sum(y_pred, axis=1)
    else:
        _, y_pred = torch.max(all_outputs, 1)
        y_pred = y_pred.cpu().numpy()

    return y_pred


def predict_attributes(race_model_path, gender_model_path, age_model_path, source, image_list, dest,
                       net_mode, depth, batch_size, workers, drop_ratio, device):

    predict_loader = PredictionDataLoader(batch_size, workers, source, image_list)

    predictions = np.asarray(predict_loader.dataset.samples)

    if race_model_path:
        race_model, race_head = create_model(depth, drop_ratio, net_mode, race_model_path, RaceHead, device)

        race_preds = predict(predict_loader, device, race_model, race_head, 'race')
        predictions = np.column_stack((predictions, race_preds))

    if gender_model_path:
        gender_model, gender_head = create_model(depth, drop_ratio, net_mode,
                                                 gender_model_path, GenderHead, device)

        gender_preds = predict(predict_loader, device, gender_model, gender_head, 'gender')
        predictions = np.column_stack((predictions, gender_preds))

    if age_model_path:
        age_model, age_head = create_model(depth, drop_ratio, net_mode, age_model_path, AgeHead, device)

        age_preds = predict(predict_loader, device, age_model, age_head, 'age')
        predictions = np.column_stack((predictions, age_preds))

    np.savetxt(path.join(dest, path.split(image_list)[1][:-4] + '_predictions.txt'),
               predictions, delimiter=' ', fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--source', '-s', help='Path to the images.')
    parser.add_argument('--image_list', '-i', help='File with images names.')
    parser.add_argument('--dest', '-d', help='Path to save the predictions.')
    parser.add_argument('--batch_size', '-b', help='Batch size.', default=96, type=int)
    parser.add_argument('--model', help='Path to model.',)
    parser.add_argument('--race_model', '-rm', help='Path to the race model.')
    parser.add_argument('--gender_model', '-gm', help='Path to the gender model.')
    parser.add_argument('--age_model', '-am', help='Path to the age model.')
    parser.add_argument('--net_mode', '-n', help='Residual type [ir, ir_se].', default='ir_se', type=str)
    parser.add_argument('--depth', '-dp', help='Number of layers [50, 100, 152].', default=50, type=int)
    parser.add_argument('--workers', '-w', help='Workers number.', default=4, type=int)

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    drop_ratio = 0.4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predict_attributes(args.race_model, args.gender_model, args.age_model, args.source, args.image_list,
                       args.dest, args.net_mode, args.depth, args.batch_size,
                       args.workers, drop_ratio, device)
