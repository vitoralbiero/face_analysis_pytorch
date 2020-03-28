import torch
from .utils import get_time
from os import path


def save_state(model, head, optimizer, config, accuracy,
               to_save_folder=False, step=0, model_only=False):
    if to_save_folder:
        save_path = config.save_path
    else:
        save_path = config.model_path

    torch.save(
        model.state_dict(), path.join(save_path,
                                      'model_{}_accuracy:{}_step:{}.pth'.format(get_time(),
                                                                                accuracy,
                                                                                step)))

    if not model_only:
        torch.save(
            head.state_dict(), path.join(save_path,
                                         'head_{}_accuracy:{}_step:{}.pth'.format(get_time(),
                                                                                  accuracy,
                                                                                  step)))
        torch.save(
            optimizer.state_dict(), path.join(save_path,
                                              'optimizer_{}_accuracy:{}_step:{}.pth'.format(get_time(),
                                                                                            accuracy,
                                                                                            step)))


def load_state(model, head, optimizer, config, path_to_model, model_only=False, load_head=True):
    model.load_state_dict(torch.load(path_to_model))

    if load_head or not model_only:
        path_to_head = path_to_model.replace('model_', 'head_')
        head.load_state_dict(torch.load(path_to_head))

    if not model_only:
        path_to_optimizer = path_to_model.replace('model_', 'optimizer_')
        optimizer.load_state_dict(torch.load(path_to_optimizer))
