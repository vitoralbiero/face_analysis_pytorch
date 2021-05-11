from os import path

import torch

from .utils import get_time


def save_state(full_model, optimizer, config, accuracy, step=0, model_only=False):
    save_path = config.model_path

    torch.save(
        full_model.model.state_dict(),
        path.join(
            save_path,
            "model_{}_accuracy:{:.4f}_step:{}.pth".format(get_time(), accuracy, step),
        ),
    )

    if not model_only:
        torch.save(
            full_model.head.state_dict(),
            path.join(
                save_path,
                "head_{}_accuracy:{:.4f}_step:{}.pth".format(
                    get_time(), accuracy, step
                ),
            ),
        )
        torch.save(
            optimizer.state_dict(),
            path.join(
                save_path,
                "optimizer_{}_accuracy:{:.4f}_step:{}.pth".format(
                    get_time(), accuracy, step
                ),
            ),
        )


def load_state(
    full_model=None,
    model=None,
    head=None,
    optimizer=None,
    path_to_model=None,
    model_only=False,
    load_head=True,
):
    if full_model is not None:
        model = full_model.model
        head = full_model.head

    model.load_state_dict(torch.load(path_to_model))

    if load_head or not model_only:
        path_to_head = path_to_model.replace("model_", "head_")
        head.load_state_dict(torch.load(path_to_head))

    if not model_only:
        path_to_optimizer = path_to_model.replace("model_", "optimizer_")
        optimizer.load_state_dict(torch.load(path_to_optimizer))
