import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Union, Callable

from dataset import get_dataloader

hooked_activations = defaultdict(list)


def get_patches(
    activations: Union[torch.Tensor, np.ndarray],
    window_size: int,
    stride: int,
    padding: int = 0,
) -> Union[torch.Tensor, np.ndarray]:
    is_np = False
    if isinstance(activations, np.ndarray):
        activations = torch.from_numpy(activations)
        is_np = True

    if padding != 0:
        activations = F.pad(activations, [padding] * 4)
    out = F.unfold(activations, kernel_size=(window_size, window_size), stride=stride)

    if is_np:
        out = out.numpy()
    return out


def forward_pass(model: torch.nn.Module, dataloader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"[INFO] Running on {device}")
    model = model.to(device)
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
    with torch.inference_mode():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            _ = model(data)


def hook_gen(key: str, layer: Union[None, torch.nn.Module] = None) -> Callable:
    if layer is None:

        def hook_fn(model, input, output):
            hooked_activations[key].append(output.detach().cpu())

        return hook_fn
    else:
        if isinstance(layer, torch.nn.Conv2d):
            k, s, p = (
                int(layer.kernel_size[0]),
                int(layer.stride[0]),
                int(layer.padding[0]),
            )

            def hook_fn(model, input, output):
                inp_patches = get_patches(input[0], k, s, p).transpose(1, 2)
                out_patches = output.reshape(
                    output.shape[0], output.shape[1], -1
                ).transpose(1, 2)
                hooked_activations[key + "_input"].append(inp_patches.detach().cpu())
                hooked_activations[key + "_output"].append(out_patches.detach().cpu())

            return hook_fn
        else:
            raise NotImplementedError(f"Hook function for {layer} is not implemented")


def hook_layers(model: torch.nn.Module, targets: list[str]):
    layer_list = dict([*model.named_modules()])
    for target in targets:
        if target not in layer_list:
            print(f"[INFO] Layer {target} does not exits, skipping...")
            continue
        target_layer = layer_list[target]
        target_layer.register_forward_hook(hook_gen(target, None))
        print(f"[INFO] Hooking {target}: {target_layer}")


if __name__ == "__main__":
    import os
    import numpy as np
    from torchvision import models
    from dataset import get_dataloader

    # from models import load_model, HOOK_TARGETS, create_model

    model_name = "resnet18"
    dataset = "MNIST"
    split = "test"
    experiment_path = "../logs/resnet18_MNIST/"

    dataloader = get_dataloader(dataset, split)
    model = models.resnet18()
    model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = torch.nn.Linear(512, 10, bias=True)
    state_dict = torch.load(
        os.path.join(experiment_path, "best_state.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict)

    hook_targets = ["layer2", "layer3", "layer4"]

    hook_layers(model, hook_targets)
    forward_pass(model, dataloader)

    out_path = f"../data/{dataset.lower()}/"
    os.makedirs(out_path, exist_ok=True)

    for key in hooked_activations:
        hooked_activations[key] = torch.cat(hooked_activations[key], 0).cpu().numpy()
        print(key, "→", hooked_activations[key].shape)
        np.save(
            os.path.join(out_path, f"{key.replace('.', '_')}_{split}_act.npy"),
            hooked_activations[key],
        )
