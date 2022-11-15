"""
Saliency map generator.

From:
Neighborhood Attention Transformer.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import random
from classification.nat import *

from timm.models import create_model, load_checkpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
g = torch.Generator()
g.manual_seed(1)


def batch_salient(
    model,
    imgs,
    rounds=10,
    noise_std=0.1,
    noise_mean=0,
):
    for i in range(rounds + 1):
        noise = torch.randn(imgs.size()) * noise_std + noise_mean
        noise = noise.to(imgs.device)
        imgs.requires_grad_()
        salient = None
        if i == 0:
            preds = model(imgs)
            preds_orig = preds.clone()
        else:
            preds = model(imgs + noise)
        scores, indices = torch.max(preds, dim=1)
        scores.backward(torch.ones_like(scores))
        if salient is None:
            salient = torch.max(imgs.grad.data, dim=1)[0]
        else:
            salient += torch.max(imgs.grad.data, dim=1)[0]
    # salient /= rounds
    salient.relu_()
    salients = [to_pil_image(s.cpu().squeeze(0)).convert("RGB") for s in salient]
    return preds_orig, salients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to input image")
    parser.add_argument("--image-out", type=str, help="path to output image")

    parser.add_argument(
        "--img-size",
        default=None,
        nargs=2,
        type=int,
        metavar="N N",
        help="Image resolution (h w, e.g. --input-size 224 224) (default: 224 224)",
    )

    parser.add_argument(
        "--model",
        "-m",
        metavar="NAME",
        default="nat_mini",
        help="model architecture (default: nat_mini)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None, help="Number classes in dataset"
    )
    parser.add_argument(
        "--gp",
        default=None,
        type=str,
        metavar="POOL",
        help=(
            "Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default"
            " if None."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="use cuda (if available)"
    )

    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override mean pixel value of dataset (default: imagenet mean)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override standard deviation of of dataset (default: imagenet std)",
    )

    parser.add_argument("--noise-mean", default=0.0, type=float)
    parser.add_argument("--noise-std", default=1.0, type=float)
    parser.add_argument("--rounds", default=100, type=int)

    args = parser.parse_args()

    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
    )
    model = model.eval()
    print(f"Created model {args.model}")

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    model.to(device)

    args.img_size = args.img_size or (224, 224)
    args.mean = args.mean or IMAGENET_DEFAULT_MEAN
    args.std = args.std or IMAGENET_DEFAULT_STD

    transforms = T.Compose(
        [T.Resize(args.img_size), T.ToTensor(), T.Normalize(args.mean, args.mean)]
    )

    print(f"Loading file {args.image}")
    img = Image.open(args.image)

    img_tensor = transforms(img).to(device)

    print(
        f"Computing salient map for input of size {img_tensor.shape} with"
        f" {args.rounds} rounds."
    )

    _, salients = batch_salient(
        model,
        img_tensor.unsqueeze(0),
        noise_mean=args.noise_mean,
        noise_std=args.noise_std,
        rounds=args.rounds,
    )
    salient = salients[0]

    print(f"Saving salient map to {args.image_out}")
    salient.save(args.image_out)


if __name__ == "__main__":
    main()
