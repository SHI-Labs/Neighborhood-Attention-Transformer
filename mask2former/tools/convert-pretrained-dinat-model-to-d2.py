#!/usr/bin/env python

import pickle as pkl
import sys

import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if "model" in obj.keys():
        obj = obj["model"]
    if "state_dict" in obj.keys():
        obj = obj["state_dict"]

    res = {"model": obj, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
