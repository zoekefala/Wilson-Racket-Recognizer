#!/usr/bin/python3
import jetson_inference
import jetson_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, etc. (see --help for others)")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)

net = jetson_inference.imageNet(opt.network)

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

# Custom class descriptions for Wilson tennis rackets
class_descriptions = {
    0: "Wilson Blade",
    1: "Wilson Pro Staff",
    2: "Wilson Burn",
    3: "Wilson Ultra"
}

# Modify the class description based on Wilson tennis racket recognition
if class_desc in class_descriptions.values():
    class_idx = list(class_descriptions.values()).index(class_desc)
    class_desc = class_descriptions[class_idx]

print("The image is recognized as '{:s}' (class #{:d}) with {:.2f}% confidence.".format(class_desc, class_idx, confidence * 100))
