import torch
from torch.autograd import Variable
from networks.api_poolnet import build_model, weights_init
import numpy as np
import cv2


class Solver(object):
    def __init__(self, config):
        self.config = config
        self.build_model()
        print("Loading pre-trained model from %s..." % self.config.model)
        if self.config.cuda:
            self.net.load_state_dict(torch.load(self.config.model))
        else:
            self.net.load_state_dict(torch.load(self.config.model, map_location="cpu"))
        self.net.eval()

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))

    def score(self, image_path):
        image = load_image(image_path)
        images = torch.Tensor([image])
        with torch.no_grad():
            images = Variable(images)
            if self.config.cuda:
                images = images.cuda()
            preds = self.net(images, mode=1)
            pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
            return pred


def load_image(path):
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))

    return in_
