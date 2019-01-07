import sys
import pickle

import numpy as np
import torch
import torchvision.utils as vutils
import cv2
from skimage import color


def load_edge_from_img(path):
    x = cv2.imread(path)
    x = cv2.resize(x, (512, 512)) 
    canny_x = cv2.Canny(np.uint8(x), 50, 150)
    edges_x = torch.from_numpy(canny_x).unsqueeze(0).type(torch.FloatTensor)
    edges_x = 1./255 * edges_x
    return edges_x.unsqueeze(0)



if __name__ == "__main__":
    model_path = sys.argv[1]
    image_path = sys.argv[2]

    model = pickle.load(open(model_path, "rb"))
    edge = load_edge_from_img(image_path)

    final = model(edge.cuda())
    vutils.save_image(final, "popo.jpg", )