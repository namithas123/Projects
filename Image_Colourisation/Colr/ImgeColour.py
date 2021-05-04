import argparse
import matplotlib.pyplot as plt
from colorizers import *
parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/gray12_0.jpg')
opt = parser.parse_args()
# load colorizers
net_colorizer = network(pretrained=True).eval()
# size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
out_img = postprocess_tens(tens_l_orig, net_colorizer(tens_l_rs).cpu())
plt.figure(figsize=(9,5))
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.title('Original')
# plt.axis('off')
plt.subplot(1,1,1)
plt.imshow(out_img)
plt.title('Output ')
plt.axis('off')
plt.show()
