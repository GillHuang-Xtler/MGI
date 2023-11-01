import PIL.Image as Image
from os import listdir
import os
import numpy as np

myimages = [] #list of image filenames
dirFiles = os.listdir('.') #list of directory files
dirFiles.sort() #good initial sort but doesnt sort numerically very well
sorted(dirFiles) #sort numerically in ascending order

for files in dirFiles: #filter out all non jpgs
    if '.png' in files:
        myimages.append(files)

final_imgs = [Image.open(fn) for fn in myimages if fn.endswith('.png')]
width, height = final_imgs[0].size
res = Image.new(final_imgs[0].mode, (width * len(final_imgs), height))
for i, im in enumerate(final_imgs):
    res.paste(im, box=(i * width, 0))

res.save('compare_all_MNIST.png' )

# import PIL.Image as Image
# import sys
#
# def concat(images, wfn, hfn, xm, ym):
#     width = wfn(im.size[0] for im in images)
#     height = hfn(im.size[1] for im in images)
#     result = Image.new(images[0].mode, (width, height))
#     x = y = 0
#     for im in images:
#         print(im.size)
#         result.paste(im, (x, y))
#         x += im.size[0] * xm
#         y += im.size[1] * ym
#     return result
#
# def concat_horizontal(images):
#     return concat(images, sum, max, 1, 0)
#
# def concat_vertical(images):
#     return concat(images, max, sum, 0, 1)
#
# def plot(mode = 'h', input_filenames = 'a',):
#     input_images = [Image.open(fn[0][0]) for fn in input_filenames]
#     if mode == '-h':
#         result = concat_horizontal(input_images)
#     elif mode == '-v':
#         result = concat_vertical(input_images)
#
#     return result