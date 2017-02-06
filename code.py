# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import caffe

model='lenet.prototxt'
weights='lenet_iter_10000.caffemodel'
image='2.jpeg' 

input_image=caffe.io.load_image(image,color=False)

caffe.set_mode_cpu()

net=caffe.Classifier(model,weights,image_dims=(28,28),raw_scale=255)
out=net.predict([input_image],oversample=False)
print out
print net.blobs['conv1'].data[0][0][0][3:7]

