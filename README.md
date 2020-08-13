# Alexnet-with-Attention
Trained on CIFAR-10 and CIFAR-100 datasets 

This is a novel implementation of Alexnet, wherein we can visualize the part/area of the image which the network has focused on to determine the class of the image.

The Attention is then used as the local feature and mixed with the global features of FC layers to improve classification accuracy.

![GitHub Logo](results.png)

fig: An Illustration of the Attention regions on test images at different layer of Network.

The implementation is closely related to paper [Learn to pay Attention](https://arxiv.org/abs/1804.02391). With the difference that they have shown results with VGG-16 and 3 layers of attention, and I have implemented with Alexnet and 2 layers of attention.
