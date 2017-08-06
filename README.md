# reimagined-winner
CIFAR-10 Object Detection with improved accuracy using Fractional MaxPooling with Convolutional Neural Networks

This code uses 2-D Convolutional Neural Networks with the KERAS library and Fractional MaxPooling2D. 

Fractional Maxpooling is an advanced pooling algorithm that uses a fractional pooling ratio unlike the general MaxPooling approaches where pooling usually is done in a integer ratio (generally 2). A Fractional Pooling ratio allows better scaling of the images and allows us to use larger number of convolutional layers to learn the image better at different scales. Use of Pseudo-Random sequences adds randomness to the pooling operation with enables learning more robust features for classification.

The library for 2D Fractional Maxpooling for keras implementations is provided.

FractionalPooling2D('pool_ratio' = 4-D Tuple, 'pseudo_random' = bool, 'overlap' = bool, name = string)

[1,1.44,1.67,1] is a valid 4-D tuple for Pooling ratio for [batch_size, rows, cols, channels]

Use ( batch_input_shape ) while implementing

To have a better understanding about Fractional Maxpooling refer to :
https://arxiv.org/pdf/1412.6071.pdf
