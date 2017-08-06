# reimagined-winner
CIFAR-10 Object Detection with improved accuracy using Fractional MaxPooling with Convolutional Neural Networks

This code uses 2-D Convolutional Neural Networks with the KERAS library and Fractional MaxPooling2D. 

Fractional Maxpooling is an advanced pooling algorithm that uses a fractional pooling ratio unlike the general MaxPooling approaches where pooling usually is done in a integer ratio (generally 2). A Fractional Pooling ratio allows better scaling of the images and allows us to use larger number of convolutional layers to learn the image better at different scales. Use of Pseudo-Random sequences adds randomness to the pooling operation with enables learning more robust features for classification.

To have a better understanding about Fractional Maxpooling refer to :
https://arxiv.org/pdf/1412.6071.pdf
