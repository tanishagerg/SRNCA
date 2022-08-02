# SRNCA blogpost

##### **Stars make text bold, maybe put description here**

## Introduction
We see patterns in nature everywhere-  plant textures such as variegation, disease, and veins, animal patterns like tiger stripes and pufferfish skin, and in chemical reactions like the rings and spirals of the oscillating Belousov-Zhabotinsky reaction, to name just a few. But how do they occur? In other words, what makes these patterns arise the way they do? 

PICS/GIFS OF TEXTURES

One way to make a conclusion on how certain natural patterns arise is to analyze the chemicals inside organisms with these patterns to see how they interact with each other. However, this would be a very difficult process because all the chemicals would need to be collected and tested properly against each other. Plus, this would only work on a specific scope: chemical patterns arising in organisms. 
[MAKE MORE NAIVE]

Mathematician Alan Turing came up with another way: a model, which creates what is now known as a Turing Pattern. Through his paper called “The Chemical Basis of Morphogenesis”, he presents a system, which is called a “reaction-diffusion model” with two substances, an activator and an inhibitor, that diffuse throughout a system. The activator substance activates other cells in the system, activating more activators and inhibitors, meanwhile the inhibitor substance prevents further activation. The two substances interact with each other and eventually a stable pattern emerges. ##EXPLAIN MORE?  
##Turing didn't do this

More recently, a work by Niklasson et al. published in distill [(https://distill.pub/selforg/2021/textures/)], presents a Neural cellular automata (NCA), which can learn textures by updating its parameters as it changes to reach a target texture (backpropagating). What is exciting about this model is that not only might it be able to explain textures that we haven’t yet explained, but it could potentially learn patterns of other models, including those in reaction-diffusion models like Turing Patterns.  ##what is the difference between a reaction diffusion system and the Turing model? Is a reaction diffusion system a broader term?	[EXPLAIN MORE ABT THE DIFFERENNCE AND WHAT BACKPROPAGATING IS

https://distill.pub/selforg/2021/textures/ [I worked with a similar implementation of a neural cellular automata texture model available at –link to your fork of the repo– (also make sure to link to your experiment notebook somewhere)] The NCA I looked at is the Symbolic Regression Neural Cellular Automata (SRNCA)  LINK. To judge its texture as it is developing, the model uses the Gram matrix, which in short, flattens an image into different layers containing certain features, and calculates information about their relationships to the target texture, using dot products, to give a quantitative comparison to a target texture, which is used to calculate the loss. ##did I explain the Gram matrix properly? Is the Gram matrix result used as the loss, or is it one of the parts used to calculate the loss? [The loss is the mean squared error between the Gram matrices for the target image and the model output]

[In a convolutional neural network, the numerical values in the hidden layers are often referred to as 'features', especially in deeper layers, many operations removed from the input. These can in fact look somewhat abstract compared to the pixel-by-pixel details of the input. Style loss can be computed from these features by computing a Gram matrix. A Gram matrix is computed as the product of each vector with every other vector's transpose (i.e. the inner, or dot product) in a set of vectors. For our purposes of computing matrices corresponding to image style, the vectors consist of all the pixels for a given feature channel in a hidden layer of the convolutional network, and the set of vectors correspond to each feature channel and its vector of pixel values. Remember that the features, or activations, in a convolutional neural network layer typically have dimensions of batch size, channels, height, and width. Each element of the Gram matrix corresponds to the inner product of a feature channel vector with another feature channel vector, and this is computed for each channel with every other channel. The resulting Gram matrix has as many rows and columns as the number of channels in the layer. We can write the value for the Gram matrix element at position (i,j) as

$$
g_{i,j} = v_i v_j^T
$$

or as

$$
g_{i,j} = \langle v_i, v_j \rangle = v_i \cdot v_j
$$

where the angle brackets and the dot are different ways of specifying the inner product.

To get a style loss that can be back propagated through a model, we compute the Gram matrices for several hidden feature layers in a convolutional neural network and take the mean squared error between the Gram matrices for the image target and the model output. The end result is a loss that corresponds to the difference in textures, or style, (as parsed in the hidden layer features of the convolutional neural network) of an image rather than a direct pixel-by-pixel comparison.]

The loss in a machine learning model is a number that judges its performance, with a lower score meaning a better performance. In the SRNCA, the loss judges how well the machine’s texture matches the target texture. 


### equations test

Testing equation rendering in GitHub flavored Markdown:

$$
\frac{\partial u}{\partial t} = r_u \nabla^2 u - uv^2 + f(1-u)
$$

$$
\frac{\partial v}{\partial t} = r_v \nabla^2 v + uv^2 - v(f+k)
$$

## Hyperparameter Exploration
I was interested in modeling plant textures with this algorithm, so I tested it on these four images of plants around my house, and gave them names that I’ll refer to throughout this post: 





roundleaf eggs- Circular variegation from eggs laid on a maple leaf
orchid petal- Branching pigments on an orchid petal
alocasia- 
Veins on the underside of an Alocasia leaf
snake plant-
Striped variegation on a snake plant

