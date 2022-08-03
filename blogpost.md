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

|roundleaf - Circular variegation from eggs laid on a maple leaf|orchid petal- Branching pigments on an orchid petal|alocasia- Veins on the underside of an Alocasia leaf|snake plant- Striped variegation on a snake plant|
| ------------- | ------------- | ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182497701-d9fb831d-5105-44ff-95e5-fe0cc097ad5a.png)|![image](https://user-images.githubusercontent.com/103375681/182497726-258800cf-225b-4276-9e7c-b41c144c9fbc.png)|![image](https://user-images.githubusercontent.com/103375681/182497758-cd5479a6-8f84-4044-9651-0301c43e7a97.png)|![image](https://user-images.githubusercontent.com/103375681/182497777-704412ef-3f93-44b9-9ffe-4c932a73324a.png)|


I tested hyperparameters to learn more about their impact on the performance of the SRNCA. 

This includes: 
number of channels
Number of hidden channels
Batch size
Number of filters
Update rate 
Learning rate

I conducted most of my trials on the round leaf eggs texture leaf with eggs laid on it. What made this a great training image was that, out of these four textures, it had the highest variability in the final loss, the loss at step 16383. The loss usually ranged anywhere from .5 to 5. A higher variability of outcomes from the same parameters makes it easier to pinpoint specific trends because the data is more spread out. 

##### Distribution of final losses across different textures in 10 different random sets of hyperparameters:
![image](https://user-images.githubusercontent.com/103375681/182497556-3518b809-0342-4b0c-83e3-11ccf985c799.png)
*Each set of hyperparameters was tested on each of the four textures and corresponds to a color. As shown, the roundleaf target image had the most variability.*

## Effectiveness of the Gram Function
The first thing I did was use random values for the hyperparameters and see the result. 
I recorded the numbers for the parameters used, and the final loss that the algorithm gave, after 16383 steps. I also gave my own rating for the texture, on a scale of 0-5, with 5 being great: the color and unique patterns were extremely close to the target texture, and 0 being poor: the color or pattern was completely different from the target texture. 


Here are some examples of generated textures and how I decided to score them:
Target image: roundleaf eggs 
| | | |
| ------------- | ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182499254-a546525d-0dfc-491e-841d-00cb0b67f9fb.png)|![image](https://user-images.githubusercontent.com/103375681/182499269-fec905b3-cb2b-45ab-acf1-b1354add8f10.png)|![image](https://user-images.githubusercontent.com/103375681/182499281-2b421ef9-ca5d-4774-ade5-2ab5b9167a2a.png)|
|My rating:  5|My rating: 4|My rating: 3|
|![image](https://user-images.githubusercontent.com/103375681/182499314-e7e97845-e3d7-4547-92e0-218d1a8ca899.png)|![image](https://user-images.githubusercontent.com/103375681/182499328-388d973a-49c8-413f-9380-33a80f5a84cc.png)|![image](https://user-images.githubusercontent.com/103375681/182499339-04d465c0-74d6-4b65-9383-1b09dd6c49f5.png)|
|My rating: 2|My rating: 1|My rating: 0|

Looking at 50 random textures that were trained to reach the roundleaf eggs texture, there was a pretty strong negative linear relationship between the final loss and my rating, which makes sense as a lower loss should mean a better pattern. It shows that the Gram matrix is quite effective at capturing the elements of style important to at least one human viewer.
![image](https://user-images.githubusercontent.com/103375681/182499610-6cabeeb2-1183-4573-9660-803626cacddf.png)

## Relationships between parameter values and texture and outcomes
There didn’t seem to be any convincing linear relationships, however, between the hyperparameters and the final loss, as shown by these graphs of parameter values versus the loss:
![image](https://user-images.githubusercontent.com/103375681/182500467-d5598eb8-336f-40e0-96f1-b47474c3f031.png)

So, I looked at clusters to see if certain combinations of hyperparameters near certain values result in specific outcomes. I tried both the Umap and the PCA model. The Umap model always gave me unviable results to try [, negative values for model channels, for example,], but I tried some of the suggestions that the PCA model gave me. 

Although my sample size was only two, it seemed that the PCA model worked pretty well for finding a combination of parameters that generated a “good” pattern with a mean rating of 4.4:
From mean rating 4.4:
| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182500958-d2b34b30-8639-41a7-8989-320a7ad6880f.png)|![image](https://user-images.githubusercontent.com/103375681/182500973-f7f19deb-1d35-4f59-a134-5dff295b2887.png)|

But, not so much for generating a “bad” pattern with a mean resting of .5. However, these textures do certainly apear to not be as great as the ones from the mean rating of 4.4:

From mean rating .5:
| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182501297-1d2a05de-7b72-44fd-b1e9-dcf4bc1670a3.png)|![image](https://user-images.githubusercontent.com/103375681/182501320-3f15e2a9-c910-43c9-99a7-cc2566ff3689.png)|
With further investigation into these and other statistics models in the future, it might be possible to confidently identify what clusters of parameters behave similarly. 

## Performance relationships across different textures
Performance across different textures seemed to be the most tight relationship that I saw in exploring the hyperparameters. Combinations that didn’t work well on one pattern didn’t do very well on the other textures, and combinations that did work well seemed to work great on others too.

Back to the 10 randomly generated sets of hyperparameters tested on each of the four textures, according to the graph below of the roundleaf result (final loss) versus the other textures’ final
results for the same set.
![image](https://user-images.githubusercontent.com/103375681/182501434-93db2b14-7d33-4839-9224-974fc564a2ec.png)
*the regression lines shown are exponential*

For example, this set of hyperparameters (set 5) yielded great results across the different target textures:):
number of channels = 15
Number of hidden channels = 96
Max ca steps = 40
Batch size = 2
Number of filters = 4
Update rate = 0.559994317
Learning rate = 1e-05
| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182501810-455eaa8e-6f55-4140-abe4-2a3d6e340e77.png)|![image](https://user-images.githubusercontent.com/103375681/182501821-5821dca8-031c-4ab0-9339-3b82d50617cb.png)|
|![image](https://user-images.githubusercontent.com/103375681/182501845-3fe599cd-56d9-486b-bef9-72668f3aed8c.png)|![image](https://user-images.githubusercontent.com/103375681/182501852-877dc531-4e10-411f-a93f-09a4392eaf93.png)|

And this set of hyperparameters (set 8) yielded horrible results across the different target textures:
number of channels = 6
Number of hidden channels = 32
Max ca steps = 40
Batch size = 2
Number of filters = 6
Update rate = 0.980292
Learning rate = 1e-05
| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182501909-6a10f866-e607-4af2-a2a3-5b8c156aa49e.png)|![image](https://user-images.githubusercontent.com/103375681/182501919-36e7dcde-60c5-433c-a843-cc63ea7a82e5.png)|
|![image](https://user-images.githubusercontent.com/103375681/182501938-30c78ca5-230b-4911-a078-c158ea8d1ed3.png)|![image](https://user-images.githubusercontent.com/103375681/182501945-b2477e2a-b411-4d08-b0ae-29b9b1dea8d1.png)|






