# SRNCA blogpost

**A discussion on how we model texture generations, hyperparameter exploration on the Symbolic Regression Neural Cellular Automata model, and what they can tell us about the patterns we see in the world around us.**

## Introduction
We see patterns in nature everywhere-  plant textures such as variegation, disease, and veins, animal patterns like tiger stripes and pufferfish skin, and in chemical reactions like the rings and spirals of the oscillating Belousov-Zhabotinsky reaction, to name just a few. But how do they occur? In other words, what makes these patterns arise the way they do? 
| | | |
| ------------- | ------------- | ------------- |
|![image](https://github.com/tanishagerg/SRNCA/blob/master/blogpostgifs/8wMA.gif?raw=true)|![image](https://user-images.githubusercontent.com/103375681/182692154-16768a5a-6587-4328-93f5-7eb73f71e415.png)|![image](https://user-images.githubusercontent.com/103375681/182697402-d0218dd3-fdf2-402a-ae4f-fd5e26494bf9.png)|
|Belousov-Zhabotinsky reaction|Giraffe fur pattern|Haworthia leaf pattern|
|![image](https://user-images.githubusercontent.com/103375681/182706291-569bc34e-f8bb-4607-ad91-e76d29581ad8.png)|![image](https://user-images.githubusercontent.com/103375681/182697757-0dd3c27b-6ad7-4c27-b818-b72b8cb2cd9b.png)|![image](https://user-images.githubusercontent.com/103375681/182693201-0387786a-0c1c-467b-af2b-7df1da2e842a.png)|
|Pufferfish skin pattern|Gasteria leaf pattern|Cheetah fur pattern|

One first guess at how animal and plant patterns arise would be that they are hard-coded in deterministic genes. That would mean every limb, tooth, and stripe corresponds exactly to its specific description in a gene. However, the information storage requirements for this approach would be massive: every cell would have to have the entire blueprint for the entire organism, similar why an uncompressed tif file (where every pixel is described exactly) takes so much storage. An “uncompressed genome” describing an animal or plant would be immense, and such an organism’s fitness would likely collapse under the energetic and information-processing needs of such a genome. 

So what if the genomes could a "compressed genome" instead? Similar to a compressed tif file, where the images are essentially stored as partial instructions for how to reconstruct the images base on decompression rules (the decompression algorithm here is the other half of the reconstruction instructions). Here is where we have a more viable explanation of organismal patterning and development:  encoding of rules for development (e.g. genes describing how to produce and react to a body landscape of diffusing morphogens). 

Mathematician Alan Turing came up with a way to model this, which creates what is now known as a Turing Pattern. Through his paper called “The Chemical Basis of Morphogenesis”, he presents a mathmatical system, which is called a “reaction-diffusion model” where two substances spread (diffuse) throughout the system interact with each other (react) and eventually a stable pattern emerges. **If mathmatical rulesets like these are given to all cells to automate the pattern formation in a model, it is called a Cellular Automata. ##[I NEEDA EXPLAIN BETTER HOW IT RELATES TO CAs?]**

More recently, Neural Cellular Automatas (NCAs) have been created, including a work by **Niklasson et al. [is this the correct name?] published in distill [(https://distill.pub/selforg/2021/textures/)]. **NCA models preserve the local dynamics of rule-based CA (and many physical systems besides), but use neural networks in place of rules based on logical or mathematical functions as in a conventional CA. Texture NCA can learn textures by iteratively updating the parameters of its layers via backpropagation.**

What is exciting about NCAs model is that they may be able to learn plausible processes that could explain how different textural patterns are generated, including processes that might be further distilled into mathematical models like reaction-diffusion systems.

I worked with a similar implementation Niklasson et al's, called the the Symbolic Regression Neural Cellular Automata (SRNCA) available at [link to your fork of the repo]. To calculate the machine's style loss (The style loss in a machine learning model is a number that judges its performance as it is developing, with a lower score meaning a better performance. In the SRNCA, the loss judges how well the machine’s texture matches the target texture), the model, in short, flattens its texture into different layers that contain vectors of numerical values often refered to as features, that compare the target image texture's features to the model's texture's current features. Next, a Gram matrix is calculated for each layer. A Gram matrix is stores the dot product product of each vector with every other vector's transpose fo (i.e. the inner, or dot product) in a set of vectors. Each resulting Gram matrix has as many rows and columns as the number of channels in the layer. We can write the value for the Gram matrix element at position (i,j) as

$$
g_{i,j} = v_i v_j^T
$$

or as

$$
g_{i,j} = \langle v_i, v_j \rangle = v_i \cdot v_j
$$

where the angle brackets and the dot are different ways of specifying the inner product.

Finally, from the Gram Matrices, the loss is calculated, which is the mean squared error between the Gram matrices for the target image and the model's curent output. This loss corresponds to the difference in textures, or style, (as parsed in the hidden layer features of the convolutional neural network) of an image rather than a direct pixel-by-pixel comparison.

## Hyperparameter Exploration
I was interested in modeling plant textures with this algorithm, so I tested it on these four images of plants around my house, and gave them names that I’ll refer to throughout this post: 

|roundleaf - Circular variegation from eggs laid on a maple leaf|orchid petal- Branching pigments on an orchid petal|alocasia- Veins on the underside of an Alocasia leaf|snake plant- Striped variegation on a snake plant|
| ------------- | ------------- | ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182497701-d9fb831d-5105-44ff-95e5-fe0cc097ad5a.png)|![image](https://user-images.githubusercontent.com/103375681/182497726-258800cf-225b-4276-9e7c-b41c144c9fbc.png)|![image](https://user-images.githubusercontent.com/103375681/182497758-cd5479a6-8f84-4044-9651-0301c43e7a97.png)|![image](https://user-images.githubusercontent.com/103375681/182497777-704412ef-3f93-44b9-9ffe-4c932a73324a.png)|


I tested hyperparameters to learn more about their impact on the performance of the SRNCA. 

This includes: 

| Parameter | description |
|---|---|
| number of channels | needawrite |
| number of hidden channels | needawrite |
| max ca steps | needawrite |
| batch size | needawrite |
| number of filters | needawrite |
| update rate | needawrite |
| learning rate | needawrite |

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
*From mean rating 4.4:*
| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182500958-d2b34b30-8639-41a7-8989-320a7ad6880f.png)|![image](https://user-images.githubusercontent.com/103375681/182500973-f7f19deb-1d35-4f59-a134-5dff295b2887.png)|

But, not so much for generating a “bad” pattern with a mean resting of .5. However, these textures do certainly apear to not be as great as the ones from the mean rating of 4.4:

*From mean rating .5:*
| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182501297-1d2a05de-7b72-44fd-b1e9-dcf4bc1670a3.png)|![image](https://user-images.githubusercontent.com/103375681/182501320-3f15e2a9-c910-43c9-99a7-cc2566ff3689.png)|

With further investigation into these and other statistics models in the future, it might be possible to confidently identify what clusters of parameters behave similarly. 

## Performance relationships across different textures
Performance across different textures seemed to be the most tight relationship that I saw in exploring the hyperparameters. The way I tested this relationship was by generating 10 random sets of hyperparameters and testing them on all the four textures. I found that combinations that didn’t work well on one pattern didn’t do very well on the other textures, and combinations that did work well seemed to work great on others too. 
### Hyperparameter set 5
For example, my fifth set yielded great results across the different target textures:

| Parameter | number |
|---|---|
| channels | 15 |
| hidden channels | 96 |
| max ca steps | 40 |
| batch size | 2 |
| filters | 4 |
| update rate | 0.559994317 |
| learning rate | 1e-05 |

| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182501810-455eaa8e-6f55-4140-abe4-2a3d6e340e77.png)|![image](https://user-images.githubusercontent.com/103375681/182501821-5821dca8-031c-4ab0-9339-3b82d50617cb.png)|
|![image](https://user-images.githubusercontent.com/103375681/182501845-3fe599cd-56d9-486b-bef9-72668f3aed8c.png)|![image](https://user-images.githubusercontent.com/103375681/182501852-877dc531-4e10-411f-a93f-09a4392eaf93.png)|

### Hyperparameter set 8
And my eight set yielded horrible results across the different target textures:

| Parameter | number |
|---|---|
| channels | 6 |
| hidden channels | 32 |
| max ca steps | 40 |
| batch size | 2 |
| filters | 6 |
| update rate | 0.980292 |
| learning rate | 1e-05 |

| | |
| ------------- | ------------- |
|![image](https://user-images.githubusercontent.com/103375681/182501909-6a10f866-e607-4af2-a2a3-5b8c156aa49e.png)|![image](https://user-images.githubusercontent.com/103375681/182501919-36e7dcde-60c5-433c-a843-cc63ea7a82e5.png)|
|![image](https://user-images.githubusercontent.com/103375681/182501938-30c78ca5-230b-4911-a078-c158ea8d1ed3.png)|![image](https://user-images.githubusercontent.com/103375681/182501945-b2477e2a-b411-4d08-b0ae-29b9b1dea8d1.png)|


**At the end, I graphed the relationship between the roundleaf final loss versus other other textures' final losses for each of the ten sets of parameters.**

![image](https://user-images.githubusercontent.com/103375681/182501434-93db2b14-7d33-4839-9224-974fc564a2ec.png)

It showed  a positive, linear correlation, meaning that as the set yeilded a higher loss (a worse pattern) for the roundleaf texture, it would also yeild a higher loss for the other textures, and vice versa. It was also interesting that each distribution had an increasing spread as the loss increased. This makes sense because as set yeilds a less and less acurate result (increasing loss) for the roundleaf texture, there are more possibilities of textures to be generated which increases the range of possibile outcomes for the other textures. Finally, even with only ten samples, each of the relationships seemed to have a y-intercept of approximately 0. This is notable because it suggests that a hyperparameter set that is extremley close to zero for one texture, would likley also minimize the loss for other textures too, and therefore that there is an optimal set of hyperparameters.   

A next step for hyperparameter exploration in the algorithm is to find this optimal set by using an evolutionary strategy to evolve the parameters to the best combinations. 

## Conclusion
[It was awesome to see first-hand how well the SRNCA can learn textures, even from pictures of random plants around my house. This model allows us to find a ruleset for any texture imaginable, and it is exciting to think about how well it would work once we find the way to optimize it. Apart from optimization, It has the potential to learn rulesets in past models, like Turing patterns, and can also help us explain how patterns all around us that have not yet been explained, arise. ]

It was nice to see first-hand how well the SRNCA can learn textures, even from pictures of random plants around my house. Apart from optimization, It has the potential to learn rulesets in past models, like Turing patterns, and can also help us explain how patterns all around us that have not yet been explained, arise. 

If you would like to do some hyperparameter exploration on this model yourself, you can use [this notebook]. 

## Sources








