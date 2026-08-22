# Introduction

In this task we explored Transfer Learning, Attack methods, defense methods, and model explainability.

# Attack Methods
The main purpose of attack methods is to alter the data in such a way that it doesn't appear altered to humans, but the model fails on that data even if it was trained on this same data.

One popular method is the attack method: `FGSM attack`.

## Core Idea
When training our model, we seek to update the weights in the direction that minimizes the loss. In FGSM attack, we update the pixel's value such that we move it in the direction that maximizes the loss!

Formula:
$$
x_{new} = x + \epsilon \cdot \text{sign}\big(\nabla_x \mathcal{L}(\theta, x, y)\big)
$$

## Code

```python
def fgsm_attack(image, epsilon, data_grad):
    # Get the direction of the gradient (either +1 or -1 for every pixel)
    sign_data_grad = data_grad.sign()
    
    # Multiply by epsilon (attack strength) and add to the original image
    # Adds "noise" to the image
    bad_image = image + epsilon * sign_data_grad
    
    return bad_image
```


## Example
Here is an example from the CalTech101 dataset:

Original Image:
<img src="./Media/image.png" width="155">

Image after Attack:
<img src="./Media/image-1.png" width="155">

## How Epsilon affects the attack strength

As you can see from the formula above, the parameter $\epsilon$ tells us how far in the opposite direction of the gradient do we want to go.
Thus, $\epsilon$ represents the strength of the attack, more $\epsilon$ more damaged data!

Here is a comparison between different epsilons and how they affect the final accuracy:
For each $\epsilon$, we choose a random subset of the data of size 50, and attack the data and check the model's accuracy.
<img src="./Media/image-2.png" width="226">

As expected, when $\epsilon$ is zero, the attack doesn't actually do a thing as:
$$
x_{new} = x + 0= x
$$
Also, as expected, the more we increase $\epsilon$, the more the accuracy decreases.
But one might ask, shouldn't the accuracy strictly monotonically decrease when $\epsilon$ increases? Why is this not the case here?

The answer to this would be: Statistical Luck. When $\epsilon$ is large, it might overshoot to a place that doesn't necessarily make the model fail. But this depends on luck.

This is why we want to find a small $\epsilon$ that almost grantees bad accuracy, as a large $\epsilon$ can overshoot and give us an undesirable result.

This minimum $\epsilon$ could be between 0.1 and 0.15, as after that we can see some fluctuations start to appear.

<img src="./Media/image-3.png" width="238">

# Model Explainability


# Defense Methods
