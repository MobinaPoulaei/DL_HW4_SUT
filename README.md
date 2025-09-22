# Deep Learning Homework 4: VAE and DDPM

This repository contains my solutions and code for the fourth homework assignment of the Deep Learning course at Sharif University of Technology. This assignment focuses on generative models, including Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs), and their underlying theoretical frameworks and practical implementations.

---

### **Theoretical Section**

The theoretical portion of this homework covers the mathematical foundations and design principles of generative models. My solutions are submitted in a PDF file as required by the course. The topics covered include:

* **VAE Design**: A step-by-step derivation of the Variational Autoencoder, from likelihood estimation to the final ELBO objective. The derivation covers the challenges of direct likelihood optimization, the use of importance sampling, and the reparameterization trick for stable gradient estimation.
* **Hierarchical VAE (HVAE)**: An analysis of a two-layer HVAE model, including the extraction and interpretation of its ELBO. It also addresses the "posterior collapse" phenomenon and proposes techniques to prevent it.
* **Generative Adversarial Networks (GANs)**: An exploration of the relationship between the GAN objective function and various f-divergences. This section shows how different loss functions correspond to different statistical distances between the real and generated data distributions.
* **Score-Based Models (Diffusion)**: A study on the core objective of diffusion models, which is to learn the score function. It provides a proof of the equivalence between the score-matching objective and the denoising autoencoder loss.
* **Non-Markovian Diffusion**: An investigation into whether the Markovian assumption is necessary for the forward diffusion process. It proves that a non-Markovian process can yield the same marginal distributions, potentially enabling faster generation by creating a deterministic inference process.

---

### **Practical Section**

The practical part of this homework involves implementing and training two state-of-the-art generative models: a Variational Autoencoder and a Denoising Diffusion Probabilistic Model.

#### **Variational AutoEncoder (VAE)**

This project implements a VAE on the **Fashion-MNIST** dataset. It demonstrates how a VAE extends a standard autoencoder by learning a **probabilistic latent space**. The notebook covers:

* **AutoEncoder Implementation**: Implementing a standard autoencoder to compress images into a low-dimensional vector.
* **VAE Implementation**: Building the VAE with a Gaussian latent distribution (`mu` and `log_var`), utilizing the **reparameterization trick** to enable backpropagation through sampling.
* **Loss Function**: Understanding the two components of the VAE loss: **Reconstruction Loss** (to ensure accurate image reconstruction) and **Kullback-Leibler (KL) Divergence** (to regularize the latent space).
* **Generative Capabilities**: Exploring the learned latent space by traversing its dimensions to see how they influence the generated output.
* **Downstream Tasks**: Adding a classification head to the VAE to perform classification from the latent space.
* **Adversarial Examples**: Generating adversarial images using the Fast Gradient Sign Method (FGSM) to test the robustness of the VAE.

#### **Denoising Diffusion Probabilistic Models (DDPM)**

This notebook implements a DDPM, a powerful generative model that progressively adds noise to images in a **forward process** and then learns to reverse this process. The final model is a U-Net trained to generate **MNIST** images from random noise. The implementation includes:

* **U-Net Architecture**: Implementing the core U-Net architecture, which consists of a **contracting path** and an **expansive path**. The notebook utilizes a modified version that incorporates **ResNet blocks**, **attention blocks**, and **time embeddings** to enhance performance.
* **ResNet Block**: Implementing a fundamental building block with Group Normalization, SiLU activation, and a residual connection to handle inconsistencies in channel dimensions.
* **Attention Block**: Implementing an attention sub-module with Group Normalization, a multi-head attention mechanism from scratch, and a feed-forward layer.
* **Time Embedding**: Incorporating time information into the network using **Sinusoidal position embeddings** as proposed in the ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) paper. This allows the model to predict noise for a specific time step.
* **Final Architecture**: Combining all the sub-modules into the final DDPM U-Net model, with a specific architecture for the **contactive path**, **middle block**, and **expansive path** to generate high-quality images.

### **DDPM Training and Sampling**

The DDPM framework consists of two main processes: a **forward process** that adds noise and a **backward process** (denoising) that generates new data.

#### **Training Algorithm**

The training algorithm for the DDPM model is an iterative process. The model learns to denoise images by taking small steps of gradient descent.

1.  A probability $p_{uncond}$ is defined for unconditional training.
2.  The model samples a pair of a clean image and a condition, $(x, c)$, from the data distribution.
3.  A `mask` is created, which is set to 0 with the probability $p_{uncond}$. This mask helps the model learn to generate images without a specific condition.
4.  A time step $t$ is sampled uniformly from 1 to $T$.
5.  Noise $\epsilon$ is sampled from a standard normal distribution.
6.  The model then performs a gradient descent step to minimize the difference between the predicted noise and the actual noise:
    
    $ \nabla_{\theta} ||\epsilon - \epsilon_{\theta}(\sqrt{\bar{\alpha_t}}x_0 + \sqrt{1 - \bar{\alpha_t}}\epsilon, t, c, mask)|| $

This process repeats until the model's parameters converge.

#### **Sampling Algorithm**

The DDPM sampling algorithm uses a classifier guidance method to generate a new image based on a specific condition. The sampling process starts from pure noise and iteratively denoises the image.

1.  A guidance strength $w$ is defined.
2.  The process begins by sampling pure noise $x_T$ from a standard normal distribution.
3.  The algorithm then loops backward from $t=T$ down to 1.
4.  Inside the loop, a noise vector $z$ is sampled from a normal distribution if $t>1$; otherwise, $z$ is set to 0.
5.  The predicted noise, $\epsilon_{pred}$, is calculated using both the conditional and unconditional models:
    
    $ \epsilon_{pred} = (1+w)\epsilon_{\theta}(x_t, t, c, I) - w\epsilon_{\theta}(x_t, t, c, 0) $
    
6.  The image at the next time step, $x_{t-1}$, is then calculated using a formula that includes the current noisy image, the predicted noise, and the noise vector $z$:
    
    $ x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_{pred} \right) + \sigma_t z $
    
7.  After the loop finishes, the final denoised image $x_0$ is returned.

### **Evaluation Metric: Fréchet Inception Distance (FID)**

To evaluate the quality of the generated images, the **Fréchet Inception Distance (FID)** is used. FID is a widely used metric for assessing the similarity between the distributions of real and generated images in a feature space.

#### **How FID Works:**

1.  Both real and generated images are passed through a pretrained convolutional neural network (e.g., InceptionV3 or a lightweight alternative for MNIST).
2.  Deep feature representations are extracted from a specific intermediate layer.
3.  The distributions of these features (for real and generated images) are modeled as multivariate Gaussians.
4.  The Fréchet Distance is then computed between these two Gaussians using the formula:
    
    $ \text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}) $
    
    * $\mu_r$, $\Sigma_r$: Mean and covariance of real image features
    * $\mu_g$, $\Sigma_g$: Mean and covariance of generated image features
    
#### **Interpretation:**

* **Lower FID values** indicate that the generated images are more similar to real images.
* An FID of **0** means the generated distribution is identical to the real one.
