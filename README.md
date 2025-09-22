# Deep Learning Homework 4 - Sharif University of Technology
### Instructor: Dr. Mahdieh Soleymani

This repository contains my solutions and code for the fourth homework assignment of the Deep Learning course at Sharif University of Technology. This assignment focuses on generative models, including Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs), and their underlying theoretical frameworks and practical implementations.

---

## **Theoretical Section**

The theoretical portion of this homework covers the mathematical foundations and design principles of generative models. My solutions are submitted in a PDF file as required by the course. The topics covered include:

* **VAE Design**: A step-by-step derivation of the Variational Autoencoder, from likelihood estimation to the final ELBO objective. The derivation covers the challenges of direct likelihood optimization, the use of importance sampling, and the reparameterization trick for stable gradient estimation.
* **Hierarchical VAE (HVAE)**: An analysis of a two-layer HVAE model, including the extraction and interpretation of its ELBO. It also addresses the "posterior collapse" phenomenon and proposes techniques to prevent it.
* **Generative Adversarial Networks (GANs)**: An exploration of the relationship between the GAN objective function and various f-divergences. This section shows how different loss functions correspond to different statistical distances between the real and generated data distributions.
* **Score-Based Models (Diffusion)**: A study on the core objective of diffusion models, which is to learn the score function. It provides a proof of the equivalence between the score-matching objective and the denoising autoencoder loss.
* **Non-Markovian Diffusion**: An investigation into whether the Markovian assumption is necessary for the forward diffusion process. It proves that a non-Markovian process can yield the same marginal distributions, potentially enabling faster generation by creating a deterministic inference process.

---

## **Practical Section**

The practical part of this homework involves implementing and training two state-of-the-art generative models: a Variational Autoencoder and a Denoising Diffusion Probabilistic Model.

### **Variational AutoEncoder (VAE)**

This project implements a VAE on the **Fashion-MNIST** dataset. It demonstrates how a VAE extends a standard autoencoder by learning a **probabilistic latent space**. The notebook covers:

* **AutoEncoder Implementation**: Implementing a standard autoencoder to compress images into a low-dimensional vector.
* **VAE Implementation**: Building the VAE with a Gaussian latent distribution (`mu` and `log_var`), utilizing the **reparameterization trick** to enable backpropagation through sampling.
* **Loss Function**: Understanding the two components of the VAE loss: **Reconstruction Loss** (to ensure accurate image reconstruction) and **Kullback-Leibler (KL) Divergence** (to regularize the latent space).
* **Generative Capabilities**: Exploring the learned latent space by traversing its dimensions to see how they influence the generated output.
* **Downstream Tasks**: Adding a classification head to the VAE to perform classification from the latent space.
* **Adversarial Examples**: Generating adversarial images using the Fast Gradient Sign Method (FGSM) to test the robustness of the VAE.

### **Denoising Diffusion Probabilistic Models (DDPM)**

This notebook implements a DDPM, a powerful generative model that progressively adds noise to images in a **forward process** and then learns to reverse this process. The final model is a U-Net trained to generate **MNIST** images from random noise. The implementation includes:

* **U-Net Architecture**: Implementing the core U-Net architecture, which consists of a **contracting path** and an **expansive path**. The notebook utilizes a modified version that incorporates **ResNet blocks**, **attention blocks**, and **time embeddings** to enhance performance.
* **ResNet Block**: Implementing a fundamental building block with Group Normalization, SiLU activation, and a residual connection to handle inconsistencies in channel dimensions.
* **Attention Block**: Implementing an attention sub-module with Group Normalization, a multi-head attention mechanism from scratch, and a feed-forward layer.
* **Time Embedding**: Incorporating time information into the network using **Sinusoidal position embeddings** as proposed in the ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) paper. This allows the model to predict noise for a specific time step.
* **Final Architecture**: Combining all the sub-modules into the final DDPM U-Net model, with a specific architecture for the **contactive path**, **middle block**, and **expansive path** to generate high-quality images.
