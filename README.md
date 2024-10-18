# Gaussian Mixture GAN for Unsupervised Learning

## Overview

This project implements a **Gaussian Mixture GAN (GM-GAN)**, combining **Generative Adversarial Networks (GANs)** and Gaussian Mixtures to improve performance in unsupervised learning tasks, particularly for image generation. It is based on pretraining with a vanilla GAN model and further tuning using Gaussian mixtures.

## Problem Definition

The GM-GAN models a mixture of Gaussians to enhance GAN's ability to generate diverse data. The key elements of the Gaussian Mixture Model (GMM) used are:
- \( K \): Number of Gaussians in the mixture.
- \( c \): Range from which the Gaussians' means are sampled.
- \( \sigma \): Scaling factor for the covariance matrices.
- \( \gamma \): Learning rate for model optimization.

## Model Architecture

1. **Generator**: Generates data samples from a noise distribution.
2. **Discriminator**: Distinguishes between real and generated samples.
3. **Gaussian Mixture**: A Gaussian Mixture Model that refines the data generation by adding a probability distribution over multiple Gaussian components.

### Loss Function

1. **Generator Loss**:
   \[
   L_G = \mathbb{E}[\log(1 - D(G(z)))]
   \]
2. **Discriminator Loss**:
   \[
   L_D = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]
   \]
   where \( D \) is the discriminator, \( G \) is the generator, and \( z \) is random noise.

## Results

- **Vanilla GAN FID**: 39.45
- **Static GM-GAN FID**: 19.02 (best result on local static GM-GAN)
- **Hyperparameters**: 
  - Batch size: 64
  - Learning rate: \( 0.0002 \)
  - FID (Best Metric): 16.12 (after fine-tuning GM-GAN)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/matteo-st/gan-gaussian-mixture.git
   cd yourrepo gan-gaussian-mixture


## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --bacth_size 64



