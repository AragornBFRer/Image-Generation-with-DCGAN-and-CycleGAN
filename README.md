# Image Generation with DCGAN and CycleGAN

This project focuses on creating and training Generative Adversarial Networks (GANs) for various tasks. The project is divided into two parts: Deep Convolutional GAN (DCGAN) and CycleGAN. In the first part, we will implement a DCGAN to generate emojis from random noise samples. In the second part, we will implement a CycleGAN to translate emojis between Apple-style and Windows-style.

## Part 1: Deep Convolutional GAN (DCGAN)

In this part, we will implement a DCGAN, which utilizes a convolutional neural network as the discriminator and a network composed of transposed convolutions as the generator. The DCGAN consists of three components: the discriminator, the generator, and the training procedure.

### Discriminator Implementation

The DCGAN discriminator is implemented in the `DCDiscriminator` class in `models.py`. 

### Generator Implementation

The DCGAN generator is implemented in the `DCGenerator` class in `models.py`.

### Training Loop Implementation

The training loop for the DCGAN is implemented in the `train` function in `vanilla_gan.py`.

## Part 2: CycleGAN

CycleGAN is a GAN architecture used for image-to-image translation. It allows learning a mapping between two domains without requiring perfectly matched training pairs. In this part, we will implement a CycleGAN to translate emojis between Windows and Apple styles.

### Generator Implementation

The CycleGAN generator is implemented in the `CycleGenerator` class in `models.py`.

### Training Loop Implementation

The training loop for the CycleGAN is implemented in the `train` function in `cycle_gan.py`.

### Cycle Consistency

CycleGAN introduces a cycle consistency loss to constrain the model. The cycle consistency loss measures the similarity between the input images and their reconstructions obtained by passing through both generators in sequence. 

## References

[1] Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. "Least squares generative adversarial networks." In Proceedings of the IEEE International Conference on Computer Vision, pages 2794–2802, 2017.

[2] Alec Radford, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434, 2015.

[3] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. "Unpaired image-to-image translation using cycle-consistent adversarial networks." In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.
