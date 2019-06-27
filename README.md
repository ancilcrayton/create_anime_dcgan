# Creating Anime Characters using a DCGAN

## Purpose
This project is an implementation of a Deep Convolutional Generative Adversarial Network, which is a an extension of the Generative Adversarial Network to allow for convolutional layers in both the generator and discriminator networks. I apply this to generate anime character faces such as those from the danbooru gallery.

## Usage
Dependencies should be installed by running the command `pip install -r requirements.txt`

First, scrape anime images from danbooru using the `gallery-dl` package:
```
$ gallery-dl https://danbooru.donmai.us/posts?tags=face
```

Next, run the preprocessing script to detect faces and crop images to 64x64x3:
```
$ python preprocess.py
```

Finally, run the main script to train the model and record results into `results/`:
```
$ python run.py
```
## Samples of data
![](img/face_1000.png)

![](img/face_2000.png)

![](img/face_3000.png)

## Important features
Training GANs are tough and this includes DCGANs.

A big issue with learning this dataset was _mode collapse_. Mode collapse is the case when your generator shows a lack of diversity in its output when given random noise as an input. Another issue was domination of the discriminator during training, which often signalled mode collapse. I took the following steps to address these issues in training:

1. **Label Smoothing**: Instead of using hard assignments to train the discriminator network (1=real, 0=fake), I take random draws from a uniform distribution of U(0.7,1.2) for the real images and U(0,0.3) for the fake images. This introduces randomness into the training and aids in avoiding the discriminator loss from reaching 0 rapidly. Training for the generator network should remain the same as the goal for the generator is to learn the distribution of the input data.  
2. **Label Flipping**: Every third epoch, I flip the labels of the real and fake images in training the discriminator network. This method helps with the gradient flow as well as avoids the discriminator from rapidly approaching 0 too quickly by essentially flipping the training process of the discriminator to work with the generator.
3. **Reduce Discriminator Complexity**: The original model choice of the discriminator included three convlutional layers. This included a layer with 128 5x5 filters, a layer with 256 3x3 filters following this, and a layer with 512 3x3 filters. I excluded the middle convolutional layer to reduce the complexity of the model.
