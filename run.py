import os
import glob
import numpy as np
import time
from models import build_discriminator, build_generator, build_adversarial_model
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from scipy.misc import imread
from utils import normalize, save_rgb_img, write_log

# Create path to save sampled images from generator
if os.path.isdir('results/img/') == False:
    os.system('mkdir results/img/')

# Create function that trains model and execute upon running script
def train():
    # Set main parameters
    start_time = time.time()
    dataset_dir = "data/*.*"
    batch_size = 64
    z_shape = 100
    epochs = 1000
    dis_learning_rate = 0.005
    gen_learning_rate = 0.005
    dis_momentum = 0.5
    gen_momentum = 0.5
    dis_nesterov = True
    gen_nesterov = True
    
    # Define optimizers (can change to Adam later)
    #dis_optimizer = SGD(lr=dis_learning_rate, momentum=dis_momentum, nesterov=dis_nesterov)
    #gen_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)
    dis_optimizer = Adam(lr=1e-3, decay=1e-5)
    gen_optimizer = Adam(lr=1e-4, decay=1e-5)

    # Load images
    all_images = []
    for index, filename in enumerate(glob.glob(dataset_dir)):
        all_images.append(imread(filename, flatten=False, mode='RGB'))
    
    # Compile images into array and normailze them
    X = np.array(all_images)
    X = normalize(X)
    X = X.astype(np.float32)
    
    # Build the GAN models
    dis_model = build_discriminator()
    dis_model.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    gen_model = build_generator()
    gen_model.compile(loss='mse', optimizer=gen_optimizer)

    adversarial_model = build_adversarial_model(gen_model, dis_model)
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    
    # Record training data to the tensorboard
    tensorboard = TensorBoard(log_dir="results/logs/{}".format(time.time()), write_images=True, write_grads=True, write_graph=True)
    tensorboard.set_model(gen_model)
    tensorboard.set_model(dis_model)

    for epoch in range(epochs):
        print("--------------------------")
        print("Epoch:{}".format(epoch))

        dis_losses = []
        gen_losses = []

        num_batches = int(X.shape[0] / batch_size)

        print("Number of batches:{}".format(num_batches))
        for index in range(num_batches):
            print("Batch:{}".format(index))

            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            # z_noise = np.random.uniform(-1, 1, size=(batch_size, 100))

            generated_images = gen_model.predict_on_batch(z_noise)

            # visualize_rgb(generated_images[0])

            """
            Train the discriminator model
            """

            dis_model.trainable = True

            image_batch = X[index * batch_size:(index + 1) * batch_size]

            y_real = np.ones((batch_size, )) * 0.9
            y_fake = np.zeros((batch_size, )) + 0.1 # not multiplication

            dis_loss_real = dis_model.train_on_batch(image_batch, y_real)
            dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)

            d_loss = (dis_loss_real+dis_loss_fake)/2
            print("d_loss:", d_loss)

            dis_model.trainable = False

            """
            Train the generator model(adversarial model)
            """
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            # z_noise = np.random.uniform(-1, 1, size=(batch_size, 100))

            g_loss = adversarial_model.train_on_batch(z_noise, y_real)
            print("g_loss:", g_loss)

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

        """
        Sample some images and save them
        """
	# Sample images every one hundred epochs
        if epoch % 10 == 0:
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            gen_images1 = gen_model.predict_on_batch(z_noise)

            for img in gen_images1[:2]:
                save_rgb_img(img, "results/img/gen_{}.png".format(epoch))

        print("Epoch:{}, dis_loss:{}".format(epoch, np.mean(dis_losses)))
        print("Epoch:{}, gen_loss: {}".format(epoch, np.mean(gen_losses)))

        """
        Save losses to Tensorboard after each epoch
        """
        write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

    """
    Save models
    """
    gen_model.save("results/models/generator_model.h5")
    dis_model.save("results/models/discriminator_model.h5")

    print("Time:", (time.time() - start_time))

# Execute training upon running script
if __name__ == '__main__':
    train()
