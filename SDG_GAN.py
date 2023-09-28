from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

class Generator:
    def __init__(self, latent_dim, label_dim, output_dim):
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.label_dim,))
        model_input = Concatenate()([noise, label])

        x = Dense(128, activation='relu')(model_input)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        model_output = Dense(self.output_dim, activation='tanh')(x)

        return Model([noise, label], model_output)

    def generate(self, noise, labels):
        return self.model.predict([noise, labels])

class Discriminator:
    def __init__(self, input_dim, label_dim):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.model = self.build_model()

    def build_model(self):
        data = Input(shape=(self.input_dim,))
        label = Input(shape=(self.label_dim,))
        model_input = Concatenate()([data, label])

        x = Dense(128)(model_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(x)
        x = Dense(32)(x)
        feature_output = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(feature_output)

        model_output = Dense(1, activation='sigmoid')(x)

        return Model([data, label], model_output), Model([data, label], feature_output)

    def train(self, real_samples, real_labels, fake_samples, fake_labels, valid, fake):
        d_loss_real = self.model[0].train_on_batch([real_samples, real_labels], valid)
        d_loss_fake = self.model[0].train_on_batch([fake_samples, fake_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss

class CGAN:
    def __init__(self, input_dim, label_dim, latent_dim):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.generator = Generator(self.latent_dim, self.label_dim, self.input_dim)
        self.discriminator = Discriminator(self.input_dim, self.label_dim)
        self.gan = self.build_gan()

    def build_gan(self):
        self.discriminator.model[0].trainable = False
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.label_dim,))
        x = self.generator.model([noise, label])
        gan_output = self.discriminator.model[0]([x, label])
        gan = Model([noise, label], gan_output)
        return gan

    def compile_models(self, optimizer):
        self.discriminator.model[0].compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.gan.compile(loss='mse', optimizer=optimizer) # we use Mean squared error insted of binary-crossentropy because of feature matching

    def train(self, X, Y, epochs, batch_size):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

    # To just generate fraudulent transactions
    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        fake_labels = np.ones((num_samples,))  # Assuming "1" is the label for fraudulent transactions
        return self.generator.generate(noise, fake_labels)

        # Create feature extractor model from Discriminator
        feature_extractor = self.discriminator.model[1]

        for epoch in range(epochs):
            # Select a random batch of transactions and their labels
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_samples = X.iloc[idx]
            real_labels = Y.iloc[idx]

            # Generate a batch of new transactions
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            num_classes = 2 # Number of unique classes in 'Class' column
            fake_labels = np.random.randint(0, num_classes, batch_size)
            fake_samples = self.generator.generate(noise, fake_labels)

            # Train the discriminator
            d_loss = self.discriminator.train(real_samples, real_labels, fake_samples, fake_labels, valid, fake)

            # Feature Matching
            feature_extractor = self.discriminator.model[1]
            # real features
            real_features = feature_extractor.predict([real_samples, real_labels])
            # fake features
            fake_features = feature_extractor.predict([fake_samples, fake_labels])
            # compute feature matching loss
            fm_loss = K.mean(K.abs(real_features - fake_features))

            # Train the generator with feature matching loss
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch([noise, fake_labels], valid) + fm_loss

            # Print the progress
            print("Epoch {}/{} - D loss: {:.4f} - G loss: {:.4f}".format(epoch + 1, epochs, d_loss[0], g_loss))

            # Save the generator weights every 10 epochs
            """if (epoch + 1) % 10 == 0:
                self.generator.model.save_weights("generator_weights_epoch_{}.h5".format(epoch + 1))"""


