from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Add, Dense, ReLU, LeakyReLU, BatchNormalization, Input, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import f1_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

class Generator:
    def __init__(self, latent_dim, output_dim):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # First hidden layer
        model.add(Dense(171, input_dim=self.latent_dim, activation='relu'))

        # Second hidden layer
        model.add(Dense(63, activation='relu'))
        model.add(BatchNormalization())

        # Third hidden layer
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())

        # Fourth hidden layer
        model.add(Dense(186, activation='relu'))

        # Output layer
        model.add(Dense(self.output_dim, activation='tanh'))

        return model

    def generator_save(self):
        self.save('generator_model.h5')

    def generate(self, noise):
        return self.model.predict(noise)

class Discriminator:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(83, input_dim=self.input_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(134))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train(self, real_samples, fake_samples, valid, fake):
        d_loss_real = self.model.train_on_batch(real_samples, valid)
        d_loss_fake = self.model.train_on_batch(fake_samples, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss

class GAN:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = Generator(self.latent_dim, self.input_dim)
        self.discriminator = Discriminator(self.input_dim)
        self.gan = self.build_gan()

    def build_gan(self):
        self.discriminator.model.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator.model(gan_input)
        gan_output = self.discriminator.model(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        return gan

    def compile_models(self, optimizer):
        self.discriminator.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)


    def train(self, X, epochs, batch_size):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train the discriminator
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_samples = X.iloc[idx].values
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.generate(noise)

            d_loss = self.discriminator.train(real_samples, fake_samples, valid, fake)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)

            # Print the progress
            print("Epoch {}/{} - D loss: {:.4f} - G loss: {:.4f}".format(epoch+1, epochs, d_loss[0], g_loss))

            # Save the generator weights every 10 epochs
            if (epoch+1) % 10 == 0:
                self.generator.model.save_weights("generator_weights_epoch_{}.h5".format(epoch+1))

            # Update the discriminator and GAN weights
            self.discriminator.model.trainable = False
            self.gan.layers[1].trainable = True
            self.gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)
            self.gan.layers[1].trainable = False
            self.discriminator.model.trainable = True
            self.discriminator.model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Set up the GAN architecture
input_dim = X_train_nf.shape[1]
latent_dim = 100

# Initialize GAN model
gan_model = GAN(input_dim=input_dim, latent_dim=latent_dim)

# Compile GAN model
gan_model.compile_models(optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Train GAN model
gan_model.train(X_train_nf, epochs=10, batch_size=32)

# Generate synthetic data
"""noise = np.random.normal(0, 1, (1000, 100))
generated_samples = gan_model.generator.generate(noise)

# Define random search parameters
random_search_params = {
    'optimizer': ['adam', 'rmsprop', 'sgd'],
    'learning_rate': uniform(1e-5, 1e-2),
    'latent_dim': randint(10, 300),
    'beta_1': uniform(0.1, 0.9),
    'momentum': uniform(0.1, 0.9),
    'batch_size': [16, 32, 64]
}

# Calculate the mean and variance for real and generated samples
real_mean = np.mean(minority_data, axis=0)
real_var = np.var(minority_data, axis=0)
generated_mean = np.mean(generated_samples, axis=0)
generated_var = np.var(generated_samples, axis=0)

# Compare the mean and variance
print("Real mean: ", real_mean)
print("Generated mean: ", generated_mean)
print("Real variance: ", real_var)
print("Generated variance: ", generated_var)"""