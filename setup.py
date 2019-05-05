from setuptools import setup

setup(
    name='neural_net',
    version='0.1',
    packages=['models', 'models.utils', 'models.generative', 'models.generative.autoencoders',
              'models.generative.autoencoders.vae', 'models.discriminative',
              'models.discriminative.artificial_neural_networks',
              'models.discriminative.artificial_neural_networks.hebbian_network', 'models.semi_supervised',
              'models.semi_supervised.utils', 'models.semi_supervised.deep_generative_models',
              'models.semi_supervised.deep_generative_models.layers',
              'models.semi_supervised.deep_generative_models.models',
              'models.semi_supervised.deep_generative_models.inference'],
    url='',
    license='',
    author='simon j pelletier',
    author_email='',
    description=''
)
