def evaluate(l):
    return compas.evaluate(compas_models[l.name], COMPAS_TEST_SET, print_freq=1000),           mnist.evaluate(mnist_input_models[l.name], MNIST_TEST_SET, print_freq=1000),           mnist.evaluate(mnist_cnn_5concepts_models[l.name], MNIST_TEST_SET, print_freq=1000),           mnist.evaluate(mnist_cnn_20concepts_models[l.name], MNIST_TEST_SET, print_freq=1000)
