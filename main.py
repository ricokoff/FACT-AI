import argparse

parser = argparse.ArgumentParser(description='Pass parameters to SENN models.')
parser.add_argument('--mnist_index', type=int, metavar='mnist_index', default=18,
                    help='An index for the MNIST dataset')
parser.add_argument('--compas_lambda', metavar='compas_lambda', default='E1',
                    help='Gradient regularization coefficient for COMPAS. Possible values are:'
                         'ZERO, E4, E3, E2, E1, ONE')
parser.add_argument('--mnist_lambda', metavar='mnist_lambda', default='E1',
                    help='Gradient regularization coefficient for MNIST. Possible values are:'
                         'ZERO, E4, E3, E2, E1, ONE')

args = parser.parse_args()
mnist_index = args.mnist_index
compas_lambda = args.compas_lambda
mnist_lambda = args.mnist_lambda

import warnings

from tqdm import tqdm
from api.parameters import RegLambda, HType, NConcepts
from api.models import load_compas, load_mnist

from api import accuracy, explicitness, faithfulness, stability

warnings.filterwarnings("ignore")

print("===== A REVIEW OF SELF-EXPLAINING NEURAL NETWORKS =====")
print("Loading available models:")

compas_models = {}
mnist_input_models = {}
mnist_cnn_5concepts_models = {}
mnist_cnn_20concepts_models = {}

for l in RegLambda:
    formatted_lambda = '{:0.0e}'.format(l.value) if l.value not in [0, 1] else l.value
    print("- lambda={}".format(formatted_lambda))
    print("Loading COMPAS...")
    compas_models[l.name] = load_compas(reg_lambda=l, show_specs=False)
    print("Loading MNIST (htype=input)...")
    mnist_input_models[l.name] = load_mnist(reg_lambda=l, h_type=HType.INPUT, show_specs=False)
    print("Loading MNIST (htype=cnn, concepts=5)...")
    mnist_cnn_5concepts_models[l.name] = load_mnist(reg_lambda=l, show_specs=False)
    print("Loading MNIST (htype=cnn, concepts=20)...")
    mnist_cnn_20concepts_models[l.name] = load_mnist(reg_lambda=l, n_concepts=NConcepts.TWENTY, show_specs=False)
    print("")

print("> Reproducibility")
print("##### Accuracy #####")

print("Computing accuracy of each model on the relative test set...")
compas_accuracy = {}
mnist_input_accuracy = {}
mnist_cnn_5concepts_accuracy = {}
mnist_cnn_20concepts_accuracy = {}

pbar = tqdm(RegLambda)
for l in pbar:
    formatted_lambda = '{:0.0e}'.format(l.value) if l.value not in [0, 1] else l.value
    pbar.set_description("Evaluating accuracy with lambda={}".format(formatted_lambda))
    compas_accuracy[l.name] = accuracy.evaluate_compas(compas_models[l.name])
    mnist_input_accuracy[l.name] = accuracy.evaluate_mnist(mnist_input_models[l.name])
    mnist_cnn_5concepts_accuracy[l.name] = accuracy.evaluate_mnist(mnist_cnn_5concepts_models[l.name])
    mnist_cnn_20concepts_accuracy[l.name] = accuracy.evaluate_mnist(mnist_cnn_20concepts_models[l.name])

accuracies = [compas_accuracy, mnist_input_accuracy, mnist_cnn_5concepts_accuracy, mnist_cnn_20concepts_accuracy]
titles = ["COMPAS", "MNIST Input", "MNIST CNN 5 Concepts", "MNIST CNN 20 Concepts"]
accuracy.plot_accuracy_comparison(accuracies, titles)

print("")
print("The explainability plot shows a sample image from the MNIST datset on the left, the relevance scores "
      "in the middle and the prototypes of each basic concepts on the right."
      "The idea is that high relevance score for a certain concept should indicate similarity between "
      "the sample image and the prototypes of that image. For our models, this is often not the case. "
      "Different models can be used in this function, including different setting for the number of "
      "prototypes and layout.")
print("")

index = mnist_index
explicitness.plot_digit_activation_concept_grid(mnist_cnn_5concepts_models[mnist_lambda], index, layout='horizontal')

print("##### Extension - Synthetic images instead of prototypes #####")
print("")
print("As an extension to the original paper, we defined a function that searches for image settings "
      "activating the concepts most, attempting to produces human interpretable concepts. Unfortunately, "
      "the synthetic images are not very human interpretable. Different settings are applied to visualize "
      "the synthetic images.")
print("alpha is a regularization term pushing other concepts to 0 to a certain extend, making the current "
      "concept more strongly visible.")
print("beta is a regularization term able to reduce the noise, as shown in the image below.")


prototypes = explicitness.visualize_concepts(mnist_cnn_5concepts_models[mnist_lambda],
                                             p1=[0, 0, 1],
                                             p2=[0, 1, 1],
                                             method=["zero"] * 3,
                                             x0=None,
                                             show_loss=False,
                                             print_freqs=[0, 0],
                                             best_of=1,
                                             compact=False)

print("##### Faithfulness #####")
print("The faithfullness of the concept describe how important the contribution of each concept is. "
      "In other words, what happens if we take one concept and remove. Below we can see a MNIST sample "
      "and the relevance score plotted per concept, indicated by the blue bars. In orange, The probability "
      "drop when the concept is removed.")


index = mnist_index
faithfulness.plot_faithfulness(mnist_cnn_20concepts_models[mnist_lambda], index, show_h=False, show_htheta=False)

print("##### Stability #####")
print("The authors of the original paper describe a tradeoff between the model performance and "
      "the regularization strength on theta(x). The higher the regularization, the more linear"
      " (and therefor explanable) the model gets, the lower the predictive performance becomes. And vice versa.")
print("")

stability.plot_lipschitz_accuracy(models=list(compas_models.values()),
                                  reg_lambdas=[l for l in RegLambda],
                                  accuracies=list(compas_accuracy.values()))

print("##### Stability - MNIST #####")
print("To make sure that the model keeps it local linearity, small changes/perturbations "
      "should not have great influence on the relevance scores produced by theta(x). "
      "As shown below, for a regularized model the small perturbations have no influence "
      "on the relevance scores. Unregularized however, shows quite some difference in the "
      "relevance scores.")
print("")


index = mnist_index
model = mnist_cnn_5concepts_models[RegLambda[mnist_lambda].name]
unregularized_model = mnist_cnn_5concepts_models[RegLambda.ZERO.name]
stability.plot_digit_noise_activation_regularized_unregularized(model, unregularized_model, index, 5)

print("##### Stability - COMPAS #####")
print("As similar to the MNIST models, we applied a small change in the inputs to determine "
      "of the model is not influenced by it. In two settings we change the ethnicity from african "
      "american to other. It is shown that of the regularized model the relevance scores stay "
      "relatively the same, while the unregularized model is showing some differences. This can "
      "also contribute to the fairness of an algorithm, since ethnicity should not have a huge "
      "impact on the recidivism scores.")


x = [
    0.,  # Two_yr_Recidivism
    0.23,  # Number_of_Priors
    0.,  # Age_Above_FourtyFive
    1.,  # Age_Below_TwentyFive
    1.,  # African_American
    0.,  # Asian
    0.,  # Hispanic
    0.,  # Native_American
    0.,  # Other
    0.,  # Female
    0.,  # Misdemeanor
]

y = [
    0.,  # Two_yr_Recidivism
    0.23,  # Number_of_Priors
    0.,  # Age_Above_FourtyFive
    1.,  # Age_Below_TwentyFive
    0.,  # African_American
    0.,  # Asian
    0.,  # Hispanic
    0.,  # Native_American
    1.,  # Other
    0.,  # Female
    0.,  # Misdemeanor
]

model = compas_models[RegLambda[compas_lambda].name]
unregularized_model = compas_models[RegLambda.ZERO.name]
stability.plot_input_values_regularized_unregularized_explanation(model, unregularized_model, [x, y])
