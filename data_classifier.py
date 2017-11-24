"""data_classifier.py

This file contains feature extraction methods and harness
code for data classification

Champlain College CSI-480, Fall 2017
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import most_frequent
import naive_bayes
import perceptron
import perceptron_pacman
import perceptron_numpy
import samples
import sys
import util
from pacman import GameState
import matplotlib.pyplot as plt
import numpy as np
import logistic

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basic_feature_extractor_digit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.get_pixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.get_pixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basic_feature_extractor_face(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.get_pixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.get_pixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhanced_feature_extractor_digit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features =  basic_feature_extractor_digit(datum)

    "*** YOUR CODE HERE ***"
    util.raise_not_defined()

    return features



def basic_feature_extractor_pacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.get_legal_actions():
        successor = state.generate_successor(0, action)
        food_count = successor.get_food().count()
        feature_counter = util.Counter()
        feature_counter['food_count'] = food_count
        features[action] = feature_counter
    return features, state.get_legal_actions()

def enhanced_feature_extractor_pacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basic_feature_extractor_pacman(state)[0]
    for action in state.get_legal_actions():
        features[action] = util.Counter(features[action], **enhanced_pacman_features(state, action))
    return features, state.get_legal_actions()

def enhanced_pacman_features(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    return features


def contest_feature_extractor_digit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basic_feature_extractor_digit(datum)
    return features

def enhanced_feature_extractor_face(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basic_feature_extractor_face(datum)
    return features

def analysis(classifier, guesses, test_labels, test_data, raw_test_data, print_image):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_image(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - test_labels is the list of true labels
    - test_data is the list of training datapoints (as util.Counter of features)
    - raw_test_data is the list of training datapoints (as samples.Datum)
    - print_image is a method to visualize the features
    (see its use in the odds ratio part in run_classifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = test_labels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print raw_test_data[i]
    #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def print_image(self, pixels, ax=None):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
                
        if ax:            
            pixels = np.asarray(np.asarray(image.pixels).T)
            im = np.zeros((pixels.shape[0], pixels.shape[1],3))
            im[pixels>=1] = [0,1,0]
            im[pixels<1] = [0,0,0]
            ax.imshow(im, interpolation='nearest')
        else :
            print(image)

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python data_classifier.py <options>
  EXAMPLES:   (1) python data_classifier.py
                  - trains the default most_frequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python data_classifier.py -c naive_bayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhanced_feature_extractor_digits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

def learning_rate_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, [float(s) for s in value.split(',')])


def read_command( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['most_frequent', 'nb', 'naive_bayes', 'perceptron', 'perceptron_numpy', 'logistic', 'minicontest'], default='most_frequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-n', '--num_weights', help=default("Num Weights to Print (when --weights enabled), default: 100"), default=100, type="int")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agent_to_clone', help=default("Pacman agent to copy"), default=None, type="str")
    parser.add_option('-l', '--learning_rates', help=default("Learning rates to use for gradient descent, can be a comma separated list or single value"), 
                      default=[0.2], type="str", action='callback', callback=learning_rate_callback)


    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    if not options.classifier == 'minicontest':
        print("using enhanced features?:\t" + str(options.features))
    else:
        print("using minicontest feature extractor")
    print("training set size:\t" + str(options.training))
    if(options.data=="digits"):
        print_image = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).print_image
        if (options.features):
            feature_function = enhanced_feature_extractor_digit
        else:
            feature_function = basic_feature_extractor_digit
        if (options.classifier == 'minicontest'):
            feature_function = contest_feature_extractor_digit
    elif(options.data=="faces"):
        print_image = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).print_image
        if (options.features):
            feature_function = enhanced_feature_extractor_face
        else:
            feature_function = basic_feature_extractor_face
    elif(options.data=="pacman"):
        print_image = None
        if (options.features):
            feature_function = enhanced_feature_extractor_pacman
        else:
            feature_function = basic_feature_extractor_pacman
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if(options.data=="digits"):
        legal_labels = list(range(10))
    else:
        legal_labels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legal_labels or options.label2 not in legal_labels:
            print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print(USAGE_STRING)
            sys.exit(2)

    if(options.classifier == "most_frequent"):
        classifier = most_frequent.MostFrequentClassifier(legal_labels)
    elif(options.classifier == "naive_bayes" or options.classifier == "nb"):
        classifier = naive_bayes.NaiveBayesClassifier(legal_labels)
        classifier.set_smoothing(options.smoothing)
        if (options.autotune):
            print("using automatic tuning for naivebayes")
            classifier.automatic_tuning = True
        else:
            print("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legal_labels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legal_labels,options.iterations)
    elif(options.classifier == "perceptron_numpy"):
        if options.data != 'pacman':
            classifier = perceptron_numpy.OptimizedPerceptronClassifier(legal_labels,options.iterations)
    elif(options.classifier == "logistic"):
        if options.data != 'pacman':
            classifier = logistic.SoftmaxClassifier(legal_labels,options.iterations)
            classifier.learning_rates = options.learning_rates

    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contest_classifier(legal_labels)
    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['agent_to_clone'] = options.agent_to_clone

    args['classifier'] = classifier
    args['feature_function'] = feature_function
    args['print_image'] = print_image

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def run_classifier(args, options):
    feature_function = args['feature_function']
    classifier = args['classifier']
    print_image = args['print_image']
    
    # Load data
    num_training = options.training
    num_test = options.test

    if(options.data=="pacman"):
        agent_to_clone = args.get('agent_to_clone', None)
        training_data, validation_data, test_data = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agent_to_clone, (None, None, None))
        training_data = training_data or args.get('training_data', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validation_data = validation_data or args.get('validation_data', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        test_data = test_data or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        raw_training_data, training_labels = samples.load_pacman_data(training_data, num_training)
        raw_validation_data, validation_labels = samples.load_pacman_data(validation_data, num_test)
        raw_test_data, test_labels = samples.load_pacman_data(test_data, num_test)
    else:
        raw_training_data = samples.load_data_file("digitdata/trainingimages", num_training,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        training_labels = samples.load_labels_file("digitdata/traininglabels", num_training)
        raw_validation_data = samples.load_data_file("digitdata/validationimages", num_test,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validation_labels = samples.load_labels_file("digitdata/validationlabels", num_test)
        raw_test_data = samples.load_data_file("digitdata/testimages", num_test,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        test_labels = samples.load_labels_file("digitdata/testlabels", num_test)


    # Extract features
    print("Extracting features...")
    training_data = list(map(feature_function, raw_training_data))
    validation_data = list(map(feature_function, raw_validation_data))
    test_data = list(map(feature_function, raw_test_data))

    # Conduct training and testing
    print("Training...")
    classifier.train(training_data, training_labels, validation_data, validation_labels)
    print("Validating...")
    guesses = classifier.classify(validation_data)
    correct = [guesses[i] == validation_labels[i] for i in range(len(validation_labels))].count(True)
    print(str(correct), ("correct out of " + str(len(validation_labels)) + " (%.1f%%).") % (100.0 * correct / len(validation_labels)))
    print("Testing...")
    guesses = classifier.classify(test_data)
    correct = [guesses[i] == test_labels[i] for i in range(len(test_labels))].count(True)
    print(str(correct), ("correct out of " + str(len(test_labels)) + " (%.1f%%).") % (100.0 * correct / len(test_labels)))
    analysis(classifier, guesses, test_labels, test_data, raw_test_data, print_image)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naive_bayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.find_high_odds_features(label1,label2)
        if(options.classifier == "naive_bayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print(string3)
        print_image(features_odds)


    if((options.weights) & ((options.classifier in ["perceptron_numpy", "logistic"]))):

        for i,l in enumerate(classifier.legal_labels):
            features_weights = classifier.find_high_weight_features(l, options.num_weights)
            print(("=== Plotting Features with high weight for label %d ==="%l))
            ax = plt.subplot(1, len(classifier.legal_labels), 1+i)
            print_image(features_weights, ax)
        plt.show()

if __name__ == '__main__':
    # Read input
    args, options = read_command( sys.argv[1:] )
    # Run classifier
    run_classifier(args, options)
