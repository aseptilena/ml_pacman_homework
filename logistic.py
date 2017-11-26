"""logistic.py

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

# Perceptron implementation
import util

PRINT = True

import numpy as np
import tensorflow as tf


class SoftmaxClassifier:
    """
    A softmax (multinomial logistic regression) classifier.
    
    This Class will perform softmax classification using TensorFlow

    Note that the variable 'datum' in this code refers to a counter of features
    """

    def __init__(self, legal_labels, max_iterations):
        self.legal_labels = legal_labels
        self.type = "logistic"
        self.max_iterations = max_iterations
        self.learning_rates = [0.2]
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.matmul(self.x, self.W) + self.b
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # create TensorFlow session
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def train(self, training_data, training_labels, validation_data, validation_labels):
        """
        The training loop for the softmax classifier passes through the training data several
        times and updates the weight vector for each label based on the cross entropy loss
        
        You will need to setup tensor flow variables, computation graph, 
        and optimization procedure, then run the training step self.max_iterations
        times.
        
        This should be very similar to what is shown
           https://www.tensorflow.org/get_started/mnist/beginners
        except for where the data is coming from
        
        Important note: this should operate in batch mode, using all training_data 
            for each batch
        """
        self.features = list(training_data[0].keys())  # could be useful later

        learning_rate = self.learning_rates[0]

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        for x in range(self.max_iterations):
            batch_xs = np.asarray([datum.values_as_numpy_array() for datum in training_data])
            converted_labels = []
            for label in training_labels:
                index_value = label
                array = [0 for x in range(10)]
                array[index_value] = 1
                converted_labels.append(array)
            batch_ys = np.asarray(converted_labels)
            self.sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

        test_images = np.asarray([datum.values_as_numpy_array() for datum in validation_data])
        test_labels = []

        for label in validation_labels:
            index_value = label
            array = [0 for x in range(10)]
            array[index_value] = 1
            test_labels.append(array)
        test_labels = np.asarray(test_labels)

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(self.sess.run(accuracy, feed_dict={self.x: test_images,
                                                 self.y_: test_labels}))


    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        output = tf.argmax(self.y, 1)
        return self.sess.run(output, feed_dict={self.x: [datum.values_as_numpy_array() for datum in data]})


def find_high_weight_features(self, label, num=100):
    """
    Returns a list of the num features with the greatest weight for some label
    """

    # this function is optional for this classifier, but if you want to
    # visualize the weights of this classifier, you will need to implement
    # it

    util.raise_not_defined()
