"""logistic.py
Author:              Victoria Hayes
Class:               CSI-480-02
Assignment:          Supervised Learning Assignment
Date Assigned:       I dunno, November 8th?
Due Date:            November 27, 2017 11:59 pm

Description:
uses tensorflow to use linear regression to figure out what an ascii image represents (q4 and q5)

Certification of Authenticity:
I certify that this is entirely my own work, except where I have been provided
code by the instructor, or given fully-documented references to the work of
others. I understand the definition and consequences of plagiarism and
acknowledge that the assessor of this assignment may, for the purpose of
assessing this assignment:
    Reproduce this assignment and provide a copy to another member of academic
    staff; and/or Communicate a copy of this assignment to a plagiarism checking
    service (which may then retain a copy of this assignment on its database for
    the purpose of future plagiarism checking)

This code has been adapted from that provided by Prof. Joshua Auerbach:
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

        accuracies_stored = []
        weights_stored = []

        for learning_rate in self.learning_rates:
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
            accuracies_stored.append(self.sess.run(accuracy, feed_dict={self.x: test_images,
                                                                        self.y_: test_labels}))
            weights_stored.append(self.W.eval())
            cat = "meow"
        best_index = accuracies_stored.index(max(accuracies_stored))
        tf.assign(self.W, weights_stored[best_index])


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

        weight_label = self.W[:, label].eval()
        w_list = weight_label.tolist()
        unique_values = set(w_list)
        sorted_list = list(unique_values)
        sorted_list.sort(reverse=True)
        high_weights = []

        for value in sorted_list:
            indeces = np.argwhere(weight_label == value)
            for index in indeces:
                if len(high_weights) >= num:
                    break
                x = index[0] // 28
                y = index[0] % 28
                high_weights.append(tuple([x, y]))
        res = np.asarray(high_weights)
        return high_weights
