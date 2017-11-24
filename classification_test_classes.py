"""classification_test_classes.py

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


from hashlib import sha1
import test_classes
# import json

from collections import defaultdict
from pprint import PrettyPrinter
pp = PrettyPrinter()

# from game import Agent
from pacman import GameState
# from ghost_agents import RandomGhost, DirectionalGhost
import random, math, traceback, sys, os
# import layout, pacman
# import autograder
# import grading

import data_classifier, samples

VERBOSE = False



# Data sets
# ---------

EVAL_MULTIPLE_CHOICE=True

num_training = 100
TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def read_digit_data(training_size=100, test_size=100):
    rootdata = 'digitdata/'
    # loading digits data
    raw_training_data = samples.load_data_file(rootdata + 'trainingimages', training_size,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    training_labels = samples.load_labels_file(rootdata + "traininglabels", training_size)
    raw_validation_data = samples.load_data_file(rootdata + "validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validation_labels = samples.load_labels_file(rootdata + "validationlabels", TEST_SET_SIZE)
    raw_test_data = samples.load_data_file("digitdata/testimages", test_size,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    test_labels = samples.load_labels_file("digitdata/testlabels", test_size)
    print("Extracting features...")
    feature_function = data_classifier.basic_feature_extractor_digit
    training_data = list(map(feature_function, raw_training_data))
    validation_data = list(map(feature_function, raw_validation_data))
    test_data = list(map(feature_function, raw_test_data))
    return (training_data, training_labels, validation_data, validation_labels, raw_training_data, raw_validation_data, test_data, test_labels, raw_test_data)

def read_suicide_data(training_size=100, test_size=100):
    rootdata = 'pacmandata'
    raw_training_data, training_labels = samples.load_pacman_data(rootdata + '/suicide_training.pkl', training_size)
    raw_validation_data, validation_labels = samples.load_pacman_data(rootdata + '/suicide_validation.pkl', test_size)
    raw_test_data, test_labels = samples.load_pacman_data(rootdata + '/suicide_test.pkl', test_size)
    training_data = []
    validation_data = []
    test_data = []
    return (training_data, training_labels, validation_data, validation_labels, raw_training_data, raw_validation_data, test_data, test_labels, raw_test_data)

def read_contest_data(training_size=100, test_size=100):
    rootdata = 'pacmandata'
    raw_training_data, training_labels = samples.load_pacman_data(rootdata + '/contest_training.pkl', training_size)
    raw_validation_data, validation_labels = samples.load_pacman_data(rootdata + '/contest_validation.pkl', test_size)
    raw_test_data, test_labels = samples.load_pacman_data(rootdata + '/contest_test.pkl', test_size)
    training_data = []
    validation_data = []
    test_data = []
    return (training_data, training_labels, validation_data, validation_labels, raw_training_data, raw_validation_data, test_data, test_labels, raw_test_data)


small_digit_data = read_digit_data(20)
big_digit_data = read_digit_data(1000)

suicide_data = read_suicide_data(1000)
contest_data = read_contest_data(1000)

def tiny_data_set():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(0,0,0), count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    training_labels = [1,        1,            1           , 1           ,      -1     ,      -1]

    validation = [count(1,0,1)]
    validation_labels = [ 1]

    test = [count(1,0,1)]
    test_labels = [-1]

    return (training,training_labels,validation,validation_labels,test,test_labels);


def tiny_data_set_peceptron_and_mira():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    training_labels = [1,            1,            1,          -1      ,      -1]

    validation = [count(1,0,1)]
    validation_labels = [ 1]

    test = [count(1,0,1)]
    test_labels = [-1]

    return (training,training_labels,validation,validation_labels,test,test_labels);


DATASETS = {
    "small_digit_data": lambda: small_digit_data,
    "big_digit_data": lambda: big_digit_data,
    "tiny_data_set": tiny_data_set,
    "tiny_data_set_peceptron_and_mira": tiny_data_set_peceptron_and_mira,
    "suicide_data": lambda: suicide_data,
    "contest_data": lambda: contest_data
}

DATASETS_LEGAL_LABELS = {
    "small_digit_data": list(range(10)),
    "big_digit_data": list(range(10)),
    "tiny_data_set": [-1,1],
    "tiny_data_set_peceptron_and_mira": [-1,1],
    "suicide_data": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP'],
    "contest_data": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP']
}


# Test classes
# ------------

def get_accuracy(data, classifier, feature_function=data_classifier.basic_feature_extractor_digit):
    training_data, training_labels, validation_data, validation_labels, raw_training_data, raw_validation_data, test_data, test_labels, raw_test_data = data
    if feature_function != data_classifier.basic_feature_extractor_digit:
        training_data = list(map(feature_function, raw_training_data))
        validation_data = list(map(feature_function, raw_validation_data))
        test_data = list(map(feature_function, raw_test_data))
    classifier.train(training_data, training_labels, validation_data, validation_labels)
    guesses = classifier.classify(test_data)
    correct = [guesses[i] == test_labels[i] for i in range(len(test_labels))].count(True)
    acc = 100.0 * correct / len(test_labels)
    serialized_guesses = ", ".join([str(guesses[i]) for i in range(len(test_labels))])
    print(str(correct), ("correct out of " + str(len(test_labels)) + " (%.1f%%).") % (acc))
    return acc, serialized_guesses


class GradeClassifierTest(test_classes.TestCase):

    def __init__(self, question, test_dict):
        super(GradeClassifierTest, self).__init__(question, test_dict)

        self.classifier_module = test_dict['classifier_module']
        self.classifier_class = test_dict['classifier_class']
        self.dataset_name = test_dict['dataset_name']

        self.accuracy_scale = int(test_dict['accuracy_scale'])
        self.accuracy_thresholds = [int(s) for s in test_dict.get('accuracy_thresholds','').split()]
        self.exact_output = test_dict['exact_output'].lower() == "true"

        self.automatic_tuning = test_dict['automatic_tuning'].lower() == "true" if 'automatic_tuning' in test_dict else None
        self.max_iterations = int(test_dict['max_iterations']) if 'max_iterations' in test_dict else None
        self.feature_function = test_dict['feature_function'] if 'feature_function' in test_dict else 'basic_feature_extractor_digit'
        
        self.time_scale = float(test_dict['time_scale']) if 'time_scale' in test_dict else 1.0
        self.compare_time = test_dict['compare_time'] if 'compare_time' in test_dict else None
        self.record_time = test_dict['record_time'] if 'record_time' in test_dict else None
        
        self.learning_rates = [float(s) for s in test_dict['learning_rates'].split()] if 'learning_rates' in test_dict else None

        self.max_points = len(self.accuracy_thresholds) * self.accuracy_scale


    def grade_classifier(self, module_dict):
        feature_function = getattr(data_classifier, self.feature_function)
        data = DATASETS[self.dataset_name]()
        legal_labels = DATASETS_LEGAL_LABELS[self.dataset_name]

        classifier_class = getattr(module_dict[self.classifier_module], self.classifier_class)

        if self.max_iterations != None:
            classifier = classifier_class(legal_labels, self.max_iterations)
        else:
            classifier = classifier_class(legal_labels)

        if self.automatic_tuning != None:
            classifier.automatic_tuning = self.automatic_tuning
        
        if self.learning_rates != None:
            classifier.learning_rates = self.learning_rates

        return get_accuracy(data, classifier, feature_function=feature_function)


    def execute(self, grades, module_dict, solution_dict):
    
        if self.compare_time or self.record_time:
            import time
            start = time.time()
    
        accuracy, guesses = self.grade_classifier(module_dict)


        if self.compare_time or self.record_time:
            end = time.time()
              
        if self.compare_time or self.record_time:
            duration = end - start
            if self.compare_time:
                with open(self.compare_time) as f:
                    compare_duration = float(f.read())
                    self.add_message("Original took: %2.4fs, optimized took %2.4fs" % 
                                    (compare_duration, duration))
                if duration * self.time_scale > compare_duration :
                    self.add_message("Solution does not appear to be optimized"+
                        " using numpy.  Try first rerunnig q1.")
                    return self.test_partial(grades, 0, self.max_points)
            if self.record_time:
                with open(self.record_time,"w") as f:
                    f.write(str(duration))
                     
        # Either grade them on the accuracy of their classifer,
        # or their exact
        if self.exact_output:
            gold_guesses = solution_dict['guesses']
            if guesses == gold_guesses:
                total_points = self.max_points
            else:
                self.add_message("Incorrect classification after training:")
                self.add_message("  student classifications: " + guesses)
                self.add_message("  correct classifications: " + gold_guesses)
                total_points = 0
        else:
            # Grade accuracy
            total_points = 0
            for threshold in self.accuracy_thresholds:
                if accuracy >= threshold:
                    total_points += self.accuracy_scale

            # Print grading schedule
            self.add_message("%s correct (%s of %s points)" % (accuracy, total_points, self.max_points))
            self.add_message("    Grading scheme:")
            self.add_message("     < %s:  0 points" % (self.accuracy_thresholds[0],))
            for idx, threshold in enumerate(self.accuracy_thresholds):
                self.add_message("    >= %s:  %s points" % (threshold, (idx+1)*self.accuracy_scale))

        return self.test_partial(grades, total_points, self.max_points)

    def write_solution(self, module_dict, file_path):
        handle = open(file_path, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        if self.exact_output:
            _, guesses = self.grade_classifier(module_dict)
            handle.write('guesses: "%s"' % (guesses,))

        handle.close()
        return True




class MultipleChoiceTest(test_classes.TestCase):

    def __init__(self, question, test_dict):
        super(MultipleChoiceTest, self).__init__(question, test_dict)
        self.ans = test_dict['result']
        self.question = test_dict['question']

    def execute(self, grades, module_dict, solution_dict):
        student_solution = str(getattr(module_dict['answers'], self.question)())
        encrypted_solution = sha1(student_solution.strip().lower().encode("utf-8")).hexdigest()
        if encrypted_solution == self.ans:
            return self.test_pass(grades)
        else:
            self.add_message("Solution is not correct.")
            self.add_message("Student solution: %s" % student_solution)
            return self.test_fail(grades)

    def write_solution(self, module_dict, file_path):
        handle = open(file_path, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True


