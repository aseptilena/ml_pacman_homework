"""classification_agents.py

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


from pacman import Directions
from game import Agent

import random
import game
import util

class DummyOptions:
    def __init__(self):
        self.data = "pacman"
        self.training = 25000
        self.test = 100
        self.odds = False
        self.weights = False


import perceptron_pacman

class ClassifierAgent(Agent):
    def __init__(self, training_data=None, validation_data=None, classifier_type="perceptron", agent_to_clone=None, num_training=3):
        from data_classifier import run_classifier, enhanced_feature_extractor_pacman
        legal_labels = ['Stop', 'West', 'East', 'North', 'South']
        if(classifier_type == "perceptron"):
            classifier = perceptron_pacman.PerceptronClassifierPacman(legal_labels,num_training)
        self.classifier = classifier
        self.feature_function = enhanced_feature_extractor_pacman
        args = {'feature_function': self.feature_function,
                'classifier':self.classifier,
                'print_image':None,
                'training_data':training_data,
                'validation_data':validation_data,
                'agent_to_clone': agent_to_clone,
        }
        options = DummyOptions()
        options.classifier = classifier_type
        run_classifier(args, options)
    def get_action(self, state):
        from data_classifier import run_classifier, enhanced_feature_extractor_pacman
        features = self.feature_function(state)
        
        action =  self.classifier.classify([features])[0]

        return action

def score_evaluation(state):
    return state.get_score()
