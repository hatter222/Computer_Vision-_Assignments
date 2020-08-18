import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):

        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='latin1')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='latin1')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        similarity_thresholds = []
        identification_rates = []
    #Train the classifier on the given training data.
        self.classifier.fit(self.train_embeddings, self.train_labels)
        # Predict similarities for the given test data.
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)
#False alarm rate on the test data and compute the corresponding identification rate.
        for false_alarm_rate in self.false_alarm_rate_range:
            similarity_threshold = self.select_similarity_threshold(similarities, false_alarm_rate)

            idx_unknown = np.where(similarities <= similarity_threshold)
            new_prediction_labels = prediction_labels.copy()
            new_prediction_labels[idx_unknown] = UNKNOWN_LABEL
            identification_rate = self.calc_identification_rate(new_prediction_labels)

            similarity_thresholds.append(similarity_threshold)
            identification_rates.append(identification_rate)
 #doubts
        evaluation_results = {'similarity_thresholds': similarity_thresholds,'identification_rates': identification_rates}
        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):

        unknowns = similarity[np.argwhere(self.test_labels == UNKNOWN_LABEL)]
        #probab of false detection for Unknowns
        false_alarm_rate_unknown = 100-false_alarm_rate*100
        #print(false_alarm_rate_unknown)
        threshold = np.percentile(unknowns,  false_alarm_rate_unknown)
        #print(threshold)
        return threshold 

    def calc_identification_rate(self, prediction_labels):
        #normalise the Identification rate - by diving the number of trueidentification /number of knowns
        num_true_ident = np.sum(np.logical_and(self.test_labels != UNKNOWN_LABEL, prediction_labels == self.test_labels))
        #normalising --no of true Identification
        norm = np.sum(self.test_labels != UNKNOWN_LABEL)

        return num_true_ident/norm





