from ucimlrepo import fetch_ucirepo
import math
import numpy as np
import sklearn

# fetch dataset
wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets



class model:
    def __init__(self):
        pass

    def entropy_calculator(self, probabilities): # probabilities will be given as a list
        H_x = 0
        for item in probabilities:
            if item == 0:
                continue
            else:
                Entropy = -(item*math.log2(item))
                H_x += Entropy
        return H_x
    def frequency_finders(self, data):
        pdf = {}
        for value in data:
            if value in pdf:
                pdf[value] += 1
            else:
                pdf[value] = 1
        return pdf
    def pdf_finder(self, frequencies): # will be parsed as a dictionary
        pdf = []
        total = sum(frequencies.values())
        for value in frequencies.keys():
            pdf.append(frequencies[value]/total)
        return pdf

    def entropy_selection(self, data):
        frequencies = self.frequency_finders(data)
        pdf = self.pdf_finder(frequencies)
        entropy = self.entropy_calculator(pdf)
        return entropy

    def dataset_selection(self, features):
        entropies = []
        for item in features.columns:
            data = features[item]
            entropies.append(self.entropy_selection(data))
        return entropies

    def sorting(self, list):
        for i in range(0, len(list)):
            for j in range(i, len(list)):
                if list[i]<list[j]:
                    list[i], list[j] = list[j], list[i]
        return list

    def joint_entropy_finder(self, x, y): # x and y will be lists containing probabilities of events in x and y
        x_array = np.array(x)
        y_array = np.array(y)
        multiplication_table = np.outer(y, x)
        multiplication_table = multiplication_table.flatten()
        entropy = self.entropy_calculator(multiplication_table)
        return entropy

    def mutual_information(self, h_x, h_y, h_xy):
        return h_x + h_y - h_xy

def main(features, targets):
    mutual_informations = []
    untrained_model = model()
    for i in range (0, len(features.columns)):
        frequencies_x = untrained_model.frequency_finders(features.iloc[:, i])
        pdf_x = untrained_model.pdf_finder(frequencies_x)
        frequencies_y = untrained_model.frequency_finders(targets.iloc[:, 0])
        pdf_y = untrained_model.pdf_finder(frequencies_y)
        joint_entropy = untrained_model.joint_entropy_finder(pdf_y, pdf_x)
        entropy_x = untrained_model.entropy_selection(features.iloc[:, i])
        entropy_y = untrained_model.entropy_selection(targets.iloc[:, 0])
        mutual_informations.append(untrained_model.mutual_information(entropy_x, entropy_y, joint_entropy))
        untrained_model.sorting(mutual_informations)



if __name__ == "__main__":
    main(X, y)



















