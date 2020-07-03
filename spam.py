import NaiveBayesSolver
import sys

def build_NaiveBayes_model(dataset_directory, model_file, encode):
    nb = NaiveBayesSolver.NaiveBayesSolver()
    nb.train(dataset_directory, model_file, encode)

def predict_with_NaiveBayes(dataset_directory, model_file, encode):
    nb = NaiveBayesSolver.NaiveBayesSolver()
    nb.predict(dataset_directory, model_file, encode)

if __name__ == "__main__":
    (mode, technique, dataset_directory, model_file, encode) = sys.argv[1:6]

    if mode == 'train' and technique == 'bayes':
        build_NaiveBayes_model(dataset_directory, model_file, encode)

    elif mode == 'test' and technique == 'bayes':
        predict_with_NaiveBayes(dataset_directory, model_file, encode)
