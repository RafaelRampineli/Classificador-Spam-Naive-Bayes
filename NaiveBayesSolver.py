# Classificador Naive bayes

# Imports
import sys, os
import warnings
from math import log
import json
import numpy as np

class NaiveBayesSolver:

    spam_counts = {}
    notspam_counts = {}

    class_counts = {}
    model_vocabulary_size = {}
    
    def build_word_counts_model(self, files_path, encode):

        # Isto irá construir um modelo de contagem de palavras, que irá contar a ocorrência de palavra dentro de documentos e construir as contagens de classe a partir dos dados de treinamento

        # Serão dois tipos de modelos
        # 1) Binário - apenas indicando presença ou ausência de palavras em doc.
        # 2) Contínuo -neste modelo, o número de vezes que a palavra ocorre no doc (frequência) é contabilizado.

        print("Realizando a contagem de palavras no dataset de treino. Dados sendo processados. Aguarde... \n")

        dirs = os.listdir(files_path)
        for class_dir_name in dirs:
            # Aplica o FOR individual em cada arquivo dentro do diretório
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                # Concatena um arquivo individual
                document = os.path.join(files_path, class_dir_name, f)
                with open(document, 'r', encoding = encode, errors="ignore") as file:
                    words = file.read().split()
                    distinct_words = sorted(set(words))
                    for word in distinct_words:

                        self.class_counts[class_dir_name]['word_counts'][word]['frequency_count'] = \
                            self.class_counts\
                                .setdefault(class_dir_name,{})\
                                .setdefault('word_counts',{})\
                                .setdefault(word, {})\
                                .setdefault('frequency_count', 0) + words.count(word)

                        self.class_counts[class_dir_name]['word_counts'][word]['presence_count'] = \
                            self.class_counts\
                                .setdefault(class_dir_name,{})\
                                .setdefault('word_counts',{})\
                                .setdefault(word, {})\
                                .setdefault('presence_count', 0) + 1

                    self.class_counts[class_dir_name]['total_count'] = \
                        self.class_counts\
                            .setdefault(class_dir_name, {})\
                            .setdefault('total_count', 0) + 1
                    pass
            pass

        #print ("#### Top 10 palavras mais associadas com spam: ####")
        #spam_word_counts = self.class_counts["spam"]['word_counts']
        #print ("##################################################")

        # Para obter as palavras que são menos associadas com spam, primeiro obtemos as palavras "not spam"
        #notspam_word_counts = self.class_counts["notspam"]['word_counts']
        #least_associated_with_spam = {k: v for k, v in notspam_word_counts.items() if k not in spam_word_counts}
        #print ("#### Top 10 palavras menos associadas com spam: ####")
        #print ("##################################################")
        pass


    def train(self, files_path, model_file, encode):
        print("############################################################################")
        print("##### Treinando o Algoritmo Naive Bayes. Isso pode levar algum tempo. ######")
        print("############################################################################")

        self.build_word_counts_model(files_path, encode)
        self.save_model_to_file(model_file, encode)

        print("######################### Treinamento Finalizado. ##########################")
        print("############################################################################\n")

    def save_model_to_file(self, file_name, encode):

        print ("Salvando o modelo no diretório e arquivo especificado... \n")
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        with open(file_name, 'w+', encoding = encode, errors='ignore') as filehandle:
            json.dump({'class_counts': self.class_counts},
                      filehandle, sort_keys=True, indent=4, ensure_ascii=False)


    def load_model_from_file(self, file_name, encode):
       
        with open(file_name, 'r', encoding = encode) as filehandle:
            model = json.load(filehandle, encoding = "ISO-8859-1")
            self.class_counts = model['class_counts']
        self.model_vocabulary_size = len(self.class_counts.get('spam').get('word_counts')) \
                                     + len(self.class_counts.get('notspam').get('word_counts'))


    def get_word_presence_class_log_prob(self, word, output_class):
        # Adicionado suavização laplace, para evitar problemas devido a palavras invisíveis
        # Obteremos alguns avisos ocasionais devido à descodificação unicode, que funcionalmente não causa nenhum problema no sistema 
        # e portanto podem ser ignorados
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return log(float(self.class_counts[output_class]['word_counts'].get(word, {}).get('presence_count', 0) + 1.0) / \
                       (self.class_counts[output_class]['total_count'] + self.model_vocabulary_size))

    def get_word_frequency_class_log_prob(self, word, output_class):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return log(float(self.class_counts[output_class]['word_counts'].get(word,{}).get('frequency_count',0) + 1.0)/\
                (self.class_counts[output_class]['total_count'] + self.model_vocabulary_size))

    def get_class_prob(self, output_class):
        total_count = 0
        for key in self.class_counts.keys():
            total_count = total_count + self.class_counts[key]['total_count']
        return log(float(self.class_counts[output_class]['total_count'])/\
            (total_count))

    def predict_simple(self, document, encode):
        with open(document, 'r', encoding = encode, errors="ignore") as f:
            contents = f.read()
            words = set(contents.split())
            max_prob = -sys.maxsize
            max_class = None
            for output_class in ('spam', 'notspam'):
                p = self.get_class_prob(output_class)
                for word in words:
                    p = p + self.get_word_presence_class_log_prob(word, output_class)
                if p > max_prob:
                    max_prob = p
                    max_class = output_class
            return max_class

    def predict_with_word_frequencies(self, document, encode):
        with open(document, 'r', encoding = encode, errors="ignore") as f:
            contents = f.read()
            words = contents.split()
            max_prob = -sys.maxsize
            max_class = None
            for output_class in ('spam', 'notspam'):
                p = self.get_class_prob(output_class)
                for word in words:
                    p = p + self.get_word_frequency_class_log_prob(word, output_class)
                if p > max_prob:
                    max_prob = p
                    max_class = output_class
            return max_class


    def predict(self, files_path, model_file, encode):
        self.load_model_from_file(model_file, encode)
        dirs = os.listdir(files_path)
        print("############################################################################")
        print("######### Isso pode levar algum tempo. Tem café ai????######################\n")
        print("############################################################################")
        print("######## Fazendo previsões com a presença de palavras no modelo... #########")
        print("############################################################################\n")
        for class_dir_name in dirs:
            total_test_cases = 0
            correct_predictions = 0
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                total_test_cases += 1
                predicted_class = self.predict_simple(document, encode)
                if predicted_class == class_dir_name:
                    correct_predictions += 1
            print("### Prevendo Acurácia para a classe: %s ### \n" % class_dir_name)
            print("Total de observações analisadas: %d  " % (total_test_cases))
            print("Observações com Classificação Correta: %d  " % (correct_predictions))
            print("Acurácia: %.2f  " % (float(correct_predictions) / total_test_cases))

        print("############################################################################")
        print("######## Fazendo previsões com a frequência de palavras no modelo...########")
        print("############################################################################\n")
        for class_dir_name in dirs:
            total_test_cases = 0
            correct_predictions = 0
            for f in os.listdir(os.path.join(files_path, class_dir_name)):
                document = os.path.join(files_path, class_dir_name, f)
                total_test_cases += 1
                predicted_class = self.predict_with_word_frequencies(document, encode)
                if predicted_class == class_dir_name:
                    correct_predictions += 1
            print("### Prevendo Acurácia para a classe: %s ### \n" % class_dir_name)
            print("Total de observações analisadas: %d  " % (total_test_cases))
            print("Observações com Classificação Correta: %d  " % (correct_predictions))
            print("Acurácia: %.2f  " % (float(correct_predictions) / total_test_cases))
        print("############################################################################")
        print("############################ Modelo Concluído ##############################")
        print("############################################################################")



