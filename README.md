# Classificador de Spam utilizando Naive Bayes

Mini Projeto 1 desenvolvido durante o módulo 16 do curso de Machine Learning da Formação Cientista de Dados da Data Science Academy (DSA).

O objetivo deste projeto é construir um classificador de Spam usando o algoritmo de classificação Naive Bayes. Construiremos esse algoritmo a partir do zero sem usar bibliotecas, o que será muito útil para construção de aplicações analíticas.

O modelo de documento que usaremos aqui é um modelo de saco de palavras (bag of words). Usaremos dois tipos de modelo bag of words:

A. Com base na presença de palavra (se uma palavra aparece no documento ou não, o que tornará os atributos de entrada binários)
B. Com base na frequência de palavras (frequência de ocorrência de palavra no documento, o que tornará os atributos de entrada contínuos).

Um saco de palavras (bag of words) é uma representação de um texto como um agrupamento de palavras, sem qualquer consideração da sua estrutura gramatical ou da ordem das palavras. É simplesmente um histograma sobre as palavras da língua, e cada documento é representado como um vetor sobre estas palavras. As entradas neste vetor simplesmente correspondem à presença ou à ausência da palavra correspondente (quando se utiliza o tipo A acima ou a frequência da ocorrência da palavra quando se usa o caso B acima).

## EXECUÇÃO:

Nosso classificador é na verdade um aplicativo que será executado via linha de comando. Você precisa ter o interpretador do Python 3 instalado. 
A execução do aplicativo deve ser feito da seguinte forma:

-1- Abra um terminal ou prompt de comando.
-2- Navegue até o diretório onde estão os arquivos que você baixou.
-3- Execute o aplicativo para os dados de treino, a fim de treinar o modelo.
-4- Execute o aplicativo com os dados de teste, para avaliar o modelo.
-5- Atingindo o nível de acurácia desejado, seu aplicativo analítico para classificação de Spam está pronto para receber novos conjuntos de dados e realizar a classificação do que é spam e do que não é.

# MODELO EXECUÇÃO:

$ python spam.py train bayes train/ output/resultado_treino.txt latin1



- python – nome do interpretador
- spam.py – nome do seu aplicativo Python (nome do script)
- train – tipo de operação do aplicativo, que será executado em modo treinamento
- bayes – nome do método de classificação, no caso Naive Bayes.
- train/ - nome do diretório onde estão os dados de treino
- output/resultado_treino.txt – arquivo onde será gravado o resultado do modelo de classificação treinado.
- latin1 - Encoding utilizando durante a execução.

$ python spam.py test bayes test/ output/resultado_teste.txt latin1

- python – nome do interpretador
- spam.py – nome do seu aplicativo Python (nome do script)
- test – tipo de operação do aplicativo, que será executado em modo treinamento
- bayes – nome do método de classificação, no caso Naive Bayes.
- test/ - nome do diretório onde estão os dados de treino
- output/resultado_teste.txt – arquivo onde será gravado o resultado das previsões do modelo.
- latin1 - Encoding utilizando durante a execução.


