# Classificador de Spam utilizando Naive Bayes

Mini Projeto 1 desenvolvido durante o módulo 16 do curso de Machine Learning da Formação Cientista de Dados da Data Science Academy (DSA).

O objetivo deste projeto é construir um classificador de Spam usando o algoritmo de classificação Naive Bayes. Construiremos esse algoritmo a partir do zero sem usar bibliotecas, o que será muito útil para construção de aplicações analíticas.

O modelo de documento que usaremos aqui é um modelo de saco de palavras (bag of words). Usaremos dois tipos de modelo bag of words:

A. Com base na presença de palavra (se uma palavra aparece no documento ou não, o que tornará os atributos de entrada binários)
B. Com base na frequência de palavras (frequência de ocorrência de palavra no documento, o que tornará os atributos de entrada contínuos).

Um saco de palavras (bag of words) é uma representação de um texto como um agrupamento de palavras, sem qualquer consideração da sua estrutura gramatical ou da ordem das palavras. É simplesmente um histograma sobre as palavras da língua, e cada documento é representado como um vetor sobre estas palavras. As entradas neste vetor simplesmente correspondem à presença ou à ausência da palavra correspondente (quando se utiliza o tipo A acima ou a frequência da ocorrência da palavra quando se usa o caso B acima).
