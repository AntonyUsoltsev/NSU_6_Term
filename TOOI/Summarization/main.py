import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import networkx as nx

# Загрузка стоп-слов и инициализация лемматизатора
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("russian"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    sentences = sent_tokenize(text)
    clean_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if
                 word.isalnum() and word.lower() not in stop_words]
        clean_sentences.append(words)
    return clean_sentences


def similarity(sentence1, sentence2):
    words1 = set(sentence1)
    words2 = set(sentence2)
    if len(words1.union(words2)) == 0:
        return 0
    return len(words1.intersection(words2)) / len(words1.union(words2))


def build_similarity_matrix(sentences):
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = similarity(sentences[i], sentences[j])
    return similarity_matrix


def textrank(text, num_sentences=5):
    # Предобработка текста
    clean_sentences = preprocess_text(text)

    # Построение матрицы сходства
    similarity_matrix = build_similarity_matrix(clean_sentences)

    # Создание графа на основе матрицы сходства и подсчет голосов с использованием page rank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # Выбор наиболее важных предложений
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(clean_sentences)), reverse=True)
    summary = [s for _, s in ranked_sentences[:num_sentences]]
    return '. '.join([' '.join(sentence) for sentence in summary])


with open("text.txt", 'r', encoding='utf-8') as file:
    textfile = file.read()
    summary = textrank(textfile, num_sentences=5)
    print(summary)
