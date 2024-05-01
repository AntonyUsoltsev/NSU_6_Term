import spacy
import pytextrank

# Загрузка модели языка для spaCy
nlp = spacy.load("ru_core_news_sm")

# Добавляем кастомный pipeline к `spaCy`
nlp.add_pipe("textrank", last=True)


def textrank_pytextrank(text, num_sentences=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    summary = []
    for sent in doc._.textrank.summary(limit_sentences=num_sentences):
        summary.append(sent.text)
    return ' '.join(summary)


# Пример использования
with open("text.txt", 'r', encoding='utf-8') as file:
    text = file.read()
    summary_pytextrank = textrank_pytextrank(text)
    print(summary_pytextrank)
