# -- coding: utf-8 --
import streamlit as st
import pickle
import string
import nltk
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# Loading necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialization
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(words)

# Sentiment analysis function
def analyze_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)

# Main Streamlit interface
st.title("Защита цифровых двойников")

input_sms = st.text_area("Введите текст, который подозревается на мошеничество или буллинг")
input_sms = GoogleTranslator(source='ru', target='en').translate(input_sms)
if st.button('Анализировать'):
    # Предобработка
    transformed_sms = preprocess_text(input_sms)

    # Векторизация и предсказание
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Имитируем выполнение задачи
        progress.progress(i + 1)
        
    # Вывод результата предсказания
    if result == 0:
        st.header("Результат: Мошеничество")
        st.markdown("*Высокая вероятность фишинга*")
    else:
        st.header("Результат: Не мошеничество")
        st.markdown("*Низкая вероятность фишинга*")
        
        
    # Анализ настроений
    sentiment = analyze_sentiment(input_sms)

    # Вывод оценок настроения
    st.subheader(f"Общий словарь настроений: {sentiment}")
    st.subheader(f"Негатив: {sentiment['neg'] * 100:.2f}%")
    st.subheader(f"Нейтральность : {sentiment['neu'] * 100:.2f}%")
    st.subheader(f"Позитив: {sentiment['pos'] * 100:.2f}%")
    
        # Создание круговой диаграммы
    labels = ['Негативная часть', 'Нейтральная часть', 'Позитивная часть']
    sizes = [sentiment['neg'] * 100, sentiment['neu'] * 100, sentiment['pos'] * 100]
    colors = ['red', 'grey', 'green']
    explode = (0.2, 0.2, 0.2)  # выделение долей

    plt.figure(figsize=(6, 4))  # Установка размера фигуры
    plt.clf()  # Очистка текущей фигуры
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)
    plt.axis('equal')  # Убедитесь, что круговая диаграмма будет кругом.
    plt.title("Распределение настроений")  # Заголовок диаграммы

    # Отображение диаграммы на Streamlit
    st.pyplot(plt)
    

    # Определение общего настроения
    if sentiment['compound'] >= 0.05:
        overall_sentiment = "Позитивный"
    elif sentiment['compound'] <= -0.05:
        overall_sentiment = "Негативный"
    else:
        overall_sentiment = "Нейтральный"

    st.subheader(f"Текст в целом был оценен как: {overall_sentiment}")

    # Анализ полярности и субъективности
    blob = TextBlob(input_sms)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity * 100
    objective = 100 - subjectivity

    # Вывод полярности и субъективности
    if polarity < 0:
        sentiment_color = 'Негативный окрас'
    elif polarity == 0:
        sentiment_color = 'Нейтральный окрас'
    else:
        sentiment_color = 'Позитивный окрас'

    st.subheader(f'Полярность: {sentiment_color} ({polarity})')
    st.subheader(f'Степень субъективности: {subjectivity:.2f}%')
    st.subheader(f'Степень объективности: {objective:.2f}%')

    # Текст не токсичен и нет мошеничества
    if polarity > 0 and sentiment['compound'] >= 0.05 and result == 1:
        st.balloons()

