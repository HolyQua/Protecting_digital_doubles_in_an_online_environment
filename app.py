# -- coding: windows-1251 --
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

# �������� ����������� ��������
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# �������������
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
sentiment_analyzer = SentimentIntensityAnalyzer()

# ������� ��� ������������� �����
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(words)

# ������� ��� ������� ����������
def analyze_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)

# �������� ��������� Streamlit
st.title("������ �������� ���������")

input_sms = st.text_area("������� �����, ������� ������������� �� ������������ ��� �������")
input_sms = GoogleTranslator(source='ru', target='en').translate(input_sms)
if st.button('�������������'):
    # �������������
    transformed_sms = preprocess_text(input_sms)

    # ������������ � ������������
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # ��������� ���������� ������
        progress.progress(i + 1)
        
    # ����� ���������� ������������
    if result == 0:
        st.header("���������: ������������")
        st.markdown("*������� ����������� �������*")
    else:
        st.header("���������: �� ������������")
        st.markdown("*������ ����������� �������*")
        
        
    # ������ ����������
    sentiment = analyze_sentiment(input_sms)

    # ����� ������ ����������
    st.subheader(f"����� ������� ����������: {sentiment}")
    st.subheader(f"�������: {sentiment['neg'] * 100:.2f}%")
    st.subheader(f"������������� : {sentiment['neu'] * 100:.2f}%")
    st.subheader(f"�������: {sentiment['pos'] * 100:.2f}%")
    
        # �������� �������� ���������
    labels = ['���������� �����', '����������� �����', '���������� �����']
    sizes = [sentiment['neg'] * 100, sentiment['neu'] * 100, sentiment['pos'] * 100]
    colors = ['red', 'grey', 'green']
    explode = (0.2, 0.2, 0.2)  # ��������� �����

    plt.figure(figsize=(6, 4))  # ��������� ������� ������
    plt.clf()  # ������� ������� ������
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)
    plt.axis('equal')  # ���������, ��� �������� ��������� ����� ������.
    plt.title("������������� ����������")  # ��������� ���������

    # ����������� ��������� �� Streamlit
    st.pyplot(plt)
    

    # ����������� ������ ����������
    if sentiment['compound'] >= 0.05:
        overall_sentiment = "����������"
    elif sentiment['compound'] <= -0.05:
        overall_sentiment = "����������"
    else:
        overall_sentiment = "�����������"

    st.subheader(f"����� � ����� ��� ������ ���: {overall_sentiment}")

    # ������ ���������� � ��������������
    blob = TextBlob(input_sms)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity * 100
    objective = 100 - subjectivity

    # ����� ���������� � ��������������
    if polarity < 0:
        sentiment_color = '���������� �����'
    elif polarity == 0:
        sentiment_color = '����������� �����'
    else:
        sentiment_color = '���������� �����'

    st.subheader(f'����������: {sentiment_color} ({polarity})')
    st.subheader(f'������� ��������������: {subjectivity:.2f}%')
    st.subheader(f'������� �������������: {objective:.2f}%')

    # ����� �� �������� � ��� ������������
    if polarity > 0 and sentiment['compound'] >= 0.05 and result == 1:
        st.balloons()

