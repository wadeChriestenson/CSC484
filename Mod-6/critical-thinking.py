# Web scraping with Requests, Beautiful Soup, stop word removal, and WordCloud

import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# Step 1: Download python.org contents
url = "https://www.python.org"
response = requests.get(url)

# Step 2: Parse HTML with Beautiful Soup
soup = BeautifulSoup(response.text, "html.parser")

# Step 3: Extract visible text
text = soup.get_text(separator=" ")

# Step 4: Clean the text
text = re.sub(r"[^A-Za-z\s]", "", text)
text = text.lower()

# Step 5: Tokenize words
words = word_tokenize(text)

# Step 6: Remove stop words
stop_words = set(stopwords.words("english"))

filtered_words = [
    word for word in words
    if word not in stop_words and len(word) > 2
]

# Step 7: Join words back into one string
clean_text = " ".join(filtered_words)

# Step 8: Create word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(clean_text)

# Step 9: Display word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud from Python.org")
plt.show()