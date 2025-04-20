import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk

cleaned_data_path = 'cleaned_data_backup.csv'
data = pd.read_csv(cleaned_data_path)

print(data.head())
print(data.describe())
print(data.info())

data['sentiment'].value_counts().plot(kind = 'bar' , color = ['skyblue' , 'salmon'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation =0)
plt.show()

data['review_length'] = data['cleaned_review'].apply(len) 

data['review_length'].plot(kind = 'hist', bins = 50, color='lightblue', edgecolor='black')
plt.title('Review length Distribution')
plt.xlabel('Review length')
plt.ylabel('Frequency')
plt.show()

positive_reviews = ' '.join(data[data['sentiment']=='positive']['cleaned_review'] )
positive_wordcloud = WordCloud(width = 800, height = 400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10,5))
plt.imshow(positive_wordcloud, interpolation='bilinear' )
plt.axis('off')
plt.title('Wordcloud for positive reviews')
plt.show()

negative_reviews = ' '.join(data[data['sentiment']=='negative']['cleaned_review'] )
negative_wordcloud = WordCloud(width = 800, height = 400, background_color='white').generate(negative_reviews)
plt.figure(figsize=(10,5))
plt.imshow(negative_wordcloud, interpolation='bilinear' )
plt.axis('off')
plt.title('Wordcloud for negative reviews')
plt.show()

positive_words = ' '.join(data[data['sentiment'] == 'positive']['cleaned_review']).split()
negative_words = ' '.join(data[data['sentiment'] == 'negative']['cleaned_review']).split()

positive_word_freq = Counter(positive_words)
negative_word_freq = Counter(negative_words)

print("Most common words in positive reviews:")
print(positive_word_freq.most_common(10))

print("Most common words in negative reviews:")
print(negative_word_freq.most_common(10))



