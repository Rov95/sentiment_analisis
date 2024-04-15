import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


plt.style.use('ggplot')

sia = SentimentIntensityAnalyzer()

path = r"C:\Users\andre\OneDrive\Escritorio\scripts\archive\Reviews.csv"

df = pd.read_csv(path)

scores = {}
total_rows = len(df)
for i, row in enumerate(df.itertuples(), 1):
    text = row.Text
    row_id = row.Id
    scores[row_id] = sia.polarity_scores(text)
    print(f"Processed {i} out of {total_rows} rows", end="\r")

vaders = pd.DataFrame(scores).T
vaders = vaders.reset_index().rename(columns={'index':'Id'})
vaders = vaders.merge(df, how='left')

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score By Amazon Star Review')


fig, axs = plt.subplots(1, 3, figsize=(15,5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')

plt.show()










