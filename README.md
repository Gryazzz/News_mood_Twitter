
 **News sentiment analysis**

All comments are inside the document.
3 trends:
1. CNN sends tweets with more negative polarity than other sources. CBS is the source of the most positive tweets.
2. CNN and FoxNews both have the highest negative rates, but also the highest 'Like' rate, means that people like the way theese news sources deliver their opinions.
3. Daily tweet sentiment comparison shows that very often when CBS and BBC  have a positive tweet polarity, CNN, NPR and FoxNews are on the negative side. This research needs some more data.


```python
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from config import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import numpy as np
import seaborn as sns
from datetime import datetime

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Set the news sources list
targets = ['@BBC', '@CBS', '@CNN', '@FoxNews', '@nytimes', '@NPR']
```


```python
# Get 100 tweets data from each account

# Variable to store emotional lists from each source
total_mood = []

for target in targets:
    
    last_tweet = None
    
    tweet_counter = 0
    
    for x in range(5):
        
        all_data = api.user_timeline(target, count=20, max_id=last_tweet, page=x)
            
        for tweet in all_data:
            
            emotions = analyzer.polarity_scores(tweet['text'])
            
            total_mood.append({'source': target,
                             'compound': emotions['compound'],
                             'positive': emotions['pos'],
                             'negative': emotions['neg'],
                             'neutral': emotions['neu'],
                             'tweets_ago': tweet_counter,
                             'text': tweet['text'],
                             'time': tweet['created_at'],
                              'likes': tweet['favorite_count'],
                              'RT': tweet['retweet_count']})
            
            tweet_counter -= 1

        last_tweet = tweet["id"] - 1
    
len(total_mood)
```




    600




```python
# Create a DF from the received data
df = pd.DataFrame(total_mood)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RT</th>
      <th>compound</th>
      <th>likes</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>source</th>
      <th>text</th>
      <th>time</th>
      <th>tweets_ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>0.0000</td>
      <td>27</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>🌶🤯 A man who ate the world's hottest chilli pe...</td>
      <td>Tue Apr 10 16:58:05 +0000 2018</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>RT @bbccomedy: Henry of Eight, the Tudor Kim K...</td>
      <td>Tue Apr 10 16:34:54 +0000 2018</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.0000</td>
      <td>21</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>👭 Ten celebrity pairs who look so freakily ali...</td>
      <td>Tue Apr 10 16:32:08 +0000 2018</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>0.4227</td>
      <td>56</td>
      <td>0.150</td>
      <td>0.514</td>
      <td>0.336</td>
      <td>@BBC</td>
      <td>😱 That's quite the party trick! \n#Doodlebugs ...</td>
      <td>Tue Apr 10 16:02:03 +0000 2018</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>0.6705</td>
      <td>0</td>
      <td>0.066</td>
      <td>0.726</td>
      <td>0.208</td>
      <td>@BBC</td>
      <td>RT @TWBBC: "Our liberty is at risk when we han...</td>
      <td>Tue Apr 10 15:47:43 +0000 2018</td>
      <td>-4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Look at each news source separately
plt.rcParams.update(plt.rcParamsDefault) # Set default plot style
sentilist = ['compound', 'positive', 'negative']
current_date = datetime.now().date().strftime("%m.%d.%Y")
for i in range(len(sentilist)):
    sns.factorplot(data=df, x="tweets_ago", y=sentilist[i], col="source", hue='source')
    plt.title(f'Sentiment comparison by {sentilist[i]} ({current_date})')
    plt.savefig(f'Output/Sentiment_comparison_by_{sentilist[i]}_on_{current_date}.png')
```

**From theese graphs we can see, that CBS and BBC are generally more positive than the other sources. CNN on the other hand, is mostly on the negative side.**


```python
# Let's take a closer look on the distribution of the each sentiment for each user, using Bar charts.

plt.figure(figsize=(13,9))
for x in range(len(sentilist)):
    plt.subplot(2,2,x+1)
    ax = sns.barplot('source', sentilist[x], data=df, linewidth=1, edgecolor=".1")
    ax.set_title(f'Average {sentilist[x]} rate for each news source {current_date}')
    ax.set_xlabel('')
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2., p.get_height()*0.1, '{:1.3f}'.format(p.get_height()), ha="center")
        ax.set_xticklabels(ax.get_xticklabels(),rotation=15)
plt.savefig(f'Output/Sentiment_distribution_by_source_on_{current_date}.png')
```

**Our assumptions were confirmed: CBS is a leader in positive tweets creating, BBC is on the second place. NY Times and NPR are close to neutral overage score. CNN has the highest negative rate and FoxNews is the second major source of the negative tweets.**

**It is interesting to see which source has more likes and retweets.**


```python
newlist = ['likes', 'RT']
plt.figure(figsize=(10,5))
for x in range(len(newlist)):
    plt.subplot(1,2,x+1)
    ax = sns.barplot('source', newlist[x], data=df, linewidth=1, edgecolor=".1")
    ax.set_title(f'Number of {newlist[x]} for each news source')
    ax.set_xlabel('')
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2., 10, '{:1.0f}'.format(p.get_height()), ha="center", color='w', weight='bold')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.savefig(f'Output/Likes_and_RT_by_source.png')

```

**Interesting correlation - CNN and FoxNews with their negative tweets have significantly more likes, than positive CBS and BBC. Speaking about NPR, seems like it's negative polarity is much different from FoxNews's because people don't like it a lot. Also FoxNews and CNN are the sources with the highest retweet rate, what we can't say about NPR and CBS.**

**Below I want to compare the tweet creation date and the overal emotion rate during this date for each source**

It seems like some sources send their tweets more often than others, hence I need to adjust the tweets amount, to make the data compatible


```python
total_mood2 = []

for target in targets:
    
    #print(target)
    last_tweet = None
    
    tweet_counter = 0
    
    for x in range(10):
        
        if target == '@NPR':
            all_data = api.user_timeline(target, count=35, max_id=last_tweet, page=x)
        elif target == '@nytimes':
            all_data = api.user_timeline(target, count=50, max_id=last_tweet, page=x)
        elif target == '@CNN':
            all_data = api.user_timeline(target, count=70, max_id=last_tweet, page=x)
        elif target == '@FoxNews':
            all_data = api.user_timeline(target, count=130, max_id=last_tweet, page=x)
        elif target == '@BBC':
            all_data = api.user_timeline(target, count=10, max_id=last_tweet, page=x)
        else:
            all_data = api.user_timeline(target, count=2, max_id=last_tweet, page=x)
        #print(len(all_data))    
            
        for tweet in all_data:
            
            emotions = analyzer.polarity_scores(tweet['text'])
            
            total_mood2.append({'source': target,
                             'compound': emotions['compound'],
                             'positive': emotions['pos'],
                             'negative': emotions['neg'],
                             'neutral': emotions['neu'],
                             'tweets_ago': tweet_counter,
                             'text': tweet['text'],
                             'time': tweet['created_at'],
                              'likes': tweet['favorite_count'],
                              'RT': tweet['retweet_count']})
            
            tweet_counter -= 1

        last_tweet = tweet["id"] - 1
        
len(total_mood2)
```




    2560




```python
df2 = pd.DataFrame(total_mood2)
df2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RT</th>
      <th>compound</th>
      <th>likes</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>source</th>
      <th>text</th>
      <th>time</th>
      <th>tweets_ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.5106</td>
      <td>4</td>
      <td>0.216</td>
      <td>0.784</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>Meet Rebekah - a former professional footballe...</td>
      <td>Tue Apr 10 18:00:27 +0000 2018</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0.0000</td>
      <td>29</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>🌶🤯 A man who ate the world's hottest chilli pe...</td>
      <td>Tue Apr 10 16:58:05 +0000 2018</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>RT @bbccomedy: Henry of Eight, the Tudor Kim K...</td>
      <td>Tue Apr 10 16:34:54 +0000 2018</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.0000</td>
      <td>22</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>👭 Ten celebrity pairs who look so freakily ali...</td>
      <td>Tue Apr 10 16:32:08 +0000 2018</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>0.4227</td>
      <td>59</td>
      <td>0.150</td>
      <td>0.514</td>
      <td>0.336</td>
      <td>@BBC</td>
      <td>😱 That's quite the party trick! \n#Doodlebugs ...</td>
      <td>Tue Apr 10 16:02:03 +0000 2018</td>
      <td>-4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The code below converts twitter date format to the regular dates and adds it to a new column.
# Converting code was taken from StackOverflow and I haven't figured out how it works yet.

import re
datedf = df2.copy()
datedf['conv_time'] = ''
for ind, row in datedf.iterrows():
    twitter_time = row['time']
    remove_ms = lambda x:re.sub("\+\d+\s","",x) # some lambda magic
    mk_dt = lambda x:datetime.strptime(remove_ms(x), "%a %b %d %H:%M:%S %Y") # some lambda magic
    my_form = lambda x:"{:%m-%d-%y}".format(mk_dt(x)) # some lambda magic
    datedf.at[ind, 'conv_time'] = my_form(twitter_time)
datedf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RT</th>
      <th>compound</th>
      <th>likes</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>source</th>
      <th>text</th>
      <th>time</th>
      <th>tweets_ago</th>
      <th>conv_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.5106</td>
      <td>4</td>
      <td>0.216</td>
      <td>0.784</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>Meet Rebekah - a former professional footballe...</td>
      <td>Tue Apr 10 18:00:27 +0000 2018</td>
      <td>0</td>
      <td>04-10-18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0.0000</td>
      <td>29</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>🌶🤯 A man who ate the world's hottest chilli pe...</td>
      <td>Tue Apr 10 16:58:05 +0000 2018</td>
      <td>-1</td>
      <td>04-10-18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>RT @bbccomedy: Henry of Eight, the Tudor Kim K...</td>
      <td>Tue Apr 10 16:34:54 +0000 2018</td>
      <td>-2</td>
      <td>04-10-18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.0000</td>
      <td>22</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>👭 Ten celebrity pairs who look so freakily ali...</td>
      <td>Tue Apr 10 16:32:08 +0000 2018</td>
      <td>-3</td>
      <td>04-10-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>0.4227</td>
      <td>59</td>
      <td>0.150</td>
      <td>0.514</td>
      <td>0.336</td>
      <td>@BBC</td>
      <td>😱 That's quite the party trick! \n#Doodlebugs ...</td>
      <td>Tue Apr 10 16:02:03 +0000 2018</td>
      <td>-4</td>
      <td>04-10-18</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_date = datedf.groupby(['conv_time', 'source'])
grouped_df = grouped_date.mean().reset_index('source')
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>RT</th>
      <th>compound</th>
      <th>likes</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>tweets_ago</th>
    </tr>
    <tr>
      <th>conv_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>03-07-18</th>
      <td>@CBS</td>
      <td>35.000000</td>
      <td>-0.101150</td>
      <td>11.500000</td>
      <td>0.115000</td>
      <td>0.8095</td>
      <td>0.075000</td>
      <td>-18.5</td>
    </tr>
    <tr>
      <th>03-16-18</th>
      <td>@CBS</td>
      <td>24.000000</td>
      <td>0.325700</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.9110</td>
      <td>0.089000</td>
      <td>-16.5</td>
    </tr>
    <tr>
      <th>03-20-18</th>
      <td>@BBC</td>
      <td>45.400000</td>
      <td>0.120520</td>
      <td>71.200000</td>
      <td>0.043300</td>
      <td>0.8609</td>
      <td>0.095800</td>
      <td>-94.5</td>
    </tr>
    <tr>
      <th>03-20-18</th>
      <td>@CNN</td>
      <td>305.833333</td>
      <td>-0.191357</td>
      <td>542.666667</td>
      <td>0.112733</td>
      <td>0.8464</td>
      <td>0.040867</td>
      <td>-664.5</td>
    </tr>
    <tr>
      <th>03-21-18</th>
      <td>@CNN</td>
      <td>572.750000</td>
      <td>-0.137655</td>
      <td>961.850000</td>
      <td>0.082600</td>
      <td>0.8785</td>
      <td>0.038900</td>
      <td>-639.5</td>
    </tr>
  </tbody>
</table>
</div>



**CBS makes to much noise with it's Super positive tweets, so I exclude it from the data**


```python
nocbs = grouped_df.reset_index()
nocbs_df = nocbs[nocbs['source'] !='@CBS'].set_index('conv_time')
nocbs_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>RT</th>
      <th>compound</th>
      <th>likes</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>tweets_ago</th>
    </tr>
    <tr>
      <th>conv_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>03-20-18</th>
      <td>@BBC</td>
      <td>45.400000</td>
      <td>0.120520</td>
      <td>71.200000</td>
      <td>0.043300</td>
      <td>0.860900</td>
      <td>0.095800</td>
      <td>-94.5</td>
    </tr>
    <tr>
      <th>03-20-18</th>
      <td>@CNN</td>
      <td>305.833333</td>
      <td>-0.191357</td>
      <td>542.666667</td>
      <td>0.112733</td>
      <td>0.846400</td>
      <td>0.040867</td>
      <td>-664.5</td>
    </tr>
    <tr>
      <th>03-21-18</th>
      <td>@CNN</td>
      <td>572.750000</td>
      <td>-0.137655</td>
      <td>961.850000</td>
      <td>0.082600</td>
      <td>0.878500</td>
      <td>0.038900</td>
      <td>-639.5</td>
    </tr>
    <tr>
      <th>03-21-18</th>
      <td>@NPR</td>
      <td>67.057143</td>
      <td>-0.014871</td>
      <td>121.342857</td>
      <td>0.066429</td>
      <td>0.872429</td>
      <td>0.061171</td>
      <td>-332.0</td>
    </tr>
    <tr>
      <th>03-21-18</th>
      <td>@nytimes</td>
      <td>162.500000</td>
      <td>0.004614</td>
      <td>333.642857</td>
      <td>0.068643</td>
      <td>0.873000</td>
      <td>0.058214</td>
      <td>-492.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.rcParams.update(plt.rcParamsDefault)
plt.figure(figsize=(13,20))
for i in range(len(sentilist)):
    plt.subplot(4,1,i+1)
    ax = sns.barplot(x=nocbs_df.index, y=sentilist[i], hue='source', data=nocbs_df, ci=None)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
    ax.set_xlabel('')
    ax.set_title(f'{sentilist[i]} comparison over time on {current_date}')
    ax.grid(ls='dotted', linewidth=1.5)
plt.savefig(f'Output/Daily_sentiment_comparison_on_{current_date}.png')
```

**Generally we can see the same trends, that were noticed before: same day tweets from BBS are more positive comparing to CNN, FoxNews and NPR. Theese plots need more research. One way is to get a specific headline and compare the way it was interpreted by different sources.**


```python
# Colored dots, what can be better?
sns.set()
sns.lmplot(x='tweets_ago', y='compound', data=df, hue="source", fit_reg=False, scatter=True, size=10,
           scatter_kws={"s":85})
plt.ylabel('Tweet Polarity')
plt.title(f'Sentiment analysis of media tweets ({current_date})')
plt.savefig(f'Output/Sentiment_analysis_of_media_tweets({current_date}).png')
```
