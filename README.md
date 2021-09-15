# Kaggle_Disaster-Tweets
This is the code I used in my contribution to the Kaggle competition on Natural Language Processing with Disaster Tweets.

It is not the cleanest code I ever produced, nor is it excelling at what it does (in comparison with a range of contributions that exceeded this model's accuracy score), but it delivers some solid accuracy scores with rather simple means.

In short: I used two different approaches after vectorizing the disaster tweets in train.csv/text_col (with either a CountVectorizer or a TfidfVectorizer), that is a classic sentiment analysis using a multinomial naive bayes classifier as well as a deep learning approach with a Sequential model consisting of 7 layers, using Dropout and a sigmoid activation function in order to classify the outcome into binary values. The sentiment analysis consistently scored above 80% accuracy on my validation data (largely independent of any tweaks to the model, f.i. changing n-grams), whereas the deep learning model only got to about 75% accuracy. According to Kaggle, my best score using a Tfidf vectorizer and the naive bayes classifier was 0.79160.

Given the data's inherently terrible classification (no matter which tweet you check manually, there is a high chance it is wrongly classified), I don't see any reason to further improve my code as chances are high one would have to reclassify the whole dataset. Ignoring this issue would evidently create a model which is incapable of classifying actual real world data due to the aforementioned shortcomings. Tl;dr: This will do.
