# Fake-news-detection

## 1. Task definition
Fake news is a growing concern in our society. In a study led by NYU and Stanford [1], the majority of participants turned out to be really good at identifying “true news”. However,  when it came to inauthentic, or fake, news, even the most confident among them had difficulties spotting the signs and rejecting made-up facts, even when given the opportunity to fact-check them online.
We decided to study how machine learning can help better spot the signs of fake news, notably via an in-depth analysis and utilization of features extracted from news content. We designed various classifiers of news articles to help readers spot fake ones. As input, they take text from the article, and output a class (“fake” or “true”). We compared their results and sensitivity to features change, to bring to light some hidden characteristics of fake news content.


## 2. Literature review
Fake News classification has been a popular topic in the past few years, and two main approaches emerged, as referenced in [2]: either relying on news content (often used for traditional news), or on auxiliary information such as context (relevant for social media news).
For news content based approaches, features are generally extracted as linguistic-based and visual-based. Linguistic-based features capture specific characteristics of fake news based on document organizations from different levels, such as characters, words, sentences. They are typically categorized as lexical or syntactic features [3]. Another related approach, style-based, aims at capturing the manipulator's specific writing styles and sensational headlines that commonly occur in fake news content, such as deception [4] and non-objectivity [3].

As new computational algorithms have been improved, the models for fake news detection also improved. Beyond legacy linear classifiers, several research projects addressed the issue in recent years, often combining state-of-the-art NLP techniques [5] with Neural Networks [6]. These projects notably leverage the improvement introduced by word vectorization techniques [7] to extract meaningful representations of text content.
Within Stanford, a few AI projects took a stab at this topic as well. In particular, the Fake News Detection project [8] trained several Neural Networks (such as a DNN+BERT model) to classify news and tweets, and managed to obtain an Accuracy of 0.84 and a F1 Score of 0.89 on their Test set with their best model. Also, the DeepNewsNet project [9] trained and evaluated various NN (FC, LSTM, hybrid) on the LIAR dataset, and used Sentiment distribution predictions to improve their model. In both cases, the project focused on comparing models, not on analyzing which features impacted the most the prediction.
 
 
## 3. Dataset
We chose two existing datasets related to fake news, both from Kaggle: Fake vs. Real News Dataset [10] and FakeNewsNet Dataset [11].
This first dataset was collected from verified news articles. The truthful articles were obtained by crawling articles from Reuters.com. As for the fake news articles, they were collected from different sources, mostly unreliable websites flagged by Politifact. The dataset contains different types of articles; however, the majority focus on political and World news topics. 
The second dataset is a repository for an ongoing data collection project for fake news research at ASU. Like in the first dataset, fake news was collected from different websites flagged as unreliable, and most of the news are related to political and World news topics.
As “certified” fake news is a rare commodity, and we wanted to get as much training data as possible, we decided to merge them. 

![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/WordCloud_real.png?raw=true)\
Word Cloud of Real news

![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/WordCloud_fake.png?raw=true)\
Word Cloud of Fake news
Our first step was to preprocess the data. To start, we standardized our datasets by removing all other information (images, videos, etc.) besides the title and body text of the articles. We then merged each title with their body texts, converted the text to lowercase, tokenized and lemmatized our input, removed stop words, punctuation and special characters using NLTK library and custom routines. We also suppressed several words with high frequency (up to 15 most frequent words per class) and too easy to classify (urls, hashtags, references). 
We verified that this initial step worked properly by inspecting the most common unigrams and bigrams, as well as Five-number summary and linguistic structure of both real and fake corpus. We noticed that fake news was mostly longer than real ones, and used a larger vocabulary.

Initial distribution of number of words per news
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/Corpus_distribution.png?raw=true)\

Initial total unique words in:
   Real news: 67162
   Fake news: 88722

Common unique words: 39116


We also carried out topic exploration with pyLDAvis on an LDA model created from our tokens. We gained several insights into our dataset beyond just the frequency of words; by adjusting the relevance metric for our LDA model, we were able to see that our dataset falls into one of four news topic clusters: local government, police, and domestic violence; foreign policy and involvement in the Middle East; federal government, congress, and laws; and the 2016 election.
To further balance our dataset, and to avoid outliers, we reduced the maximum length of each article to 700 words (less than 3% of the dataset has more words).
After this step of data cleaning, we managed to get 22256 fake news and 21323 real ones. 
From this, we constructed our training, validation, and testing sets by taking a random sample of 70% (~30k examples), 15% (~7k examples), and 15% (~7k examples) of all articles, respectively.
 
 
## 4. Oracle and Baseline
We generated an original oracle for our future classifier by randomly sampling 25 articles and having both team members guess whether each was fake or real news. Each of us was accurate 84% of the time. If we combined our answers and picked the correct answer out of our guesses, we’d have a (maximum) accuracy of 88%; if we picked the wrong answer out of the two, we’d have an (minimum) accuracy of 80%. We then generated an expanded oracle, classifying 50 unique articles each. One of us got precision, recall, and F1 scores of 78%, and the other got a score of 94% for the three. On average, we got a weighted average score of 86% for the three. From these oracles, we can see that a human classifier receives precision, recall, and F1 scores of between 84% and 86%. Our results are below:
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/Oracle.png?raw=true)\
 
Now that we have an understanding of human performance on a small number of examples, we aim to establish a baseline for our classifier. To that extent, we selected 2 simple classifiers: multinomial Naïve Bayes and linear SVM.
In order to feed our inputs into untrained models, we needed to vectorize them. Word vectors are numerical representations of words, which became a critical component of NLP by their ability to extract meaningful characteristics of texts (such as word similarity, gender, frequency, etc.). The traditional classifiers work with word bags such as TF-IDF (term frequency-inverse document frequency) embeddings, which evaluates a word’s importance within a set of documents based on its frequency. This is done by multiplying two metrics: how many times a word appears in a specific document, and the inverse document frequency of the word across a set of documents. We implemented the two classifiers via the Scikit Learn library, with their standard setup options.
 
 
## 5. Main approach 
We developed a 3-phase approach:

A/ Preprocessing: beyond the basic preprocessing steps performed on the dataset (special characters removal, tokenization, lemmatization), we investigated more specific ones given the linguistic structure of our data. Namely, we inspected word-level and sentence-level features (total number of words, unique words, frequency, etc.), comparing both fake & real corpus, and adjusting the dataset accordingly to further balance it. 
We also studied two word vectorization techniques well-known in the field of NLP, namely TF-IDF (described in 4.) and Word2Vec. Developed at Google, Word2Vec uses a shallow neural network to learn word embeddings. The variant we used, the Continuous Bag of Words (CBOW) model, can be thought of as learning word embeddings of a specific word by looking at co-occurring words (its context). We implemented it via the Gensim library, with 100 dimensions vectors, a window of 5 words and a minimum frequency of 1, and trained it on our dataset.
 
B/ Modelization: beyond the two classifiers we trained as a baseline, we wanted to test state-of-the-art NLP models. To that extent we selected two additional models, which better understand nuances in context unlike aforementioned baselines: LSTM [12] and BERT [5]. 
For the LSTM, we used the simplest possible model: an embedding layer, one LSTM layer (128 units with dropout 0.2 to reduce overfitting, which significantly improved the training), one dense layer (100 units) and the output layer with sigmoid activation function for binary classification. When using external word-embeddings, the embedding layer is not trained. So we just had 130,249 parameters to train. We chose Adam optimizer and binary cross entropy loss function given the binary classification.
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/LSTM_archi.png?raw=true)\

BERT is a pre-trained model designed to learn deep bidirectional representations from unlabeled text by jointly conditioning on both left and right contexts in all layers. BERT uses WordPiece embeddings [13] with a 30.000 token vocabulary, to process and tokenize our cleaned dataset prior to training. It uses a Transformer, which learns contextual relations between words, consisting of an encoder and decoder. The encoder reads through the entire sequence of words at once (hence its “bidirectional” nature), and tries to accomplish two strategies in its training process. The first is “Masked LM,” which consists of removing 15% of the words in each article and training the model to try to predict the missing word. The second is Next Sentence Prediction, or NSP, which receives pairs of sentences and tries to determine if the second sentence is truly preceded by the first in any inputted news article. We directly control the last step of this model, “fine tuning”, in which we add an additional classification layer on top of the Transformer output [14]. We performed training with a batch size of 32 and a learning rate of 2e-5, and limited sequences to at most 128 tokens long due to memory restriction.

C/ Feature analysis: based on our trained models, we intended to investigate which features seemed to matter the most in identifying fake news. To that extent, we extracted the top features (eg words) used by our Naive Bayes model, and analyzed which ones impacted either Real or Fake prediction. We also computed a K-means of Word2Vec embeddings to generate clusters of words of each corpus, in order to get a representation of the bags of words created by the vectorization process and used for prediction by LSTM.
Finally, we performed sentiment analysis on our dataset, using two libraries based on a bag of words classifier: NLTK Vader, which leverages some heuristics to intensify or negate a sentence’s sentiment, and TextBlob, which also permits Subjectivity analysis (how factually based/opinionated a piece of text is).
 
 
## 6. Evaluation metric
During the whole project, we focused on Accuracy, Precision, Recall and F1 Score. Since our dataset was balanced between both classes, Accuracy was our metric of reference. It describes the similarity of articles we classified correctly to fake news. 
In addition, Precision measures the proportion of classified fake news that we got correct (true positives). However, because fake news datasets are usually skewed, high precision can be achieved by rarely making positive predictions. A metric that compliments this weakness is Recall, which measures the fraction of fake news that is predicted to be fake news. F1 combines precision and recall, providing an encompassing, descriptive metric.

 
## 7. Results & Analysis
Based on our initial preprocessing described in 3 and 5A, we observed the following results:
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/Initial_results.png?raw=true)\

These too high results showed that we needed to further preprocess the dataset, to remove patterns too easy to classify.
After a deeper analysis of the linguistic structure of our dataset, we decided to further balance it by keeping only news made of more than 100 words, and less than 450. We also removed all words not included in the common dictionary (almost cutting in 2 each respective dictionary, down to 30.744 unique words), as well as some words highly frequent in just one corpus. Finally, we balanced the final number of news in each corpus to be at parity: 13.632 each, which we splitted in train/validation/test sets with the same ratios as before.

After retraining with this updated preprocessing, the Naive Bayes baseline received an average accuracy of 91% on our test set. For SVM, the average accuracy reached 97%, still higher than our oracle but more reasonable than before dataset adjustments.
All evaluation metrics were globally similar for fake and real news classification on both models.
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/Baseline_results.png?raw=true)\


We then analyzed the top most important features of our Naive Bayes model, and compared them with the top frequent terms in each corpus of our preprocessed dataset:
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/NB_features.png?raw=true)\

Some terms appear twice when they are both used to predict fake and real news. If their corresponding odds are close, then it means the chances of predicting both are almost equal. 
Overall, the top features are consistent with the most frequent terms, which is logical given TF-IDF’s mechanism. It’s interesting to note that, as we reduced vocabulary to a common dictionary, many of the top features are shared by both classes. But several words remain unique top features for some classes (eg “candidate”, “day” or “asked” for Fake news, vs “called” or “end” for Real ones), and may explain the high level of accuracy of our baseline.

For the many-to-one LSTM neural net, an average accuracy and F1 score of 94% were achieved. 
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/LSTM_results.png?raw=true)\

Training BERT on our updated preprocessed dataset kept yielding an accuracy of 99.98%. 
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/Bert_results.png?raw=true)\

This high level of accuracy was confirmed as we tested BERT model’s ability to generalize on (new) fake and real news freshly generated from Politifact and Reuters: out of 8 examples scrapped from the websites (5 fake news, 3 real news), our BERT model made only 1 mistake (one fake news classified as real).

We were then curious to understand how Word2Vec, which creates embeddings of words based on their context (5 words around with our setup), would represent each corpus.

To do so, we computed a K-means of 10 clusters on the vectors generated by Word2Vec on each corpus. We used a KD Tree to get the 20 words closer to the centroids of our clusters, and displayed them in Word Clouds. Here are 4 of them for real news:
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/WordCloud_real_features.png?raw=true)\

And 4 of them for Fake news:
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/WordCloud_fake_features.png?raw=true)\

The content of these clusters was not exactly matching the most frequent bigrams and trigrams of each corpus, even if some patterns could be recognized (as can be seen in our notebook referenced at the end of this report). This may be explained by the size of the window we used for Word2Vec (5 words) and by the fact that the top features extracted by this embedding technique are not always the most frequent ones.
 
But the words highlighted there, and the tone employed, were pretty consistent with the major trends we obtained from sentiment analysis libraries Vader and TextBlob applied to each corpus of our dataset, as shown in the pictures below.
![alt text](https://github.com/d-roland/Fake-news-detection/blob/master/.ipynb_checkpoints/Sentiment_analysis.png?raw=true)\

The first graph (made with NLTK Vader) clearly shows that the content of Fake news is overall more negative than Real ones. The second (made with TextBlob) brings an additional light with a higher subjectivity in the content of Fake news. 
These observations aren’t really surprising: people writing fake news may be tempted to use more emotional vocabulary and tone, in order to reinforce the intensity of their claim and ultimately convince. To some extent, this could be perceived as manipulation.


## 8. Conclusion and Future Work
 
Our experiment was highly dependent on the specific datasets we managed to collect, and we made the choice to focus our work solely on the text content. In the real world, fake news takes many forms and, notably in social networks, requires analyzing both content, user and context to classify them correctly. To counteract this limitation, we implemented several, thorough methods of cleaning our dataset, from lemmatization to standardizing vocabulary.
 
However, despite those implicit biases in our datasets, our best networks were able to extract meaningful latent representations of news content and generate extremely accurate classifiers. We developed a three stage approach of processing, modelization, and analysis for Naive Bayes, SVM, LSTM, and BERT models. Almost all of our approaches achieved much higher accuracy, precision, recall, and F1 ratings than our Oracles. In order to probe why, we’ve analyzed our datasets through word clouds, LDA models, frequency charts, and sentiment analysis. This last technique illustrated perhaps the biggest discernible difference between fake and real news, showing a clear increase between the distributions of subjectivity ratings of  real and fake news. 
 
As possible next steps, we’d like to investigate how the sentiment and subjectivity features we managed to extract for each corpus could be used to improve the prediction further. 
We’d also be keen to integrate to our existing dataset real and fake news related to non political topics, and with additional components than pure content (eg user or context related). This may help improve the generalization of our models to broader types of fake news.
 
Our code is available on this Colab and this Github: 
https://github.com/d-roland/Fake-news-detection/blob/master/CS221_Fake_news_detection.ipynb 


## 9. References
[1] The uncomfortable truth about fake news - Financial Times / Feb 2020
[2] Fake News Detection on Social Media: A Data Mining Perspective - Kai Shu, Amy Sliva, Suhang Wang, Jiliang Tang, Huan Liun - arXiv:1708.01967 / Aug 2017
[3] A Stylometric Inquiry into Hyperpartisan and Fake News - Martin Potthast, Johannes Kiesel, Kevin Reinartz, Janek Bevendorff, and Benno Stein - arXiv:1702.05638 / 2017.
[4] Automatic deception detection: Methods for finding fake news. Proceedings of the Association for Information Science and Technology, 52(1):1–4 - Niall J Conroy, Victoria L Rubin, and Yimin Chen / 2015.
[5] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova - arXiv:1810.04805 / 2019.
[6] Detecting Fake News with Capsule Neural Networks - Mohammad Hadi Goldani, Saeedeh Momtazi, Reza Safabakhsh / Feb 2020
[7] Efficient Estimation of Word Representations in Vector Space - Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean - arXiv:1301.3781v3 / 2013
[8] Fake News Detection project - CS230 / Fall 2019
[9] DeepNewsNet project - CS230 / Winter 2019
[10] Fake vs. Real News Dataset - Kaggle - Detecting opinion spams and fake news using text
classification”, Journal of Security and Privacy, Volume 1, Issue 1, Wiley - Ahmed H, Traore I, Saad S / February 2018
[11] FakeNewsNet Dataset - Kaggle - A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media - Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan - arXiv:1809.01286 / 2018
[12] Long Short-term Memory - Sepp Hochreiter, Jurgen Schmidhuber - Neural Computation 1735-1780 / 1997
[13] Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation -  Wu & al. - arXiv:1609.08144 / 2016
[14] BERT Explained: State of the art language model for NLP - Rani Horev - Medium towards data science, 2018
