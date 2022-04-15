import random
import string
import nltk
import re
import numpy as np
import json
import pandas as pd
import nltk.corpus
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import cluster
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) # Importing English language stop words

# removes a list of words (ie. stopwords) from a tokenized list.
def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

# applies stemming to a list of tokenized words
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

# removes any words composed of less than 2 or more than 21 letters
def twoLetters(listOfTokens):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= 3 or len(token) >= 21:
            twoLetterWord.append(token)
    return twoLetterWord

# Function that will clean the text, uses removeWords, applyStemming, and twoLetters
def processCorpus(corpus, language):
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = SnowballStemmer(language)
    # countries_list = [line.rstrip('\n') for line in open('lists/countries.txt')] # Load .txt file line by line
    # nationalities_list = [line.rstrip('\n') for line in open('lists/nationalities.txt')] # Load .txt file line by line
    other_words = ['from', 'that', 'with', 'what', 'use', 'like', 'said', 'one', 'two', 'three', 'much', 'must', 'yet',
                   'first', 'see', 'they', 'you', 'second', 'third', 'could', 'would', 'us', 'they', 'go', 'went',
                   'them', 'oh', 'let', 'ah', 'sure', 'be', 'may', 'go', 'be', 'see', 'put', 'say', 'syme', 'think',
                   'come', 'give', 'way', 'shall', 'could', 'ahab', 'loue', 'went', 'quit', 'know', 'think', 'good',
                   'see', 'let', 'must', 'like', 'ask', 'let', 'thi', 'would', 'upon', 'must', 'enter', 'self', 'head',
                   'get', 'time', 'put', 'much', 'say', 'talk']

    for document in corpus:
        index = corpus.index(document)
        corpus[index] = corpus[index].replace(u'\ufffd', '8')  # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace(',', '')  # Removes commas
        corpus[index] = corpus[index].rstrip('\n')  # Removes line breaks
        corpus[index] = corpus[index].casefold()  # Makes all letters lowercase

        corpus[index] = re.sub('\W_', ' ', corpus[index])  # removes specials characters and leaves only words
        corpus[index] = re.sub("\S*\d\S*", " ", corpus[
            index])  # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub("\S*@\S*\s?", " ", corpus[index])  # removes emails and mentions (words with @)
        corpus[index] = re.sub(r'http\S+', '', corpus[index])  # removes URLs with http
        corpus[index] = re.sub(r'www\S+', '', corpus[index])  # removes URLs with www

        listOfTokens = word_tokenize(corpus[index])
        twoLetterWord = twoLetters(listOfTokens)

        listOfTokens = removeWords(listOfTokens, stopwords)
        listOfTokens = removeWords(listOfTokens, twoLetterWord)
        # listOfTokens = removeWords(listOfTokens, countries_list)
        # listOfTokens = removeWords(listOfTokens, nationalities_list)
        listOfTokens = removeWords(listOfTokens, other_words)

        listOfTokens = applyStemming(listOfTokens, param_stemmer)
        listOfTokens = removeWords(listOfTokens, other_words)
        # listOfTokens = twoLetters(listOfTokens)

        corpus[index] = " ".join(listOfTokens)
        corpus[index] = unidecode(corpus[index])

    return corpus

# This function is used to plot the Silhouette
def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(8, 6)
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

    ax1.axvline(x=silhouette_avg, color="red",
                linestyle="--")  # The vertical line for average silhouette score of all the values
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')

    y_lower = 10
    sample_silhouette_values = silhouette_samples(df, kmeans_labels)  # Compute the silhouette scores for each sample
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,
                 str(i))  # Label the silhouette plots with their cluster numbers at the middle
        y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
    plt.show()

# This function plots the Silhouette
def silhouette(kmeans_dict, clusterType,df,plot=False):
    df = df.to_numpy()
    avg_dict = dict()
    if clusterType == "km":
        labels = kmeans_dict.predict(df)
    elif clusterType == "ag":
        labels = kmeans_dict.fit_predict(df)
    elif clusterType == "em":
        labels = kmeans_dict.predict(df)
    else:
        return
    silhouette_avg = silhouette_score(df, labels)  # Average Score for all Samples
    avg_dict.update({silhouette_avg: 5})
    if (plot): plotSilhouette(df, 5, labels, silhouette_avg)

# Used by plotWords to plot the top used words
def get_top_features_cluster(tf_idf_array, prediction, n_feats, vectorizer):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

# Plots the top used words
def plotWords(dfs, n_feats):
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.show()

def main():
    # Opening text files
    soccer = open('Soccer.txt', 'r')
    basketball = open('Basketball.txt', 'r')
    tennis = open('Tennis.txt', 'r')
    cricket = open('Cricket.txt', 'r')
    formulaOne = open('FormulaOne.txt', 'r')

    # Getting headlines
    soccerHeadlines = []
    basketballHeadlines = []
    tennisHeadlines = []
    cricketHeadlines = []
    formulaOneHeadlines = []
    for i in soccer:
        soccerHeadlines.append(i.replace("\n", ""))
    for i in basketball:
        basketballHeadlines.append(i.replace("\n", ""))
    for i in tennis:
        tennisHeadlines.append(i.replace("\n", ""))
    for i in cricket:
        cricketHeadlines.append(i.replace("\n", ""))
    for i in formulaOne:
        formulaOneHeadlines.append(i.replace("\n", ""))

    # Closing text files
    soccer.close()
    basketball.close()
    tennis.close()
    cricket.close()
    formulaOne.close()

    # Label all the chunks with the author name, add them all, and shuffle them
    labedledSoccer = [(chunk, "Soccer") for chunk in soccerHeadlines]
    labedledBasketball = [(chunk, "Basketball") for chunk in basketballHeadlines]
    labedledTennis = [(chunk, "Tennis") for chunk in tennisHeadlines]
    labedledCricket = [(chunk, "Cricket") for chunk in cricketHeadlines]
    labedledFormulaOne = [(chunk, "FormulaOne") for chunk in formulaOneHeadlines]
    labeledHeadlines = labedledSoccer + labedledBasketball + labedledTennis + labedledCricket + labedledFormulaOne
    random.shuffle(labeledHeadlines)

    # Getting data ready for count vectorizer (list of sentences to be used in the count vectorizer)
    dataFrameRepresentation                     = pd.DataFrame(labeledHeadlines)
    dataFrameRepresentation.columns             = ['text', 'author']

    # Use processCorpus function to further clean the data
    language = 'english'
    dataFrameRepresentation['text'] = processCorpus(dataFrameRepresentation['text'].tolist(), language)

    # Use count vectorizer to get array of all the feature counts (BOG)
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer  = CountVectorizer()
    X           = vectorizer.fit_transform(dataFrameRepresentation['text'][:9000])
    data        = X.toarray() #Features array format after count vectorizer

    # Use Tfidf transformer to get array of feature weighted
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False)
    data        = transformer.fit_transform(data)
    data        = data.toarray()

    # Loop to label all columns which their feature name
    FeaturesDf  = pd.DataFrame(data)
    featList    = []
    for i in range(len(data[0])):
        featList.append('Feat' + str(i))
    FeaturesDf.columns = featList

    # Choose number of clusters
    chosenClusterCount = 5
    ###################################################################################################################
    #Start: Comment this part if you don't want Kmeans
    # This is to check elbow method to check suitable # of clusters, but we know it's 5 authors, so 5 clusters
    # Check which value of clusters we should have, check plot and decide elbow point
    from sklearn.cluster import KMeans
    wcss = []
    clusterCount = 11
    for i in range(1, clusterCount):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(FeaturesDf[featList])
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, clusterCount), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # When we have our elbow point (we choose 5 because we have 5 authors), set your desired number of clusters. Add cluster to each sample
    kmeans                      = KMeans(n_clusters=chosenClusterCount, init='k-means++', max_iter=300, n_init=10, random_state=0)
    predictedCluster            = kmeans.fit_predict(FeaturesDf[featList])
    FeaturesDf['Cluster']       = predictedCluster
    myType                      = kmeans
    clusterType                 = "km"
    #End: Comment this part if you don't want Kmeans
    ###################################################################################################################
    '''#Start: Comment this part if you don't want AgglomerativeClustering
    # This is the dendrogram method to check suitable # of clusters, but we know it's 5 authors, so 5 clusters
    import scipy.cluster.hierarchy as sch
    dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
    plt.show()

    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters=chosenClusterCount, affinity='euclidean', linkage='ward')
    predictedCluster = hc.fit_predict(FeaturesDf[featList])
    FeaturesDf['Cluster']   = predictedCluster
    myType                  = hc
    clusterType             = "ag"
    '''#End: Comment this part if you don't want AgglomerativeClustering
    ###################################################################################################################
    '''#Start: Comment this part if you don't want GaussianMixture
    from sklearn.mixture import GaussianMixture
    n_components = np.arange(1, 3)
    models = [GaussianMixture(n).fit(FeaturesDf[featList]) for n in n_components]
    plt.plot(n_components, [m.bic(FeaturesDf[featList]) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(FeaturesDf[featList]) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    plt.show()

    gmm = GaussianMixture(n_components=chosenClusterCount, random_state=0, covariance_type='full')
    gmm.fit(FeaturesDf[featList])
    predictedCluster = gmm.predict(FeaturesDf[featList])
    FeaturesDf['Cluster'] = predictedCluster
    myType                = gmm
    clusterType           = "em"
    '''#End: Comment this part if you don't want GaussianMixture
    ###################################################################################################################
    '''# Start: Comment this part if you don't want LDA
    from gensim import corpora
    documents = dataFrameRepresentation['text']

    # remove common words and tokenize
    texts = []
    for document in documents:
        texts.append(document.split())

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    from gensim import models
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=5)
    corpus_lda = tfidf[corpus_tfidf]

    lda_model.print_topics()

    def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus_lda, texts=documents)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    predictedCluster = []
    for index, row in df_dominant_topic.iterrows():
        predictedCluster.append(row['Dominant_Topic'])

    FeaturesDf['Cluster'] = predictedCluster
    '''# End: Comment this part if you don't want LDA
    ###################################################################################################################

    # Add a column to the dataframe with the actual author names (ones we labled manually)
    FeaturesDf['actualAuthorName'] = dataFrameRepresentation['author']

    # List of dictionaries to see which cluster (most likely) represents which author
    # In order: Soccer, Basketball, Tennis, Cricket, FormulaOne
    clusterToAuthor = [{"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}, {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
                       {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}, {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
                       {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}]

    # For each sample, compare cluster value and author.
    for index, row in FeaturesDf.iterrows():
        if row['actualAuthorName'] == "Soccer":
            if row["Cluster"] == 0.0:
                clusterToAuthor[0]["0"] = clusterToAuthor[0]["0"] + 1
            elif row["Cluster"] == 1.0:
                clusterToAuthor[0]["1"] = clusterToAuthor[0]["1"] + 1
            elif row["Cluster"] == 2.0:
                clusterToAuthor[0]["2"] = clusterToAuthor[0]["2"] + 1
            elif row["Cluster"] == 3.0:
                clusterToAuthor[0]["3"] = clusterToAuthor[0]["3"] + 1
            elif row["Cluster"] == 4.0:
                clusterToAuthor[0]["4"] = clusterToAuthor[0]["4"] + 1
        if row['actualAuthorName'] == "Basketball":
            if row["Cluster"] == 0.0:
                clusterToAuthor[1]["0"] = clusterToAuthor[1]["0"] + 1
            elif row["Cluster"] == 1.0:
                clusterToAuthor[1]["1"] = clusterToAuthor[1]["1"] + 1
            elif row["Cluster"] == 2.0:
                clusterToAuthor[1]["2"] = clusterToAuthor[1]["2"] + 1
            elif row["Cluster"] == 3.0:
                clusterToAuthor[1]["3"] = clusterToAuthor[1]["3"] + 1
            elif row["Cluster"] == 4.0:
                clusterToAuthor[1]["4"] = clusterToAuthor[1]["4"] + 1
        if row['actualAuthorName'] == "Tennis":
            if row["Cluster"] == 0.0:
                clusterToAuthor[2]["0"] = clusterToAuthor[2]["0"] + 1
            elif row["Cluster"] == 1.0:
                clusterToAuthor[2]["1"] = clusterToAuthor[2]["1"] + 1
            elif row["Cluster"] == 2.0:
                clusterToAuthor[2]["2"] = clusterToAuthor[2]["2"] + 1
            elif row["Cluster"] == 3.0:
                clusterToAuthor[2]["3"] = clusterToAuthor[2]["3"] + 1
            elif row["Cluster"] == 4.0:
                clusterToAuthor[2]["4"] = clusterToAuthor[2]["4"] + 1
        if row['actualAuthorName'] == "Cricket":
            if row["Cluster"] == 0.0:
                clusterToAuthor[3]["0"] = clusterToAuthor[3]["0"] + 1
            elif row["Cluster"] == 1.0:
                clusterToAuthor[3]["1"] = clusterToAuthor[3]["1"] + 1
            elif row["Cluster"] == 2.0:
                clusterToAuthor[3]["2"] = clusterToAuthor[3]["2"] + 1
            elif row["Cluster"] == 3.0:
                clusterToAuthor[3]["3"] = clusterToAuthor[3]["3"] + 1
            elif row["Cluster"] == 4.0:
                clusterToAuthor[3]["4"] = clusterToAuthor[3]["4"] + 1
        if row['actualAuthorName'] == "FormulaOne":
            if row["Cluster"] == 0.0:
                clusterToAuthor[4]["0"] = clusterToAuthor[4]["0"] + 1
            elif row["Cluster"] == 1.0:
                clusterToAuthor[4]["1"] = clusterToAuthor[4]["1"] + 1
            elif row["Cluster"] == 2.0:
                clusterToAuthor[4]["2"] = clusterToAuthor[4]["2"] + 1
            elif row["Cluster"] == 3.0:
                clusterToAuthor[4]["3"] = clusterToAuthor[4]["3"] + 1
            elif row["Cluster"] == 4.0:
                clusterToAuthor[4]["4"] = clusterToAuthor[4]["4"] + 1

    # Assigning the clusters to each author
    author0Cluster = int(max(clusterToAuthor[0], key=clusterToAuthor[0].get))
    author1Cluster = int(max(clusterToAuthor[1], key=clusterToAuthor[1].get))
    author2Cluster = int(max(clusterToAuthor[2], key=clusterToAuthor[2].get))
    author3Cluster = int(max(clusterToAuthor[3], key=clusterToAuthor[3].get))
    author4Cluster = int(max(clusterToAuthor[4], key=clusterToAuthor[4].get))
    authorCluster  = [author0Cluster,author1Cluster,author2Cluster,author3Cluster,author4Cluster]

    # Function that will return the cluster and index
    def clusterCounter(authorCluster,cluster):
        clusterCountList = []
        for i in range(len(authorCluster)):
            if authorCluster[i] == cluster:
                clusterCountList.append(i)
        return clusterCountList

    # Finding for each cluster, the count of it and at which index (author)
    cluster0count = clusterCounter(authorCluster,0)
    cluster1count = clusterCounter(authorCluster,1)
    cluster2count = clusterCounter(authorCluster,2)
    cluster3count = clusterCounter(authorCluster,3)
    cluster4count = clusterCounter(authorCluster,4)
    clusterCount  = [cluster0count, cluster1count, cluster2count, cluster3count, cluster4count]

    # If a cluster is repeated, distribute the missing one randomly
    for item in clusterCount:
        if item == []:
            for item2 in clusterCount:
                if len(item2) > 1:
                    replaceItem = item2.pop()
                    item.append(replaceItem)

    if clusterCount[0][0] == 0:
        author0Cluster = 0
    elif clusterCount[0][0] == 1:
        author1Cluster = 0
    elif clusterCount[0][0] == 2:
        author2Cluster = 0
    elif clusterCount[0][0] == 3:
        author3Cluster = 0
    elif clusterCount[0][0] == 4:
        author4Cluster = 0

    if clusterCount[1][0] == 0:
        author0Cluster = 1
    elif clusterCount[1][0] == 1:
        author1Cluster = 1
    elif clusterCount[1][0] == 2:
        author2Cluster = 1
    elif clusterCount[1][0] == 3:
        author3Cluster = 1
    elif clusterCount[1][0] == 4:
        author4Cluster = 1

    if clusterCount[2][0] == 0:
        author0Cluster = 2
    elif clusterCount[2][0] == 1:
        author1Cluster = 2
    elif clusterCount[2][0] == 2:
        author2Cluster = 2
    elif clusterCount[2][0] == 3:
        author3Cluster = 2
    elif clusterCount[2][0] == 4:
        author4Cluster = 2

    if clusterCount[3][0] == 0:
        author0Cluster = 3
    elif clusterCount[3][0] == 1:
        author1Cluster = 3
    elif clusterCount[3][0] == 2:
        author2Cluster = 3
    elif clusterCount[3][0] == 3:
        author3Cluster = 3
    elif clusterCount[3][0] == 4:
        author4Cluster = 3

    if clusterCount[4][0] == 0:
        author0Cluster = 4
    elif clusterCount[4][0] == 1:
        author1Cluster = 4
    elif clusterCount[4][0] == 2:
        author2Cluster = 4
    elif clusterCount[4][0] == 3:
        author3Cluster = 4
    elif clusterCount[4][0] == 4:
        author4Cluster = 4

    print("Soccer is cluster: "         +str(author0Cluster))
    print("Basketball is cluster: "       +str(author1Cluster))
    print("Tennis is cluster: "       +str(author2Cluster))
    print("Cricket is cluster: "     +str(author3Cluster))
    print("Formula One is cluster: " +str(author4Cluster))

    # Using the clusters above, add a column with all the predicted author names
    predictedAuthorName = []
    for index, row in FeaturesDf.iterrows():
            if round(int(row['Cluster'])) == round(author0Cluster):
                predictedAuthorName.append("Soccer")
            elif round(int(row['Cluster'])) == round(author1Cluster):
                predictedAuthorName.append("Basketball")
            elif round(int(row['Cluster'])) == round(author2Cluster):
                predictedAuthorName.append("Tennis")
            elif round(int(row['Cluster'])) == round(author3Cluster):
                predictedAuthorName.append("Cricket")
            elif round(int(row['Cluster'])) == round(author4Cluster):
                predictedAuthorName.append("FormulaOne")
    FeaturesDf["predictedAuthorName"] = predictedAuthorName

    # Checking how many we got right by comparing the actual author name to the predicted
    correct = 0
    incorrect = 0
    for index, row in FeaturesDf.iterrows():
        if row['predictedAuthorName'] == row['actualAuthorName']:
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    print("Out of 9,000 samples, the correct labeling count is: "  +str(correct))
    print("Out of 9,000 samples, the incorrect labeling count is: "+str(incorrect))

    # Plotting all the samples and the clusters
    # Apply the Principal of Component Analysis to reduce the space in 2 columns (each column having half the features) and visualize this instead
    import seaborn as sns
    from sklearn.decomposition import PCA
    # Reducing sample data and plotting
    reducedDataSamples = PCA(n_components=(2)).fit_transform(FeaturesDf[featList])
    results = pd.DataFrame(reducedDataSamples, columns=['One Half of the features', 'Other half of the features'])
    sns.scatterplot(x="One Half of the features", y="Other half of the features", hue=FeaturesDf['Cluster'],
                    data=results)

    ###################################################################################################################
    #Start: Comment this part if you don't want Kmeans
    # Reducing cluster data and plotting
    reducedDataClusters = PCA(n_components=(2)).fit_transform(kmeans.cluster_centers_)
    cluster0 = plt.scatter(reducedDataClusters[0][0], reducedDataClusters[0][1], s=300, c='red')
    cluster1 = plt.scatter(reducedDataClusters[1][0], reducedDataClusters[1][1], s=300, c='blue')
    cluster2 = plt.scatter(reducedDataClusters[2][0], reducedDataClusters[2][1], s=300, c='green')
    cluster3 = plt.scatter(reducedDataClusters[3][0], reducedDataClusters[3][1], s=300, c='orange')
    cluster4 = plt.scatter(reducedDataClusters[4][0], reducedDataClusters[4][1], s=300, c='yellow')
    plt.show()
    #End: Comment this part if you don't want Kmeans
    ###################################################################################################################
    '''#Start: Comment this part if you don't want AgglomerativeClustering
    plt.show()
    cluster0 = plt.scatter(reducedDataSamples[predictedCluster == 0, 0], reducedDataSamples[predictedCluster == 0, 1], s=100, c='red')
    cluster1 = plt.scatter(reducedDataSamples[predictedCluster == 1, 0], reducedDataSamples[predictedCluster == 1, 1], s=100, c='black')
    cluster2 = plt.scatter(reducedDataSamples[predictedCluster == 2, 0], reducedDataSamples[predictedCluster == 2, 1], s=100, c='blue')
    cluster3 = plt.scatter(reducedDataSamples[predictedCluster == 3, 0], reducedDataSamples[predictedCluster == 3, 1], s=100, c='cyan')
    cluster4 = plt.scatter(reducedDataSamples[predictedCluster == 4, 0], reducedDataSamples[predictedCluster == 4, 1], s=100, c='yellow')
    plt.legend((cluster0, cluster1, cluster2, cluster3, cluster4),
               ('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.show()
    '''#End: Comment this part if you don't want AgglomerativeClustering
    ###################################################################################################################

    # Plotting Silhouette
    silhouette(myType, clusterType,FeaturesDf[featList], plot=True)

    # Showing most common words
    FeaturesDf_array = FeaturesDf[featList].to_numpy()
    n_feats          = 20
    dfs              = get_top_features_cluster(FeaturesDf_array, FeaturesDf['Cluster'], n_feats, vectorizer)
    plotWords(dfs, 13)

    #Calculating the Kappa
    actualAuthors       = FeaturesDf['actualAuthorName']
    predictedAuthors    = FeaturesDf["predictedAuthorName"]
    from sklearn.metrics import cohen_kappa_score
    kappaScore = cohen_kappa_score(actualAuthors, predictedAuthors)
    print("The Kappa score is: "+str(kappaScore))

if __name__ == '__main__':
    main()