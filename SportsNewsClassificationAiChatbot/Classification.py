import random
import string
import nltk
import numpy as np
import pandas as pd
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
stop_words = set(stopwords.words('english')) # Importing English language stop words

def main():
    # Opening text files
    soccer      = open('Soccer.txt', 'r')
    basketball  = open('Basketball.txt', 'r')
    tennis      = open('Tennis.txt', 'r')
    cricket     = open('Cricket.txt', 'r')
    formulaOne  = open('FormulaOne.txt', 'r')

    # Getting headlines
    soccerHeadlines     = []
    basketballHeadlines = []
    tennisHeadlines     = []
    cricketHeadlines    = []
    formulaOneHeadlines = []
    for i in soccer:
        soccerHeadlines.append(i.replace("\n",""))
    for i in basketball:
        basketballHeadlines.append(i.replace("\n",""))
    for i in tennis:
        tennisHeadlines.append(i.replace("\n",""))
    for i in cricket:
        cricketHeadlines.append(i.replace("\n",""))
    for i in formulaOne:
        formulaOneHeadlines.append(i.replace("\n",""))

    # Closing text files
    soccer.close()
    basketball.close()
    tennis.close()
    cricket.close()
    formulaOne.close()

    # Label all the chunks with the author name, add them all, and shuffle them
    labedledSoccer       = [(chunk, "Soccer") for chunk in soccerHeadlines]
    labedledBasketball       = [(chunk, "Basketball") for chunk in basketballHeadlines]
    labedledTennis       = [(chunk, "Tennis") for chunk in tennisHeadlines]
    labedledCricket      = [(chunk, "Cricket") for chunk in cricketHeadlines]
    labedledFormulaOne       = [(chunk, "FormulaOne") for chunk in formulaOneHeadlines]
    labeledHeadlines  = labedledSoccer + labedledBasketball + labedledTennis + labedledCricket + labedledFormulaOne
    random.shuffle(labeledHeadlines)

    # Importing our vectorizer, transformer, and classifiers
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    # Comment out starting from here for count vectorizer
    # Getting data ready for count vectorizer (list of sentences to be used in the count vectorizer) 
    dataFrameRepresentation         = pd.DataFrame(labeledHeadlines)
    dataFrameRepresentation.columns = ['text', 'authors']
    XTrainSet, XTestSet, yTrainSet, yTestSet = train_test_split(dataFrameRepresentation['text'],dataFrameRepresentation['authors'], test_size=0.5,random_state=0)

    # Building the pipelines for all the classifiers
    from sklearn.pipeline import Pipeline
    multinomialNBTextClf    = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])
    SGDTextClf              = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)), ])
    LRTextClf               = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression()), ])
    SVCTextClf              = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC()), ])
    DecisionTreeTextClf     = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', DecisionTreeClassifier()), ])
    KNeighborsTextClf       = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', KNeighborsClassifier()), ])
    # Comment out ending here for count vectorizer

    # Getting predictions ready for classification reports and confusion matrices
    multinomialNBTextClf.fit(XTrainSet, yTrainSet)
    SGDTextClf.fit(XTrainSet, yTrainSet)
    LRTextClf.fit(XTrainSet, yTrainSet)
    SVCTextClf.fit(XTrainSet, yTrainSet)
    DecisionTreeTextClf.fit(XTrainSet, yTrainSet)
    KNeighborsTextClf.fit(XTrainSet, yTrainSet)

    pickle.dump(SGDTextClf, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))


    while True:
        SelectedTestHeadline = str(input("Enter your headline: "))
        testHeadline = [SelectedTestHeadline]

        print('Results for myTestpklFile: ' + str(model.predict(testHeadline)))
        print('Results for MultinomialNB: '         +str(multinomialNBTextClf.predict(testHeadline)))
        print('Results for SGD: '                   +str(SGDTextClf.predict(testHeadline)))
        print('Results for Logistic Regression: '   +str(LRTextClf.predict(testHeadline)))
        print('Results for SVC: '                   +str(SVCTextClf.predict(testHeadline)))
        print('Results for Decision Tree: '         +str(DecisionTreeTextClf.predict(testHeadline)))
        print('Results for KNeighbors: '            +str(KNeighborsTextClf.predict(testHeadline)))

        def chooseToContinue():
            print("\n")
            headlineContinue = input("Do you want to classify another headline (y/n): ")

            if headlineContinue in ['n','y']:
                if headlineContinue == 'y':
                    return True
                else:
                    return False
            else:
                print("Please choose a valid option")
                return chooseToContinue()

        myFlag = chooseToContinue()
        if myFlag == True:
            continue
        else:
            break

if __name__ == '__main__':
    main()