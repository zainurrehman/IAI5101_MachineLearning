# import dependencies for flask, web scrapping, loading the trained ML model.
from flask import Flask, request
import pickle
import numpy as np
import re
import json
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

# initialize the flask app
app = Flask(__name__)
# Load the model.pkl file. Note that app.py and model.pkl should be in the same directory.
# The is the model file (classification SGD model) based on Classification.py
# Note that the model was trained on March 26, 2022.
model = pickle.load(open('model.pkl', 'rb'))

# Function to get the latest headlines. 
# The function is called from the webhook function to return the latest headlines to the chatbot.
# The function is derived from the python code present in WebScrapping.py
def getHeadlines(sport):
    numOfPage = 1
    myUrls = []
    for i in range(numOfPage):
        if sport in ['Basketball', 'basketball']:
            tempUrl = "https://www.reuters.com/news/archive/basketball-nba?view=page&page=" + str(
                i + 1) + "&pageSize=10"
            myUrls.append(tempUrl)
        elif sport in ['Cricket', 'cricket']:
            tempUrl = "https://www.espncricinfo.com/latest-cricket-news?page=" + str(i + 1)
            myUrls.append(tempUrl)
        elif sport in ['Soccer', 'soccer']:
            tempUrl = "https://www.reuters.com/news/archive/soccer-england?view=page&page=" + str(
                i + 1) + "&pageSize=10"
            myUrls.append(tempUrl)
        elif sport in ['F1', 'Formula One', 'f1', 'FormulaOne', 'formula one', 'Formula one']:
            tempUrl = "https://www.reuters.com/news/archive/sport-f1?view=page&page=" + str(i + 1) + "&pageSize=10"
            myUrls.append(tempUrl)
        elif sport in ['Tennis', 'tennis']:
            tempUrl = "https://www.reuters.com/news/archive/sport-tennis?view=page&page=" + str(i + 1) + "&pageSize=10"
            myUrls.append(tempUrl)

    # Opening the URLs, downloading the HTML pages, and closing the connection
    myHtmlPages = []
    for i in range(numOfPage):
        uClient = uReq(myUrls[i])
        pageHtml = uClient.read()
        uClient.close()
        myHtmlPages.append(pageHtml)

    # Parsing the HTML pages
    myParsedPages = []
    for i in range(numOfPage):
        pageSoup = soup(myHtmlPages[i], "html.parser")
        myParsedPages.append(pageSoup)

    # Grabbing all the headline items in all the pages
    myHeadlines = []
    for i in range(numOfPage):
        if sport in ['Cricket', 'cricket']:
            headlines = myParsedPages[i].findAll("h1", {"class": "ds-text-title-s ds-font-bold ds-text-typo-title ds-mb-1"})
            myHeadlines.append(headlines)
        else:
            headlines = myParsedPages[i].findAll("h3", {"class": "story-title"})
            myHeadlines.append(headlines)

    # Saving the string format of each headline for each page
    myStringHeadlines = []
    for headlines in myHeadlines:
        for i in range(5):
            if sport in ['Cricket', 'cricket']:
                myStringHeadlines.append(
                    str(headlines[i]).replace('<span class="story-title h3"><span>', '').replace('</span></span>',
                                                                                                 '').replace(
                        '</span></span>,', ''))
            else:
                myStringHeadlines.append(
                    str(headlines[i]).replace('<h3 class="story-title">', '').replace('\n', '').replace(
                        '								', '').replace('</h3>', ''))

    theURL = myUrls[0]
    myStringHeadlines.append("Please refer to the link for more details: "+theURL)
    return myStringHeadlines

# Rules and history variables
# The rules and histroy is returned to the chatbot when the History and rules intentes are invoked.
basketball_rules="----- Read more: https://en.wikipedia.org/wiki/Rules_of_basketball ----- Rules of basketball,From Wikipedia: The rules of basketball are the rules and regulations that govern the play, officiating, equipment and procedures of basketball. While many of the basic rules are uniform throughout the world, variations do exist. Most leagues or governing bodies in North America, the most important of which are the National Basketball Association and NCAA, formulate their own rules. In addition, the Technical Commission of the International Basketball Federation (FIBA) determines rules for international play; most leagues outside North America use the complete FIBA ruleset."
basketball_history="-----Read more: https://en.wikipedia.org/wiki/History_of_basketball -----History of basketball, From Wikipedia: The history of basketball began with its invention in 1891 in Springfield, Massachusetts by Canadian physical education instructor James Naismith as a less injury-prone sport than football. Naismith was a 31-year old graduate student when he created the indoor sport to keep athletes indoors during the winters.[1] The game became established fairly quickly and grew very popular as the 20th century progressed, first in America and then in other parts of the world. After basketball became established in American colleges, the professional game followed. The American National Basketball Association (NBA), established in 1946, grew to a multibillion-dollar enterprise by the end of the century, and basketball became an integral part of American culture."
cricket_rules="----- Read more: https://en.wikipedia.org/wiki/Laws_of_Cricket#The_Laws_today ----- Rules of cricket,From Wikipedia: The Laws of Cricket is a code which specifies the rules of the game of cricket worldwide. The earliest known code was drafted in 1744 and, since 1788, it has been owned and maintained by its custodian, the Marylebone Cricket Club (MCC) in London. There are currently 42 Laws (always written with a capital L) which outline all aspects of how the game is to be played. MCC has re-coded the Laws six times, the seventh and latest code being released in October 2017. The 2nd edition of the 2017 Code came into force on 1 April 2019. The first six codes prior to 2017 were all subject to interim revisions and so exist in more than one version."
cricket_history="----- Read more: https://en.wikipedia.org/wiki/History_of_cricket ----- History of cricket,From Wikipedia: The sport of cricket has a known history beginning in the late 16th century. Having originated in south-east England, it became the country's national sport in the 18th century and has developed globally in the 19th and 20th centuries. International matches have been played since 1844 and Test cricket began, retrospectively recognised, in 1877. Cricket is the world's second most popular spectator sport after association football (soccer). Governance is by the International Cricket Council (ICC) which has over one hundred countries and territories in membership although only twelve currently play Test cricket."
formulaone_rules="----- Read more: https://en.wikipedia.org/wiki/Formula_One_regulations#Current_rules_and_regulations ----- Rules of Formula 1,From Wikipedia: An F1 car can be no more than 200 cm wide and 95 cm tall. Though there is no maximum length, other rules set indirect limits on these dimensions, and nearly every aspect of the car carries size regulations; consequently the various cars tend to be very close to the same size. The car and driver must together weigh at least 740 kg."
formulaone_history="----- Read more: https://en.wikipedia.org/wiki/History_of_Formula_One ----- History of Formula 1,From Wikipedia: Formula One automobile racing has its roots in the European Grand Prix championships of the 1920s and 1930s, though the foundation of the modern Formula One began in 1946 with the Fédération Internationale de l'Automobile's (FIA) standardisation of rules, which was followed by a World Championship of Drivers in 1950. The sport's history parallels the evolution of its technical regulations. In addition to the world championship series, non-championship Formula One races were held for many years, the last held in 1983 due to the rising cost of competition. National championships existed in South Africa and the United Kingdom in the 1960s and 1970s."
tennis_rules="----- Read more: https://en.wikipedia.org/wiki/Tennis#Manner_of_play ----- Rules of tennis,From Wikipedia: The players or teams start on opposite sides of the net. One player is designated the server, and the opposing player is the receiver. The choice to be server or receiver in the first game and the choice of ends is decided by a coin toss before the warm-up starts. Service alternates game by game between the two players or teams. For each point, the server starts behind the baseline, between the centre mark and the sideline. The receiver may start anywhere on their side of the net. When the receiver is ready, the server will serve, although the receiver must play to the pace of the server."
tennis_history="----- Read more: https://en.wikipedia.org/wiki/History_of_tennis ----- History of tennis,From Wikipedia: The racket sport traditionally named lawn tennis, now commonly known simply as tennis, is the direct descendant of what is now denoted real tennis or royal tennis, which continues to be played today as a separate sport with more complex rules. Most rules of (lawn) tennis derive from this precursor and it is reasonable to see both sports as variations of the same game. Most historians believe that tennis was originated in the monastic cloisters in northern France in the 12th century, but the ball was then struck with the palm of the hand; hence, the name jeu de paume (game of the palm). It was not until the 16th century that rackets came into use, and the game began to be called tennis. It was popular in England and France, and Henry VIII of England was a big fan of the game, now referred to as real tennis."
soccer_rules="----- Read more: https://en.wikipedia.org/wiki/Association_football#Gameplay ----- Rules of soccer,From Wikipedia: Association football is played in accordance with a set of rules known as the Laws of the Game. The game is played using a spherical ball of 68–70 cm (27–28 in) circumference, known as the football (or soccer ball). Two teams of eleven players each compete to get the ball into the other team's goal (between the posts and under the bar), thereby scoring a goal. The team that has scored more goals at the end of the game is the winner; if both teams have scored an equal number of goals then the game is a draw. Each team is led by a captain who has only one official responsibility as mandated by the Laws of the Game: to represent their team in the coin toss prior to kick-off or penalty kicks."
soccer_history="----- Read more: https://en.wikipedia.org/wiki/History_of_association_football ----- History of soccer,From Wikipedia: Association football, more commonly known as football or soccer, is rooted in ancient sports such as Tsu' Chu played in Han Dynasty China and Kemari invented some 500-600 years later in Japan. Similar games existed in ancient Greece and Rome although little details remain of their rules or organisation. The more recent development of modern day association football has its origins in medieval ball games and English public school games. The modern game of association football originated with mid-nineteenth century efforts between local football clubs to standardize the varying sets of rules, culminating in formation of The Football Association in London, England in 1863. The rules drafted by the association allowed clubs to play each other without dispute, and specifically banned both handling of the ball (except by goalkeepers) and hacking during open field play. After the fifth meeting of the association a schism emerged between association football and the rules played by the Rugby school, later to be called rugby football. Football has been an Olympic sport ever since the second modern Summer Olympic Games in 1900."

# Create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])

# Webhook function
def webhook():
    req = request.get_json(silent=True, force=True)
    fulfillmentText = ''

    # Creating parameters to save the dialogflow raw response.
    # query_result to save all the raw request under "queryResult" json variable
    # userTextNewsRaw to save the text in "queryText" json variable
    # parameters to save the parameters in "parameters" json variable
    query_result = req.get('queryResult')
    userTextNewsRaw = query_result.get('queryText')
    parameters = query_result.get("parameters")
    
    # Code for classify headline intent    
    if query_result.get('action') == 'classify_headline':
        testHeadline = re.findall('"([^"]*)"', userTextNewsRaw)
        if len(testHeadline) == 0:
            fulfillmentText="Please enter the news in quotations"
        else:
            testHeadline = str(testHeadline)
            testHeadline = [testHeadline]
            resultDialogFlow = model.predict(testHeadline)
            lists = resultDialogFlow.tolist()
            json_str = json.dumps(lists)
            fulfillmentText = json_str
            fulfillmentText = fulfillmentText.replace('["', '').replace('"]', '').lower()
            fulfillmentText = "This is a " + fulfillmentText + " headline."

    # Code for latest news intent. 
    # getHeadlines() function is being called here
    elif query_result.get('action') == 'latest_news':
        parameterTopicName = parameters.get("latest_news_topic")
        testHeadline = parameterTopicName
        testHeadline = str(testHeadline)
        myHeadlines = getHeadlines(testHeadline)
        json_str = json.dumps(myHeadlines)
        fulfillmentText = json_str
        fulfillmentText = fulfillmentText.replace('[','').replace(']','')
    
    # Code for sports history intent.
    elif query_result.get('action') == 'sports_history':
        parameterTopicName = parameters.get("latest_news_topic")
        if parameterTopicName=="Basketball":
            fulfillmentText=basketball_history
        elif parameterTopicName=="Cricket":
            fulfillmentText = cricket_history
        elif parameterTopicName=="FormulaOne":
            fulfillmentText = formulaone_history
        elif parameterTopicName=="Soccer":
            fulfillmentText = soccer_history
        elif parameterTopicName=="Tennis":
            fulfillmentText = tennis_history
    
    # Code for sports rules intent.
    elif query_result.get('action') == 'sports_rules':
        parameterTopicName = parameters.get("latest_news_topic")
        if parameterTopicName=="Basketball":
            fulfillmentText=basketball_rules
        elif parameterTopicName=="Cricket":
            fulfillmentText = cricket_rules
        elif parameterTopicName=="FormulaOne":
            fulfillmentText = formulaone_rules
        elif parameterTopicName=="Soccer":
            fulfillmentText = soccer_rules
        elif parameterTopicName=="Tennis":
            fulfillmentText = tennis_rules
    
    # Return resonse to the chatbot.
    return {
        "fulfillmentText": fulfillmentText,
        "source": "webhookdata"
    }

# run the app
if __name__ == '__main__':
    app.run(debug=True)