from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

# To quickly test the code, set numOfPage = 5
numOfPage = 2
# Saving URL pages that contain the soccer headlines
myUrls = []
for i in range(numOfPage):
    # Basketball
    #tempUrl = "https://www.reuters.com/news/archive/basketball-nba?view=page&page="+str(i+1)+"&pageSize=10"
    # Cricket
    #tempUrl = "https://www.espncricinfo.com/latest-cricket-news?page="+str(i+1)
    # Soccer
    #tempUrl = "https://www.reuters.com/news/archive/soccer-england?view=page&page="+str(i+1)+"&pageSize=10"
    # Formula One
    #tempUrl = "https://www.reuters.com/news/archive/sport-f1?view=page&page="+str(i+1)+"&pageSize=10"
    # Tennis
    tempUrl = "https://www.reuters.com/news/archive/sport-tennis?view=page&page="+str(i+1)+"&pageSize=10"
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
    headlines = myParsedPages[i].findAll("h3",{"class":"story-title"})
    myHeadlines.append(headlines)
    #Comment out only for cricket
    #headlines = myParsedPages[i].findAll("span", {"class": "story-title h3"})
    #myHeadlines.append(headlines)
    
# Saving the string format of each headline for each page
myStringHeadlines = []
for headlines in myHeadlines:
    for i in range(10):
        myStringHeadlines.append(str(headlines[i]).replace('<h3 class="story-title">','').replace('\n','').replace('								','').replace('</h3>',''))
        #myStringHeadlines.append([str(headlines[i]).replace('<h3 class="story-title">','').replace('\n','').replace('								','').replace('</h3>',''),"Label"])
        # Comment out only for Cricket
        #myStringHeadlines.append(str(headlines[i]).replace('<span class="story-title h3"><span>', '').replace('</span></span>', '').replace('</span></span>,', ''))

#for item in myStringHeadlines:
#    print(item)

print(len(myStringHeadlines))

textfile = open("TennisPic.txt", "w")
for element in myStringHeadlines:
    try:
        textfile.write(element + "\n")
    except UnicodeEncodeError:
        pass
textfile.close()
