#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre :
# @Chapitre : 09 - Détermination d'opinons grâce à la classification de textes
#
# Modules necessaires :
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   SCIKIT-LEARN : 0.21.0
#   NLTK : 3.4
#
# Pour installer un module :
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------

#Chargement du fichier
import pandas as pnd
messagesTwitter = pnd.read_csv("datas/rechauffementClimatique.csv", delimiter=";")

#Informations sur le nombres d'observations et leur contenu
print(messagesTwitter.shape)
print(messagesTwitter.head(2))

#Transformation de la feature Croyance
messagesTwitter['CROYANCE'] = (messagesTwitter['CROYANCE']=='Yes').astype(int)
print(messagesTwitter.head(100))

#Fonction de normalisation
import re
def normalisation(message):
    message = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', message)
    message = re.sub('@[^\s]+','USER', message)
    message = message.lower().replace("ё", "е")
    message = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', message)
    message = re.sub(' +',' ', message)
    return message.strip()


#Normalisation
messagesTwitter["TWEET"] = messagesTwitter["TWEET"].apply(normalisation)
print(messagesTwitter.head(10))

#Chargement des StopWords
from nltk.corpus import stopwords
stopWords = stopwords.words('english')

#Suppression des Stops Words dans les différentes phrases
messagesTwitter['TWEET'] = messagesTwitter['TWEET'].apply(lambda message: ' '.join([mot for mot in message.split() if mot not in (stopWords)]))
print(messagesTwitter.head(10))


#Stemmatisation
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
messagesTwitter['TWEET'] = messagesTwitter['TWEET'].apply(lambda message: ' '.join([stemmer.stem(mot) for mot in message.split(' ')]))
print(messagesTwitter.head(10))


#Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
messagesTwitter['TWEET'] = messagesTwitter['TWEET'].apply(lambda message: ' '.join([lemmatizer.lemmatize(mot) for mot in message.split(' ')]))
print(messagesTwitter.head(10))

print("Fin de la préparation !")


#Jeux d'apprentissage et de test :
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messagesTwitter['TWEET'].values,  messagesTwitter['CROYANCE'].values,test_size=0.2)


#Creation du pipeline d'apprentissage
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

etapes_apprentissage = Pipeline([('frequence', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('algorithme', MultinomialNB())])


#Apprentissage
modele = etapes_apprentissage.fit(X_train,y_train)

from sklearn.metrics import classification_report
print(classification_report(y_test, modele.predict(X_test), digits=4))

#Nouvelle Phrase :
phrase = str(input("Please input your tweet here to see if it believed in climate change : "))
print(phrase)

#Normalisation
phrase = normalisation(phrase)

#Suppression des stops words
phrase = ' '.join([mot for mot in phrase.split() if mot not in (stopWords)])

#Stemmatization
phrase =  ' '.join([stemmer.stem(mot) for mot in phrase.split(' ')])

#Lemmitization
phrase = ' '.join([lemmatizer.lemmatize(mot) for mot in phrase.split(' ')])
print (phrase)

prediction = modele.predict([phrase])
print(prediction)
if(prediction[0]==0):
    print(">> Ne croit pas au rechauffement climatique...")
else:
    print(">> Croit au rechauffement climatique...")



#------ Utilisation de SVM ---

#Définition du Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
etapes_apprentissage = Pipeline([('frequence', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('algorithme', svm.SVC(kernel='linear', C=2))])


#Apprentissage
modele = etapes_apprentissage.fit(X_train,y_train)
from sklearn.metrics import classification_report
print(classification_report(y_test, modele.predict(X_test), digits=4))

#Recherche du meilleur paramètre C
from sklearn.model_selection import GridSearchCV
parametresC = {'algorithme__C':(1,2,4,5,6,7,8,9,10,11,12)}

rechercheCOptimal = GridSearchCV(etapes_apprentissage, parametresC,cv=2)
rechercheCOptimal.fit(X_train,y_train)
print(rechercheCOptimal.best_params_)


#Nouveau Paramètre C=1
etapes_apprentissage = Pipeline([('frequence', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('algorithme', svm.SVC(kernel='linear', C=1))])

modele = etapes_apprentissage.fit(X_train,y_train)
from sklearn.metrics import classification_report
print(classification_report(y_test, modele.predict(X_test), digits=4))