from django.shortcuts import render
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def home(request):
    return render(request, 'WebApp/index.html')

def getPredictions(new_review):
    ps = PorterStemmer()
    cv = pickle.load(open('SentimentAnalysisScaler.sav', 'rb'))
    classifier = pickle.load(open('SentimentAnalysisModel.sav', 'rb'))

    new_review = re.sub('[^a-zA-Z]', ' ', str(new_review))
    print(new_review)

    new_review = new_review.lower()
    new_review = new_review.split()
    all_stopwords = set(stopwords.words('english'))
    all_stopwords.remove('not')
    print(new_review)

    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    print(new_review)

    new_corpus = [new_review]
    print(new_corpus)

    new_X_test = cv.transform(new_corpus).toarray()
    print(new_X_test)
    
    new_y_pred = classifier.predict(new_X_test).round()
    labels = ['N', 'O', 'P']
    new_y_pred = np.argmax(new_y_pred)
    [labels[new_y_pred]]
    print(new_y_pred)

    if new_y_pred == 'O':
        return 'okay'
    elif new_y_pred == 'P':
        return 'good'
    elif new_y_pred == 'N':
        return 'bad'
    else:
        return 'error'

def result(request):

    new_review = (request.POST.get('new_review', False))
    print(f"Fetched {new_review}")

    result = getPredictions(new_review)
    print(result)
    return render(request, 'WebApp/result.html', {'result': result})