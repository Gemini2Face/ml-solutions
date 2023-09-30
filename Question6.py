import io
import pandas as pd
import requests
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report , recall_score
from sklearn.ensemble import VotingClassifier



url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv"

response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))
df['great_customer_class'].value_counts()
df.shape

df.dropna(inplace=True)
y=df['great_customer_class']
df.shape
label_encoder = preprocessing.LabelEncoder()
df['workclass_clean']= label_encoder.fit_transform(df['workclass']) 
df['marital-status_clean']= label_encoder.fit_transform(df['marital-status'])
df['occupation_clean']= label_encoder.fit_transform(df['occupation'])
df['race_clean']= label_encoder.fit_transform(df['race'])
df['sex_clean']= label_encoder.fit_transform(df['sex'])
df.drop(columns=['workclass','marital-status','sex','occupation','race','user_id'],inplace=True)
df.head()
X=df
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.7,random_state = 0)



def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):

    classifier.fit(X_train, y_train)
    

    y_pred = classifier.predict(X_test)
    

    accuracy = accuracy_score(y_test, y_pred)
    

    report = classification_report(y_test, y_pred)
    
    return accuracy, report


random_forest = RandomForestClassifier(random_state=0,n_estimators=5)
svm = SVC(random_state=0)
logistic_regression = LogisticRegression(max_iter=1000, random_state=0)
naive_bayes = GaussianNB()
knn = KNeighborsClassifier()


classifiers = {
    "Random Forest": random_forest,
    "Support Vector Machines": svm,
    "Logistic Regression": logistic_regression,
    "Naive Bayes": naive_bayes,
    "KNN": knn
}

results = {}

for name, clf in classifiers.items():
    accuracy, report = train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test)
    results[name] = {
        "Accuracy": accuracy,
        
        "Classification Report": report
    }


for name, metrics in results.items():
    print(f"Classifier: {name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Classification Report:\n{metrics['Classification Report']}\n")
    


random_forest = RandomForestClassifier(random_state=0,n_estimators=10)
svm = SVC(random_state=0, probability=True)  
logistic_regression = LogisticRegression(max_iter=1000, random_state=0)


voting_classifier = VotingClassifier(estimators=[
    ('Random Forest', random_forest),
    ('SVM', svm),
    ('Logistic Regression', logistic_regression)
], voting='soft')  


voting_classifier.fit(X_train, y_train)


accuracy, report = train_and_evaluate_classifier(voting_classifier, X_train, y_train, X_test, y_test)


print(f"Ensemble Classifier (Voting)")
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")
# the best metric is recall as it is an unbalanced data, hence the accuracy scares are higher
    
