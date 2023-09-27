# Download data
# !wget -O reddit_data_split.zip https://gla-my.sharepoint.com/:u:/g/personal/jake_lever_glasgow_ac_uk/EapVNOIV84tPnQuuFBNgG9UBYIWipQ9JL4QTfSgRtIacBw?download=1
# !unzip -o reddit_data_split.zip

import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
import spacy
import seaborn as sns
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier



with open('reddit_train.json') as f:
    train_data = json.load(f)
with open('reddit_val.json') as f:
    validation_data = json.load(f)
with open('reddit_test.json') as f:
    test_data = json.load(f)


#move data to pandas 
train_data = pd.DataFrame(train_data)
validation_data = pd.DataFrame(validation_data)
test_data = pd.DataFrame(test_data)

train_labels = train_data['subreddit']
validation_labels = validation_data['subreddit']
test_labels = test_data['subreddit']



def get_subreddit_counts(data):
  subreddits = {}
  for subr in data['subreddit']:
    if subr in subreddits.keys():
      subreddits[subr] = subreddits.get(subr)+1
    else:
      subreddits[subr] = 1
  return {key: value for key, value in sorted(subreddits.items())}

train_data_counts = get_subreddit_counts(train_data)
validation_data_counts = get_subreddit_counts(validation_data)
test_data_counts = get_subreddit_counts(test_data)

print("Training data: "+str(train_data_counts))
print("Validation data: "+str(validation_data_counts))
print("Testing data: "+str(test_data_counts))

#show subreddit counts and plot counts on same plot to compare
plt.bar(range(len(train_data_counts)), list(train_data_counts.values()), tick_label=list(train_data_counts.keys()), color='orange', alpha=1)
plt.bar(range(len(validation_data_counts)), list(validation_data_counts.values()), tick_label=list(validation_data_counts.keys()), color='green', alpha=0.7)
plt.bar(range(len(test_data_counts)), list(test_data_counts.values()), tick_label=list(test_data_counts.keys()), color='purple', alpha=0.5)



nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')

#tokenizer 
def text_pipeline_spacy(text):
    tokens = []
    doc = nlp(text)
    for t in doc:
        if not t.is_stop and not t.is_punct and not t.is_space:
            tokens.append(t.lemma_.lower())
    return tokens

#evaluation summary 
def evaluation_summary(description, true_labels, predictions, target_classes, show_nice_cm=False):
  print("Evaluation for: " + description)
  print(classification_report(true_labels, predictions, digits=3, zero_division=0, target_names=target_classes))
  if (show_nice_cm):
    cm = confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots(figsize=(13,7)) 
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Predicted Label", ylabel="True Label", xticklabels=target_classes, 
        yticklabels=target_classes, title="Confusion matrix")
    plt.yticks(rotation=1)
  else:
    print('\nConfusion matrix:\n',confusion_matrix(true_labels, predictions)) 




 
# Feature pipeline model
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


#body and title feature union
prediction_pipeline = Pipeline([
        ('union', FeatureUnion(
          transformer_list=[
            ('body', Pipeline([
              ('selector', ItemSelector(key='body')),
              ('tf-idf', TfidfVectorizer(sublinear_tf=True, max_features = 30000)), 
              ])),
            ('title', Pipeline([
              ('selector', ItemSelector(key='title')),
              ('tf-idf', TfidfVectorizer(sublinear_tf=True, max_features = 30000)), 
              ])),
        ])
        )
    ])


train_features = prediction_pipeline.fit_transform(train_data)
test_features = prediction_pipeline.transform(test_data)

logr = LogisticRegression(solver='saga', C=1*(10**4), max_iter=1111)
logr_model = logr.fit(train_features, train_labels)
logr_tfidf_score = logr_model.score(test_features, test_labels)

print("Model accuracy with feature union: ", logr_tfidf_score)



# Combined model
clf1 = logr
clf2 = SGDClassifier(loss="log", max_iter=10)
clf3 = SVC(kernel='rbf')

#combine the three models
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', verbose=True)
eclf = eclf1.fit(train_features.todense(), train_labels)
improved_model_score = (eclf.score(test_features.todense(), test_labels))
predicted_labels = eclf.predict(test_features.todense())
evaluation_summary("Voting classifier with union of features", test_labels, predicted_labels,  s_reddits, show_nice_cm = True)