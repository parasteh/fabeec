import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import gensim
from gensim.utils import tokenize
from gensim.models import FastText
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_non_alphanum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Config Variables
tfidf_based =  True
from_scratch = True
FT_VECTOR_SIZE = 300


f = open('data/emotions.txt')
lines  =f.readlines()
emotions = {}
for idx,label in enumerate(lines):
  emotions[idx] = label.strip()
emotions


dfTrain = pd.read_csv('data/train.tsv', sep = '\t', header=None, names = ['text' , 'label', 'rater'])
dfDev = pd.read_csv('data/dev.tsv', sep = '\t', header=None, names = ['text' , 'label', 'rater'])
dfTest = pd.read_csv('data/test.tsv', sep = '\t', header=None, names = ['text' , 'label', 'rater'])
data = pd.concat([dfTrain,dfTest, dfDev], ignore_index=True)


data.text = data.text.apply(lambda s: s.lower() )


def fasttext_model_train(data, from_scratch):
  # Preprocessing like stopword removal @TODO
  ge_sentences = [ list(tokenize(s)) for s in data['text'].to_list()]
  if from_scratch:
    model = FastText(bucket= 1000000, window=3, min_count=1, size=300)
    model.build_vocab(sentences=ge_sentences)
    model.train(sentences=ge_sentences, total_examples=len(ge_sentences), epochs=10)
  else:
    print("salam")
    model = FastText.load_fasttext_format('content/cc.en.300')
    model.build_vocab(ge_sentences, update=True)
    # model.train(sentences=ge_sentences, total_examples = len(sent), epochs=5)
  return model
  #now the model has been trained, there are two ways to get the sentence vectors,
  # first, simple averaging over the word vectors, like numpy.mean(...)
  # second, using the tfidf to applying a weighted average method,
  # There is an option using the original fastext library in python where the sentence vecotr is available but can be formolated here simply by adding two more functions




#TF-IDF Using SK_learn # needs a list of lists for words and docs along with a fasttext 'model'
def tfidf_model_train(data):
  tf_idf_vect = TfidfVectorizer(stop_words=None)
  tf_idf_vect.fit(data)
  final_tf_idf = tf_idf_vect.transform(data)
  tfidf_feat = tf_idf_vect.get_feature_names()
  return tfidf_feat, final_tf_idf

#@TODO new document should be concatenated to the DATA and then the tfidf matrix should be computed again  we need another method
#to do so  >>> https://stackoverflow.com/questions/40112373/how-to-classify-new-documents-with-tf-idf4444444466444




fastText_model = fasttext_model_train(data, from_scratch)  # training fastext
dictionary, tfidf_model = tfidf_model_train(data["text"].to_list())


# Sum( tfidf[word] * wvector) / sum(tfidf(words))
# @TODO maybe later develop the BM25 alg.
def get_sentence_vector(row, sent_tokens, dictionary, tfidf_model, ft_model, tfidf_based=False):
    if not tfidf_based:
        word_vectors = []
        for token in sent_tokens:
            w_vector = ft_model.wv[token]
            word_vectors.append(w_vector)
        return np.mean(word_vectors, axis=0)
    else:
        vec_sum = np.zeros(FT_VECTOR_SIZE)
        weight = 0;
        for token in sent_tokens:
            # print(token)
            try:
                index = dictionary.index(token)
            except:
                index = -1
            if index != -1:
                w_vector = ft_model.wv[token]
                # print("Vecotr ",w_vector)
                tfidf_score = tfidf_model[row, index]
                # print(tfidf_score)
                vec_sum += (tfidf_score * w_vector)
                weight += tfidf_score
        return vec_sum / weight


from tqdm import tqdm


# Fasttext word embedding
def fasttext_embedding(model, data, dictionary, tfidf_model, tfidf_based):
    sentence_vectors = []

    row = 0;
    for sentence in tqdm(data, position=0, leave=True):
        sent_tokens = tokenize(sentence.lower())
        sentence_vectors.append(get_sentence_vector(row, sent_tokens, dictionary, tfidf_model, model, tfidf_based))
        row += 1
        # print(sentence_vectors)
    return sentence_vectors



# fastext_raw = fasttext_embedding(fastText_model, data.text.to_list(), dictionary, tfidf_model,False)
fasttext_tfidf = fasttext_embedding(fastText_model, data.text.to_list(), dictionary, tfidf_model,True)

# data['fasttext_raw'] = fastext_raw
data['fasttext_tfidf'] = fasttext_tfidf



from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
x = mlb.fit_transform( [tuple(int(x) for x in i.split(',')) for i in data.label.to_list()])
data['new_label'] = list(x)
len(data.new_label[0])



start = 0
end  =  dfTrain.shape[0]

dfTrain ['fasttext_tfidf'] = data.fasttext_tfidf[start:end]
dfTrain['new_label'] = data.new_label[start:end]


start= dfTrain.shape[0]
end=  start + dfTest.shape[0]
dfTest ['fasttext_tfidf'] = data.fasttext_tfidf[start:end].values
dfTest['new_label'] = data.new_label[start:end].values

start= end
dfDev ['fasttext_tfidf'] = data.fasttext_tfidf[start:].values
dfDev['new_label'] = data.new_label[start:].values






def get_nan_ids(df):
  j = 0
  to_drop = []
  for i in df.fasttext_tfidf.to_list():
    if np.any(np.isnan(i)):
        to_drop.append(j)
    j+=1
  return to_drop

#remove nan for Train set
to_drop = get_nan_ids(dfTrain)
dfTrain = dfTrain.drop(to_drop)
print("Number of removed items in trainin set that contains Nan : {}, the removed inexes {}".format(len(to_drop), to_drop ))

#remove nan for dev test
to_drop = get_nan_ids(dfDev)
dfDev = dfDev.drop(to_drop)
print("Number of removed items in Dev set that contains Nan  : {}, the removed inexes {}".format(len(to_drop), to_drop ))

#remove nan for test data
to_drop = get_nan_ids(dfTest)
dfTest = dfTest.drop(to_drop)
print("Number of removed items in Test set that contains Nan  : {}, the removed inexes {}".format(len(to_drop), to_drop ))

from sklearn import tree
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def classifier_pipline(X_train, Y_train, X_test):
    # svm = LinearSVC(random_state=42)
    # xgmodel_default = xgb.XGBClassifier()
    # xgmodel_tuned = xgb.XGBClassifier(max_depth=20, sub_sample=0.7, colsample_bytree=0.7, eta=0.5)
    classifiers = [
        # KNeighborsClassifier(),
        # MultiOutputClassifier(svm, n_jobs=-1),
        # MultiOutputClassifier(xgmodel_default, n_jobs=-1),
        #MultiOutputClassifier(xgmodel_tuned, n_jobs=-1),
        RandomForestClassifier(n_estimators=1000, class_weight='balanced'),
        tree.DecisionTreeClassifier()

        #     SGDRegressor(),
        # #      ,
        #        BayesianRidge(),
        # #      LassoLars(),
        # #      ARDRegression()
        #       PassiveAggressiveRegressor(),
        #      TheilSenRegressor(),
        #     LinearRegression(),
        #     xgboost.XGBRegressor(early_stopping_rounds=10),
        #     xgboost.XGBRegressor(colsample_bytree= 0.8, gamma= 0, min_child_weight= 10, learning_rate=0.07, max_depth= 3,
        #  n_estimators= 250, reg_alpha= 1e-05, reg_lambda= 0.01, subsample= 0.95, early_stopping_rounds=10
        # )
    ]
    preds = []
    for item in classifiers:
        clf = item
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        preds.append(y_pred)
    return preds





X_train = np.array(dfTrain.fasttext_tfidf.to_list() + dfDev.fasttext_tfidf.to_list())
X_test = np.array(dfTest.fasttext_tfidf.to_list())

#standard scale

from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


y_train = np.array(dfTrain.new_label.to_list() + dfDev.new_label.to_list())

y_pred = classifier_pipline(X_train, y_train, X_test)





y_test = np.array(dfTest.new_label.to_list())
y_test


def Calculate_metric(y_test, y_pred):
    y_test[y_test > .3] = 1
    y_test[y_test <= .3] = 0

    results = {}
    results["macro_precision"], results["macro_recall"], results["macro_f1"], _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro')
    results["micro_precision"], results["micro_recall"], results["micro_f1"], _ = precision_recall_fscore_support(
        y_test, y_pred, average='micro')
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    #       num_emotions = len(emotions)

    #   idx2emotion = {i: e for i, e in enumerate(emotions)}

    #   preds_mat = np.zeros((len(preds), num_emotions))
    #   true_mat = np.zeros((len(preds), num_emotions))
    #   for i in range(len(preds)):
    #     true_labels = [int(idx) for idx in true.loc[i, "labels"].split(",")]
    #     for j in range(num_emotions):
    #       preds_mat[i, j] = preds.loc[i, idx2emotion[j]]
    #       true_mat[i, j] = 1 if j in true_labels else 0

    #   threshold = 0.3 # FLAGS.threshold
    #   pred_ind = preds_mat.copy()
    #   pred_ind[pred_ind > threshold] = 1
    #   pred_ind[pred_ind <= threshold] = 0
    #   results = {}
    #   results["accuracy"] = accuracy_score(true_mat, pred_ind)
    #   results["macro_precision"], results["macro_recall"], results[
    #       "macro_f1"], _ = precision_recall_fscore_support(
    #           true_mat, pred_ind, average="macro")
    #   results["micro_precision"], results["micro_recall"], results[
    #       "micro_f1"], _ = precision_recall_fscore_support(
    #           true_mat, pred_ind, average="micro")
    #   results["weighted_precision"], results["weighted_recall"], results[
    #       "weighted_f1"], _ = precision_recall_fscore_support(
    #           true_mat, pred_ind, average="weighted")
    #   for i in range(num_emotions):
    #     emotion = idx2emotion[i]
    #     emotion_true = true_mat[:, i]
    #     emotion_pred = pred_ind[:, i]
    #     results[emotion + "_accuracy"] = accuracy_score(emotion_true, emotion_pred)
    #     results[emotion + "_precision"], results[emotion + "_recall"], results[
    #         emotion + "_f1"], _ = precision_recall_fscore_support(
    #             emotion_true, emotion_pred, average="binary")
    return results






met_results = []
for met in y_pred:
    temp = Calculate_metric(y_test, met)
    met_results.append(temp)
    print(temp)











#CNN
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, MaxPooling1D, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import models


X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
y_train_cnn = y_train.reshape(y_train.shape[0], y_train.shape[1])

X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_test_cnn = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape = (300, 1)))

model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(28, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
]

history = model.fit(X_train_cnn, y_train_cnn,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=callbacks)





y_pred = model.predict(X_test_cnn)

y_pred[y_pred > .3] = 1
y_pred[y_pred <= .3] = 0
print(y_pred)
print(y_test)
results = Calculate_metric(y_test, y_pred)
print("CNN: ")
print(results)