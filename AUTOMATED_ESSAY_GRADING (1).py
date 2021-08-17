import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re,collections

dataset = pd.read_csv('data.csv' , encoding = 'latin-1')

print (dataset)

dataset.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10,10) , notch = True )

def sentence_to_wordlist(raw_sentence):
    
    clean_sentence = re.sub("[^a-zA-Z0-9]"," ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)
    
    return tokens

def tokenize(essay):
    stripped_essay = essay.strip()
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)
    
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    
    return tokenized_sentences

def word_count(essay):
    
    essay_1 = re.sub(r'\W', ' ', essay)
    words= nltk.word_tokenize(essay_1)
    
    return len(words)

def avg_word_len(essay):
    
    essay_1 = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(essay_1)
    
    return sum(len(word) for word in words) / len(words)

def char_count(essay):
    
    essay_1 = re.sub(r'\s', '', str(essay).lower())
    
    return len(essay_1)

def sent_count(essay):
    
    sentence = nltk.sent_tokenize(essay)
    
    return len(sentence)

def count_lemmas(essay):
    
    tokenized_sentences = tokenize(essay)      
    
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence) 
        
        for token_tuple in tagged_tokens:
        
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
    
    lemma_count = len(set(lemmas))
    
    return lemma_count


def count_pos(essay):
    
    tokenized_sentences = tokenize(essay)
    
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            
    return noun_count, adj_count, verb_count, adv_count

def count_spell_error(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    data = open('big.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if not word in word_dict:
            mispell_count += 1
    
    return mispell_count


def get_count_vectors(essays):
    
    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    
    count_vectors = vectorizer.fit_transform(essays)
    
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, count_vectors

from sklearn.feature_extraction.text import CountVectorizer

data = dataset[['essay_set','essay','domain1_score']].copy()

feature_names_cv, count_vectors = get_count_vectors(data[data['essay_set'] == 1]['essay'])

X_cv = count_vectors.toarray()

y_cv = data[data['essay_set'] == 1]['domain1_score'].as_matrix()


print(X_cv)

print(y_cv)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cv, y_cv, test_size = 0.3)

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

coef = linear_regressor.coef_
print(coef)

from sklearn.metrics import mean_squared_error, r2_score , cohen_kappa_score

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# variance score
print('Variance score: %.2f' % linear_regressor.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))


from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf_regressor.fit(X_train,y_train)

y_pred1 = rf_regressor.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred1))

# variance score
print('Variance score: %.2f' % rf_regressor.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred1), y_test))


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
alphas = np.array([3, 1, 0.3, 0.1, 0.03, 0.01])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas) , cv=3)
grid.fit(X_train, y_train)

y_pred2 = grid.predict(X_test)


print(grid.best_score_)
print(grid.best_estimator_.alpha)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred2))

# Explained variance score
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred2), y_test))

from sklearn.linear_model import Ridge
alphas = np.array([3, 1, 0.3, 0.1])

ridge_regressor = Ridge()

grid = GridSearchCV(estimator = ridge_regressor, param_grid = dict(alpha=alphas) , cv=3)
grid.fit(X_train, y_train)

y_pred3 = grid.predict(X_test)

print(grid.best_score_)
print(grid.best_estimator_.alpha)


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred3))

# variance score
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred3), y_test))

features = data.copy()
    
features['char_count'] = features['essay'].apply(char_count)
    
features['word_count'] = features['essay'].apply(word_count)
    
features['sent_count'] = features['essay'].apply(sent_count)
    
features['avg_word_len'] = features['essay'].apply(avg_word_len)
    
features['lemma_count'] = features['essay'].apply(count_lemmas)
    
features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))

print(features)

%matplotlib inline
features_set1 = features[features['essay_set'] == 1]
features_set1.plot.scatter(x = 'char_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'word_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'sent_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'avg_word_len', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'lemma_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'noun_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'adj_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'verb_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'adv_count', y = 'domain1_score', s=10)

X = features_set1.iloc[:, 3:].as_matrix()

y = features_set1['domain1_score'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

coef = linear_regressor.coef_
print(coef)

from sklearn.metrics import mean_squared_error, r2_score , cohen_kappa_score

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# variance score
print('Variance score: %.2f' % linear_regressor.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))

from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf_regressor.fit(X_train,y_train)

y_pred1 = rf_regressor.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred1))

# variance score
print('Variance score: %.2f' % rf_regressor.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred1), y_test))


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
alphas = np.array([3, 1, 0.3, 0.1, 0.03, 0.01])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas) , cv=3)
grid.fit(X_train, y_train)

y_pred2 = grid.predict(X_test)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred2))

# Explained variance score
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred2), y_test))

from sklearn.linear_model import Ridge
alphas = np.array([3, 1, 0.3, 0.1])

ridge_regressor = Ridge()

grid = GridSearchCV(estimator = ridge_regressor, param_grid = dict(alpha=alphas) , cv=3)
grid.fit(X_train, y_train)

y_pred3 = grid.predict(X_test)


print(grid.best_score_)
print(grid.best_estimator_.alpha)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred3))

# variance score
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred3), y_test))


X = np.concatenate((features_set1.iloc[:, 3:].as_matrix(), X_cv), axis = 1)

y = features_set1['domain1_score'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)


coef = linear_regressor.coef_
print(coef)

from sklearn.metrics import mean_squared_error, r2_score , cohen_kappa_score

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# variance score
print('Variance score: %.2f' % linear_regressor.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))

from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf_regressor.fit(X_train,y_train)

y_pred1 = rf_regressor.predict(X_test)


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred1))

# variance score
print('Variance score: %.2f' % rf_regressor.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred1), y_test))


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
alphas = np.array([3, 1, 0.3, 0.1, 0.03, 0.01])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas) , cv=3)
grid.fit(X_train, y_train)

y_pred2 = grid.predict(X_test)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred2))

# Explained variance score
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred2), y_test))

from sklearn.linear_model import Ridge
alphas = np.array([3, 1, 0.3, 0.1])

ridge_regressor = Ridge()

grid = GridSearchCV(estimator = ridge_regressor, param_grid = dict(alpha=alphas) , cv=3)
grid.fit(X_train, y_train)

y_pred3 = grid.predict(X_test)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred3))

# variance score
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred3), y_test))

from sklearn import ensemble

params = {'n_estimators':[100, 1000], 'max_depth':[2], 'min_samples_split': [2],
          'learning_rate':[3, 1, 0.1, 0.3], 'loss': ['ls']}

gbr = ensemble.GradientBoostingRegressor()

grid = GridSearchCV(gbr, params, cv=3)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_)

mse = mean_squared_error(y_test, y_pred)
print("MSE: %.4f" % mse)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score: 1 is complete agreement
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))

