import csv
import pickle
import random
from os import listdir

import pandas as pd
import numpy as np
from pdb import set_trace

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def dataset_details():
    files = listdir('../data/')
    for file in files:
        df = load_data(file)
        total_true = len(df[(df['label'] == 'yes')])
        total_false = len(df[(df['label'] == 'no')])
        print(file, total_true, total_false)

def load_data(filename):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    with open("../data/" + filename, "r", encoding='utf-8') as csvfile:
        content = [x for x in csv.reader(csvfile, delimiter=',')]

    columns = content[0]
    content = content[1:]
    random.seed(0)
    random.shuffle(content)

    all_dataset_pd = pd.DataFrame(content[1:], columns=columns)

    all_dataset_pd['label'] = np.where((all_dataset_pd['classification'] != "WITHOUT_CLASSIFICATION"), 'yes',
                                            'no')
    all_dataset_pd.loc[:, 'code'] = 'undetermined'

    return all_dataset_pd


def combine_pickles_for_plot_supervised(path='../dump_supervised_no_est/'):
    """Combine all pickles for retrival curve"""
    """Returns all pickles by filename in a format that can be used to plot on matplotlib"""
    folders = listdir(path)
    files = []
    all_dicts = {}
    result = {}
    for folder in folders:
        files = listdir(path + folder)

        for file in files:
            with open(path + folder + '/' + file, "rb") as handle:
                res = pickle.load(handle)
                res['supervised_' + folder] = res.pop('supervisedQ2')
                res['supervised_' + folder]['apfd'] = res['apfds'].pop('supervisedQ2')
                result.setdefault(file.rsplit('.', 1)[0], []).append(res)

    final_result = {}
    for filename in result:
        a = result[filename][0]
        del_keys = []
        for key in a:
            if 'supervised_' not in key and key != 'true':
                del_keys.append(key)
        for key in del_keys:
            del a[key]
        temp = result[filename]
        for i in range(1, len(temp)):
            b = result[filename][i]
            c = [key for key in b if 'supervised_' in key]
            a.setdefault(c[0], b[c[0]])
        final_result.setdefault(filename, a)

    return final_result


def APFD_form_results(x, step_size=10):
    n = x['true'][0]
    m = x['true'][1]
    apfd = 0
    c = [key for key in x if 'supervised' in key]
    old_step = 0
    old_found = 0
    all_apfds = {}
    for key in c:
        for i in x[key]['pos'][1:]:
            new_step = old_step + step_size

            new_found = i - old_found
            old_found = i
            apfd += (new_found * ((new_step + old_step) / 2))
            old_step = new_step
        apfd = 1 - float(apfd) / n / m + 1 / (2 * n)
        all_apfds.setdefault(key, apfd)
    return all_apfds


def preprocess(df, tfer=None):
    if tfer:
        return tfer.transform(df['commenttext'])

    content = [x for x in df["commenttext"]]
    tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                              sublinear_tf=False, decode_error="ignore", max_features=4000)
    tfidfer.fit(content)
    voc = tfidfer.vocabulary_.keys()

    tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                           vocabulary=voc, decode_error="ignore")
    return tfer.fit_transform(content), tfer


def predict(clf, x_test, y_test, x_train, y_train, result_pd, col_name, bellwether_weights=None):

    if 'label' not in result_pd.columns:
        result_pd['label'] = y_test
        result_pd['yes_vote'] = 0
        result_pd['no_vote'] = 0

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    result_pd[col_name] = y_pred.tolist()

    weight = 1
    if bellwether_weights:
        weight = bellwether_weights.get(col_name)

    yes_ids = result_pd[result_pd.loc[:, col_name] == 'yes'].index
    result_pd.loc[yes_ids, 'yes_vote'] += (1 * weight * weight)

    no_ids = result_pd[result_pd.loc[:, col_name] == 'no'].index
    result_pd.loc[no_ids, 'no_vote'] += (1 * weight * weight)

def vote(read, clf_name='nbm', seed=0, all=False, temp=''):
    train_df = read.body.loc[read.body['code'] != 'undetermined']
    if all:
        test_df = read.body.loc[read.body['code'] != '']
    else:
        test_df = read.body.loc[read.body['code'] == 'undetermined']

    train_dataset_names = train_df.projectname.unique()

    result_pd = pd.DataFrame()
    for train_dataset_name in train_dataset_names:
        new_df = train_df.loc[train_df['projectname'] == train_dataset_name]
        new_df = new_df[['projectname', 'commenttext', 'label']]
        x_train, tfer = preprocess(new_df)
        y_train = new_df['label'].tolist()
        x_test = preprocess(test_df, tfer)
        y_test = test_df['label'].tolist()


        if clf_name in "dt":
            clf = DecisionTreeClassifier(random_state=seed)
        elif clf_name in "nbm":
            clf = MultinomialNB(alpha=1.0)
        elif clf_name in "svm":
            clf = SVC(random_state=seed)

        predict(clf, x_test, y_test, x_train, y_train, result_pd, train_dataset_name)

    result_pd['code_ensemble'] = np.where(result_pd['yes_vote'] > result_pd['no_vote'], 'yes', 'no')

    read.body['yes_vote'] = result_pd['yes_vote']
    #read.body['yes_vote'] = (result_pd['yes_vote'] - result_pd['yes_vote'].mean()) / (result_pd['yes_vote'].max() - result_pd['yes_vote'].min())
    with open('../temp/vote_df' + temp.rsplit('.',1)[0] + '.pkl', "wb") as h:
        pickle.dump(read.body, h)
    #read.body.to_csv('../temp/vote_df' + temp + '.csv')

def combine_n_runs(path='../dump/', dest_path='../dump/', n=10):
    files = listdir(path)

    for file in files:
        if 'ipynb' in file:
            continue
        result = {}
        with open(path + file, 'rb') as handle:
            res = pickle.load(handle)
            for i in range(n):
                result[i] = res['supervised' + str(i)]['pos']

            df = pd.DataFrame(result)

            res['supervisedQ1'] = res['supervised0'].copy()
            res['supervisedQ1']['pos'] = df.T.describe().T['25%'].tolist()
            res['supervisedQ2'] = res['supervised0'].copy()
            res['supervisedQ2']['pos'] = df.T.describe().T['50%'].tolist()
            res['supervisedQ3'] = res['supervised0'].copy()
            res['supervisedQ3']['pos'] = df.T.describe().T['75%'].tolist()
            a = APFD_form_results(res, step_size=10)
            res['apfds'] = a
        with open(dest_path + '.'.join(file.split('.')[:-1]) + '.pkl', "wb") as handle:
            pickle.dump(res, handle)



def combine_n_runs_for_median_only(path='../dump_90_9/', dest_path='../dump/', n=10):
    files = listdir(path)

    for file in files:
        if 'ipynb' in file:
            continue
        result = {}
        est = {}
        thres = {}
        with open(path + file, 'rb') as handle:
            res = pickle.load(handle)
            for i in range(n):
                result[i] = res['supervised' + str(i)]['pos']
                est[i] = res['supervised' + str(i)]['est']
                #thres[i] = res['supervised' + str(i)]['thres']

            df = pd.DataFrame(result)
            df_est = pd.DataFrame(est)

            res['supervised'] = res['supervised0'].copy()
            res['supervised']['pos'] = df.T.describe().T['50%'].tolist()

            res['supervised']['est'] = df_est.T.describe().T['50%'].tolist()

            a = APFD_form_results(res, step_size=10)
            res['apfds'] = a
        with open(dest_path + '.'.join(file.split('.')[:-1]) + '.pkl', "wb") as handle:
            pickle.dump(res, handle)



