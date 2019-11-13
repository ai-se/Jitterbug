from __future__ import division, print_function


import numpy as np

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from os import listdir
import random

from collections import Counter

from sk import rdivDemo
import pandas as pd
from demos import cmd
from pdb import set_trace

try:
   import cPickle as pickle
except:
   import pickle


class Treatment():

    def __init__(self,x_content,y_content):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = "Some Model"


    def preprocess(self):
        # tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
        #                         sublinear_tf=False,decode_error="ignore")
        # tfidf = tfidfer.fit_transform(self.x_content)
        # weight = tfidf.sum(axis=0).tolist()[0]
        # kept = np.argsort(weight)[-self.fea_num:]
        # self.voc = np.array(tfidfer.vocabulary_.keys())[np.argsort(tfidfer.vocabulary_.values())][kept]
        # ##############################################################
        #
        # ### Term frequency as feature, L2 normalization ##########
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        # self.train_data = tfer.fit_transform(self.x_content)
        # self.test_data = tfer.fit_transform(self.y_content)


        tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,sublinear_tf=False,decode_error="ignore")
        self.train_data = tfer.fit_transform(self.x_content)
        self.test_data = tfer.fit_transform(self.y_content)
        ascend = np.argsort(tfer.vocabulary_.values())
        self.voc = [tfer.vocabulary_.keys()[i] for i in ascend]


    # def draw(self):
    #     from sklearn.externals.six import StringIO
    #     from IPython.display import Image
    #     from sklearn.tree import export_graphviz
    #     import pydotplus
    #     dot_data = StringIO()
    #     export_graphviz(self.model, out_file=dot_data,
    #                     filled=True, rounded=True,
    #                     special_characters=True)
    #     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #     Image(graph.create_png())



    def train(self, x_label):
        assert len(x_label)==len(self.x_content), "Size of training labels does not match training data."
        self.model.fit(self.train_data,x_label)

    def confusion(self,decisions, y_label):
        tp,fp,fn,tn = 0,0,0,0
        for i, d in enumerate(decisions):
            gt = y_label[i]
            if d=="yes" and gt=="yes":
                tp+=1
            elif d=="yes" and gt=="no":
                fp+=1
            elif d=="no" and gt=="yes":
                fn+=1
            elif d=="no" and gt=="no":
                tn+=1
        return tp,fp,fn,tn

    def AUC(self,labels):
        stats = Counter(labels)
        fn = stats["yes"]
        tn = stats["no"]
        tp,fp, auc = 0,0,0.0
        for label in labels:
            if label == "yes":
                tp+=1
                fn-=1
            else:
                dfpr = float(fp)/(fp+tn)
                fp+=1
                tn-=1
                tpr = float(tp)/(tp+fn)
                fpr = float(fp)/(fp+tn)
                auc+=tpr*(fpr-dfpr)
        return auc

    def APFD(self,labels):
        n = len(labels)
        m = Counter(labels)["yes"]
        apfd = 0
        for i,label in enumerate(labels):
            if label == 'yes':
                apfd += (i+1)
        apfd = 1-float(apfd)/n/m+1/(2*n)

        return apfd





    def eval(self, y_label):
        assert len(y_label)==len(self.y_content), "Size of test labels does not match test data."
        decisions = self.model.predict(self.test_data)
        tp,fp,fn,tn = self.confusion(decisions, y_label)
        result = {}
        result["precision"] = float(tp) / (tp+fp)
        result["recall"] = float(tp) / (tp+fn)
        result["fall-out"] = float(fp) / (fp+tn)
        result["f1"] = 2*result["precision"]*result["recall"]/(result["precision"]+result["recall"])


        pos_at = list(self.model.classes_).index("yes")
        probs = self.model.predict_proba(self.test_data)[:,pos_at]
        order = np.argsort(probs)[::-1]
        labels = np.array(y_label)[order]
        result["AUC"] = self.AUC(labels)
        result["APFD"] = self.APFD(labels)
        return result

class SVM(Treatment):

    def __init__(self,x_content,y_content):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = svm.SVC(kernel="linear",probability=True,class_weight="balanced")

class RF(Treatment):

    def __init__(self,x_content,y_content):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = RandomForestClassifier(class_weight="balanced")

class DT(Treatment):

    def __init__(self,x_content,y_content):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = DecisionTreeClassifier(class_weight="balanced",max_depth=8)

class NB(Treatment):

    def __init__(self,x_content,y_content):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = MultinomialNB()

class LR(Treatment):

    def __init__(self,x_content,y_content):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = LogisticRegression(class_weight="balanced")


def load(path="../new_data/"):
    data={}
    for file in listdir(path+"data/"):
        if file==".DS_Store":
            continue
        df0 = pd.read_csv(path+"data/"+file)
        df1 = pd.read_csv(path+"coded/"+file)
        new_code = []
        time = []
        for abs in df0["Abstract"]:
            new_code.append(df1[df1["Abstract"]==abs]["code"].values[0])
            time.append(df1[df1["Abstract"]==abs]["time"].values[0])
        df0["new_code"] = new_code
        df0["time"] = time
        data[file.split(".")[0]] = df0
    return data

def load_new(path="../new_data/processed/"):
    data={}
    for file in listdir(path):
        if file==".DS_Store":
            continue
        df = pd.read_csv(path+file)
        data[file.split(".")[0]] = df
    return data


def show_result(results):
    metrics = results["ant"]["SVM"]["old"].keys()
    treatments = results["ant"].keys()

    for metric in metrics:
        df = {"Treatment":treatments}
        columns=["Treatment"]
        for data in results:
            columns+=[data+"_new", data+"_old"]
            df[data+"_new"] = [round(results[data][treatment]["new"][metric], 2) for treatment in treatments]
            df[data+"_old"] = [round(results[data][treatment]["old"][metric], 2) for treatment in treatments]
        pd.DataFrame(df,columns=columns).to_csv("../results/"+metric+".csv", line_terminator="\r\n", index=False)


def validate():
    data = load_new()
    content = []
    label = []
    pre_label = []
    for target in data:
        content += [c.decode("utf8","ignore").lower() for c in data[target]["Abstract"]]
        label += [c for c in data[target]["code"]]
        pre_label += [c for c in data[target]["pre_label"]]
    df = pd.DataFrame({"ID":range(len(content)), "Abstract": content, "code": label, "pre_label":pre_label},columns=["ID","Abstract","code","pre_label"])

    conflicted = df[df["pre_label"]=="yes"][df["code"]=="no"]
    conflicted.rename(columns = {'code':'groundtruth'}, inplace = True)
    conflicted.to_csv("../results/validate.csv", line_terminator="\r\n", index=False)


def hack():
    data = load()
    content = []
    label = []
    keys = ["fixme","todo","workaround","hack"]
    for target in data:
        content += [c.decode("utf8","ignore").lower() for c in data[target]["Abstract"]]
        label += [c for c in data[target]["code"]]

    x=DT(content,content)
    indices = {}
    for key in keys:
        indices[key] = [i for i,c in enumerate(content) if key in c]
    new_label = ["no"]*len(label)
    for key in keys:
        for i in indices[key]:
            new_label[i]="yes"
        tp,fp,fn,tn = x.confusion(new_label, label)
        metrics = {"tp": tp, "fp": fp, "fn": fn}
        print(key)
        print(metrics)
    start = 0
    for target in data:
        end = len(data[target]["code"])+start
        data[target]["pre_label"] = new_label[start:end]
        start=end
        data[target].to_csv("../new_data/processed/"+target+".csv", line_terminator="\r\n", index=False)

    for i,l in enumerate(label):
        if l=="no" and new_label[i]=="yes":
            print(content[i])
            set_trace()

def highest_prec():
    data = load()
    content = []
    label = []
    for target in data:
        content += [c.decode("utf8","ignore").lower() for c in data[target]["Abstract"]]
        label += [c for c in data[target]["code"]]
    x=DT(content,content)
    x.preprocess()
    precs = {}
    for key in x.voc:
        ids = [i for i,c in enumerate(content) if key in c]
        new_label = ["no"]*len(label)
        for i in ids:
            new_label[i]="yes"
        tp,fp,fn,tn = x.confusion(new_label, label)
        prec = float(tp)/(tp+fp)
        precs[key]=prec
    order = np.argsort(precs.values())[::-1][:10]
    for o in order:
        print((precs.keys()[o], precs.values()[o]))


def exp():
    data = load()
    treatments = [SVM, RF, DT, NB, LR]
    results={}
    for target in data:
        results[target]={}
        x_content = []
        x_label_old = []
        x_label_new = []
        y_label = []
        y_content = []
        for project in data:
            if project==target:
                tmp = data[target][data[target]["code"]==data[target]["new_code"]]
                y_label += [c for c in tmp["code"]]
                y_content += [c.decode("utf8","ignore") for c in tmp["Abstract"]]
            else:
                x_content += [c.decode("utf8","ignore") for c in data[project]["Abstract"]]
                x_label_old += [c for c in data[project]["code"]]
                x_label_new += [c for c in data[project]["new_code"]]
        for model in treatments:
            treatment = model(x_content,y_content)
            treatment.preprocess()
            treatment.train(x_label_old)
            result_old = treatment.eval(y_label)
            treatment.train(x_label_new)
            result_new = treatment.eval(y_label)
            results[target][model.__name__]={"new":result_new, "old":result_old}

    show_result(results)

def stats():
    data = load()
    columns=["Project","label","original","questioned","changed"]
    table = {c:[] for c in columns}
    for project in data:
        df = data[project]
        table["Project"].extend([project,project])
        table["label"].extend(["yes","no"])
        O = Counter(df["code"])
        table["original"].extend([O["yes"],O["no"]])
        questioned = df[df["time"]>0]
        Q = Counter(questioned["code"])
        C = Counter(questioned[questioned["code"]!=questioned["new_code"]]["code"])
        table["questioned"].extend([Q["yes"],Q["no"]])
        table["changed"].extend([C["yes"],C["no"]])
    pd.DataFrame(table, columns=columns).to_csv("../results/questioned.csv", line_terminator="\r\n", index=False)

def learner():
    data = load()
    x_content = []
    x_label_old = []
    for project in data:
        x_content += [c.decode("utf8","ignore") for c in data[project]["Abstract"]]
        x_label_old += [c for c in data[project]["code"]]
    treatment = DT(x_content,x_content)
    treatment.preprocess()
    treatment.train(x_label_old)
    features = treatment.model.feature_importances_
    order = np.argsort(features)[::-1][:20]
    out = []
    for o in order:
        out.append((treatment.voc[o],features[o]))
    print(out)
    # treatment.draw()


if __name__ == "__main__":
    eval(cmd())