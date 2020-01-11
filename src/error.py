from __future__ import division, print_function


import numpy as np
from fastread import FASTREAD
from transfer import Transfer
from estimate import Estimate
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
        # self.test_data = tfer.transform(self.y_content)

        tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                               sublinear_tf=False, decode_error="ignore")
        self.train_data = tfer.fit_transform(self.x_content)
        self.test_data = tfer.transform(self.y_content)

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
        if tp==0:
            result["precision"]=0
            result["recall"]=0
            result["f1"]=0
        else:
            result["precision"] = float(tp) / (tp+fp)
            result["recall"] = float(tp) / (tp+fn)
            result["f1"] = 2*result["precision"]*result["recall"]/(result["precision"]+result["recall"])
        if fp==0:
            result["fall-out"]=0
        else:
            result["fall-out"] = float(fp) / (fp+tn)



        pos_at = list(self.model.classes_).index("yes")
        probs = self.model.predict_proba(self.test_data)[:,pos_at]
        order = np.argsort(probs)[::-1]
        labels = np.array(y_label)[order]
        result["AUC"] = self.AUC(labels)
        result["APFD"] = self.APFD(labels)
        result["p@10"] = Counter(labels[:10])["yes"] / float(len(labels[:10]))
        result["p@100"] = Counter(labels[:100])["yes"] / float(len(labels[:100]))
        return result

class SVM(Treatment):

    def __init__(self,x_content,y_content,seed=0):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = svm.SVC(kernel="linear",probability=True,class_weight="balanced",random_state=seed)

class RF(Treatment):

    def __init__(self,x_content,y_content,seed=0):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = RandomForestClassifier(class_weight="balanced",random_state=seed)

class DT(Treatment):

    def __init__(self,x_content,y_content,seed=0):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = DecisionTreeClassifier(class_weight="balanced",max_depth=8,random_state=seed)

class NB(Treatment):

    def __init__(self,x_content,y_content,seed=0):
        np.random.seed(seed)
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = MultinomialNB()

class LR(Treatment):

    def __init__(self,x_content,y_content,seed=0):
        self.x_content = x_content
        self.y_content = y_content
        self.fea_num=4000
        self.model = LogisticRegression(class_weight="balanced",random_state=seed)


def active(data,start = "pre"):
    thres = 0
    starting = 1
    uncertain_thres = 20
    read=FASTREAD()
    read.create(data)


    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        if pos < starting or pos+neg<thres:
            if start=="random":
                read.code_batch(read.random())
            else:
                a,b,c,d =read.query_pre()
                read.code_batch(c)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if pos<uncertain_thres:
                read.code_batch(a)
            else:
                read.code_batch(c)
    return read.APFD()

def transfer(data, target, seed = 0):
    np.random.seed(seed)
    uncertain_thres = 0
    read=Transfer()
    read.create(data, target)
    step = len(data[target])
    step = 10


    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        a,b,c,d =read.train(weighting=True,pne=False)

        if pos<uncertain_thres:
            read.code_batch(a[:step])
        else:
            read.code_batch(c[:step])
    return read

def estimate(data, data_all, target, seed = 0):
    np.random.seed(seed)
    uncertain_thres = 0
    read=Estimate()
    read.create(data,data_all,target)
    step = 10
    count = 0

    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))
        count+=1
        if pos + neg >= total or count>1:
            break

        a,b,c,d =read.train(weighting=True,pne=False)

        if pos<uncertain_thres:
            read.code_batch(a[:step])
        else:
            read.code_batch(c[:step])

    return read



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
    validate = pd.read_csv("../new_data/validate/validate0.csv")
    fps = set(validate[validate["code"]=="no"]["Abstract"].tolist())


    data={}
    for file in listdir(path):
        if file==".DS_Store":
            continue
        df = pd.read_csv(path+file)
        truth = []
        for i in range(len(df["pre_label"])):
            if df["pre_label"][i]=="no":
                tmp = df["code"][i]
            elif df["Abstract"][i] in fps:
                tmp = "no"
            else:
                tmp = "yes"
            truth.append(tmp)
        df["label"] = truth
        df["code"] = ["undetermined"]*len(truth)
        data[file.split(".")[0]] = df.loc[:,["projectname","classification","Abstract","pre_label","label","code"]]

    return data

def load_rest():
    data = load_new()
    rest = {target: data[target][data[target]["pre_label"]=="no"] for target in data}
    return rest

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

def show_result_processed(results):
    metrics = results["ant"]["SVM"].keys()
    treatments = results["ant"].keys()

    for metric in metrics:
        df = {"Treatment":treatments}
        columns=["Treatment"]
        collects = []
        for data in results:
            columns.append(data)
            df[data] = [round(np.median(results[data][treatment][metric]), 2) for treatment in treatments]

        pd.DataFrame(df,columns=columns).to_csv("../results/processed_"+metric+".csv", index=False)


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
    conflicted.to_csv("../new_data/validate/validate.csv", line_terminator="\r\n", index=False)



def ken_validate():
    validate = pd.read_csv("../new_data/validate/validate.csv")
    ken = pd.read_csv("../new_data/validate/ken_validate.csv")
    ids = []
    the_list = ken["ID"].tolist()
    for id in validate["ID"]:
        ids.append(the_list.index(id))
    validate["code"] = ken["code"][ids]
    validate.to_csv("../new_data/validate/validate0.csv", line_terminator="\r\n", index=False)

def hack():
    data = load()
    content = []
    label = []
    keys = [u'todo', u'fixme', u'hack', u'workaround', u'xxx', u'defer',
       u'renederer', u'dms', u'yuck', u'addtrigger']
    for target in data:
        content += [c.decode("utf8","ignore").lower() for c in data[target]["Abstract"]]
        label += [c for c in data[target]["code"]]

    x=DT(content,content)
    x.preprocess()
    indices = {}
    for key in keys:
        id = x.voc.index(key)
        indices[key] = [i for i in range(x.train_data.shape[0]) if x.train_data[i,id] > 0]
    new_label = ["no"]*len(label)

    for key in keys:
        for i in indices[key]:
            new_label[i]="yes"
        tp,fp,fn,tn = x.confusion(new_label, label)
        metrics = {"tp": tp, "fp": fp, "fn": fn}
        print(key)
        print(metrics)
    set_trace()
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

    def get_top_prec(matrix,label):
        poses = np.where(label=="yes")[0]
        count_tp = np.array(np.sum(matrix[poses],axis=0))[0]
        count_p = np.array(np.sum(matrix,axis=0))[0]

        prec = np.nan_to_num(count_tp*(count_tp/count_p)**3)
        order = np.argsort(prec)[::-1]
        top_prec = count_tp[order[0]]/count_p[order[0]]

        print({'tp':count_tp[order[0]], 'fp':count_p[order[0]]-count_tp[order[0]], 'prec':top_prec})
        return order[0], top_prec

    def remove(data, label, left, id):
        to_remove = set()
        p = 0
        tp = 0
        for row in left:
            if data[row,id]>0:
                to_remove.add(row)
                p+=1
                if label[row]=="yes":
                    tp+=1
        left = list(set(left)-to_remove)
        return left, {"p":p, "tp":tp}

    data = load()
    # data = load_rest()

    for target in data:
        content = []
        label = []
        for target2 in data:
            if target2==target:
                continue
            content += [c.decode("utf8","ignore").lower() for c in data[target2]["Abstract"]]
            label += [c for c in data[target2]["code"]]

        label = np.array(label)
        content_test = [c.decode("utf8","ignore").lower() for c in data[target]["Abstract"]]
        label_test = [c for c in data[target]["code"]]
        x=DT(content,content_test)
        x.preprocess()
        left_train = range(x.train_data.shape[0])
        left_test = range(x.test_data.shape[0])
        words_id = []
        precs = []

        x.train_data[x.train_data.nonzero()]=1
        est=0
        tp = 0
        for i in xrange(20):
            id, prec = get_top_prec(x.train_data[left_train],label[left_train])
            words_id.append(id)
            precs.append(prec)

            left_train,_ = remove(x.train_data,label,left_train, id)
            left_test,removed = remove(x.test_data,label_test,left_test, id)
            set_trace()
            est+=removed["p"]*prec
            tp+=removed["tp"]
            print((est,tp))


        print(target)
        print(np.array(x.voc)[words_id])
        print((est,tp))




    set_trace()



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

def insert_dict(single,multiple):
    for key in single:
        if key not in multiple:
            multiple[key] = []
        multiple[key].append(single[key])
    return multiple

def exp_rest():
    data = load_rest()
    treatments = [SVM, RF, DT, NB, LR]
    # treatments = [RF]
    results={}
    for target in data:
        results[target]={}
        x_content = []
        x_label_old = []
        y_label = []
        y_content = []
        for project in data:
            if project==target:
                y_label += data[target]["label"].tolist()
                y_content += [c.decode("utf8","ignore") for c in data[target]["Abstract"]]
            else:
                x_content += [c.decode("utf8","ignore") for c in data[project]["Abstract"]]
                x_label_old += [c for c in data[project]["label"]]
        for model in treatments:
            result_old = {}
            num = 30 if model==RF else 1
            for seed in range(num):
                treatment = model(x_content,y_content,seed=seed)
                treatment.preprocess()
                treatment.train(x_label_old)
                try:
                    result_seed = treatment.eval(y_label)
                except:
                    set_trace()
                result_old = insert_dict(result_seed,result_old)
            results[target][model.__name__]=result_old
    set_trace()
    show_result_processed(results)



def exp_active():
    data = load_new()
    ns = []
    for key in data:
        n=len(data[key])
        print(key+": %d" %n)
        ns.append(n)
    print(sum(ns))
    apfds = {key: active(data[key],start="random") for key in data}
    print(apfds)
    set_trace()

def exp_transfer(seed=0):
    data = load_rest()
    ns = []
    for key in data:
        n=len(data[key])
        print(key+": %d" %n)
        ns.append(n)
    print(sum(ns))
    apfds = {key: transfer(data, key, seed).APFD() for key in data}
    print(apfds)
    with open("../dump2/"+str(seed)+".pickle","w") as f:
        pickle.dump(apfds,f)

def exp_transfer_details(seed=0):
    data = load_rest()
    ns = []
    for key in data:
        n=len(data[key])
        print(key+": %d" %n)
        ns.append(n)
    print(sum(ns))
    records = {key: transfer(data, key, seed).record for key in data}
    est = {key: (records[key]['pos'][-1],records[key]['est'][1]) for key in records}
    print(est)
    set_trace()

def exp_estimate_details(seed=0):
    data = load_rest()
    data_all = load_new()
    ns = []
    for key in data:
        n=len(data[key])
        print(key+": %d" %n)
        ns.append(n)
    print(sum(ns))
    records = {key: estimate(data, data_all, key, seed) for key in data}
    set_trace()
    est = {key: (records[key].get_allpos()-records[key].record['pos'][0],records[key].record['est'][1]-records[key].record['pos'][0]) for key in records}
    print(est)
    set_trace()

def collect_result():
    transfer_result = {}
    for seed in range(30):
        with open("../dump/"+str(seed)+".pickle","r") as f:
            apfds = pickle.load(f)
        transfer_result = insert_dict(apfds,transfer_result)
    medians = {key:np.median(transfer_result[key]) for key in transfer_result}
    set_trace()

if __name__ == "__main__":
    eval(cmd())