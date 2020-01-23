from __future__ import division, print_function


import numpy as np
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from pdb import set_trace
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if len(t)<20 and len(t)>2]

class TM():

    def __init__(self,data,target,seed=0):
        np.random.seed(seed)
        self.data = data
        self.target = target

    def preprocess(self):
        def entropy(Nwt,N,Nt,Nw):
            return  np.nan_to_num(Nwt/N*np.log2(Nwt*N/Nt/Nw))

        self.x_content = []
        self.x_label = []
        for key in self.data:
            if key == self.target:
                self.y_content = [re.sub(r'[^a-zA-Z]', ' ', c.decode("utf8","ignore")) for c in self.data[key]["Abstract"]]
                self.y_label = self.data[key]["label"].tolist()
            else:
                self.x_content.append([re.sub(r'[^a-zA-Z]', ' ', c.decode("utf8","ignore")) for c in self.data[key]["Abstract"]])
                self.x_label.append(self.data[key]["label"].tolist())
        self.prediction = np.array([0]*len(self.y_label))

    def train(self):
        def entropy(Nwt,N,Nt,Nw):
            return  np.nan_to_num(Nwt/N*np.log2(Nwt*N/Nt/Nw))

        for i,content in enumerate(self.x_content):
            #  feature selection
            tfer = TfidfVectorizer(tokenizer=LemmaTokenizer(),lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                                   sublinear_tf=False, stop_words="english", decode_error="ignore")
            X = tfer.fit_transform(content)
            X[X != 0] = 1
            keys = np.array(tfer.vocabulary_.keys())[np.argsort(tfer.vocabulary_.values())]

            poses = np.where(np.array(self.x_label[i])=="yes")[0]
            N = X.shape[0]
            Nt = len(poses)
            NT = N-Nt
            Nw =np.array(X.sum(axis=0))[0]
            NW = N-Nw
            Nwt = np.array(X[poses].sum(axis=0))[0]
            NWt = Nt-Nwt
            NwT = Nw-Nwt
            NWT = N-Nt-Nw+Nwt
            IG = entropy(Nwt,N,Nt,Nw)+entropy(NWt,N,Nt,NW)+entropy(NwT,N,NT,Nw)+entropy(NWT,N,NT,NW)
            selected = keys[np.argsort(IG)[::-1][:int(X.shape[1]*0.1)]].tolist()

            # train model with selected features
            tfer = TfidfVectorizer(tokenizer=LemmaTokenizer(),lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                                   sublinear_tf=False, stop_words="english", decode_error="ignore",vocabulary=selected)
            X = tfer.fit_transform(content)
            X[X != 0] = 1
            Y = tfer.transform(self.y_content)
            Y[Y != 0] = 1
            model = MultinomialNB()
            model.fit(X, self.x_label[i])
            # get predictions
            self.prediction+=np.array([1 if l=="yes" else -1 for l in model.predict(Y)])



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

    def eval(self):
        decisions = ["yes" if votes>0 else "no" for votes in self.prediction]

        tp, fp, fn, tn = self.confusion(decisions, self.y_label)
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

        order = np.argsort(self.prediction)[::-1]
        labels = np.array(self.y_label)[order]

        result["AUC"] = self.AUC(labels)
        result["APFD"] = self.APFD(labels)
        result["p@10"] = Counter(labels[:10])["yes"] / float(len(labels[:10]))
        result["p@100"] = Counter(labels[:100])["yes"] / float(len(labels[:100]))
        return result
