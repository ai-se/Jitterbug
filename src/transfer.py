from __future__ import print_function, division
import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
import time
import os
from sklearn import preprocessing
import pandas as pd
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


class Transfer(object):
    def __init__(self,model="RF"):
        self.fea_num = 4000
        self.step = 10
        self.enough = 30
        self.kept=50
        self.atleast=100
        self.enable_est = True
        self.interval = 500000000
        self.ham=False
        self.seed = 0
        if model=="RF":
            self.model = RandomForestClassifier(class_weight="balanced")
        elif model=="NB":
            self.model = MultinomialNB()
        elif model == "LR":
            self.model = LogisticRegression(class_weight="balanced")
        elif model == "DT":
            self.model = DecisionTreeClassifier(class_weight="balanced",max_depth=8)
        elif model == "SVM":
            self.model = SGDClassifier(class_weight="balanced")





    def create(self,data,target):
        self.flag=True
        self.hasLabel=True
        self.record={"x":[],"pos":[],'est':[]}
        self.body={}
        self.est=[]
        self.last_pos=0
        self.last_neg=0
        self.record_est={"x":[],"semi":[],"sigmoid":[]}
        self.record_time = {"x":[],"pos":[]}
        self.round = 0
        self.est_num = 0



        self.target = target
        self.loadfile(data[target])

        self.create_old(data)
        self.preprocess()

        return self



    def loadfile(self,data):
        self.body = data
        self.body['code']=["undetermined"]*len(self.body)
        self.body['fixed']=[0]*len(self.body)
        self.body['count']=[0]*len(self.body)
        self.body['time']=[0.0]*len(self.body)
        # self.start_time = np.min(self.body['time'])
        self.newpart = len(self.body)
        return

    ### Use previous knowledge, labeled only
    def create_old(self,data):
        bodies = [self.body]
        for key in data:
            if key == self.target:
                continue
            body = data[key]
            label = body["label"]
            body['code']=pd.Series(label)
            body['fixed']=pd.Series([1]*len(label))
            body['count']=pd.Series([0]*len(label))
            bodies.append(body)

        self.body = pd.concat(bodies,ignore_index = True)




    def get_numbers(self):
        total = len(self.body["code"][:self.newpart])
        pos = Counter(self.body["code"][:self.newpart])["yes"]
        neg = Counter(self.body["code"][:self.newpart])["no"]


        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
            self.record['est'].append(int(self.est_num))
        self.pool = np.where(np.array(self.body['code'][:self.newpart]) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code'][:self.newpart]))) - set(self.pool))
        return pos, neg, total


    def preprocess(self):
        content0 = [c.decode("utf8", "ignore") for c in self.body["Abstract"][self.newpart:]]
        content = [c.decode("utf8","ignore") for c in self.body["Abstract"]]

        tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,sublinear_tf=False,decode_error="ignore")
        tfer.fit(content0)
        self.csr_mat = tfer.transform(content)
        self.voc = tfer.vocabulary_.keys()



        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###
        # tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
        #                         sublinear_tf=False,decode_error="ignore",max_features=4000)
        # tfidfer.fit(content)
        # self.voc = tfidfer.vocabulary_.keys()
        #
        #
        #
        #
        # ##############################################################
        #
        # ### Term frequency as feature, L2 normalization ##########
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        # # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        # #                 vocabulary=self.voc,decode_error="ignore")
        # self.csr_mat=tfer.fit_transform(content)
        ########################################################
        return

    ## Train model ##
    def train(self):


        sample = np.where(np.array(self.body['code']) != "undetermined")[0]
        self.model.fit(self.csr_mat[sample], self.body["code"][sample])

        self.est_num, self.est = self.estimate_curve()

        uncertain_id, uncertain_prob = self.uncertain()
        certain_id, certain_prob = self.certain()

        return uncertain_id, uncertain_prob, certain_id, certain_prob

        ## Get uncertain ##

    def uncertain(self):
        pos_at = list(self.model.classes_).index("yes")
        if type(self.model).__name__ == "SGDClassifier":
            prob = self.model.decision_function(self.csr_mat[self.pool])
            order = np.argsort(np.abs(prob))   ## uncertainty sampling by prediction probability
        else:
            prob = self.model.predict_proba(self.csr_mat[self.pool])[:, pos_at]
            order = np.argsort(np.abs(prob-0.5))   ## uncertainty sampling by prediction probability
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get certain ##
    def certain(self):
        pos_at = list(self.model.classes_).index("yes")
        if type(self.model).__name__ == "SGDClassifier":
            prob = self.model.decision_function(self.csr_mat[self.pool])
            order = np.argsort(prob)
            if pos_at>0:
                order = order[::-1]
        else:
            prob = self.model.predict_proba(self.csr_mat[self.pool])[:, pos_at]
            order = np.argsort(prob)[::-1]
        return np.array(self.pool)[order],np.array(self.pool)[order]


    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)

    ## Get one random ##
    def one_rand(self):
        pool_yes = filter(lambda x: self.body['label'][x]=='yes', range(len(self.body['label'])))
        return np.random.choice(pool_yes, size=1, replace=False)

    ## Format ##
    def format(self,id,prob=[]):
        result=[]
        for ind,i in enumerate(id):
            tmp = {key: self.body[key][i] for key in self.body}
            tmp["id"]=str(i)
            if prob!=[]:
                tmp["prob"]=prob[ind]
            result.append(tmp)
        return result



    ## Code candidate studies ##
    def code(self,id,label):

        self.body["code"][id] = label
        self.body["time"][id] = time.time()


    def code_batch(self,ids):
        now = time.time()
        times = [now+id/10000000.0 for id in range(len(ids))]
        labels = self.body["label"][ids]
        self.body["code"][ids] = labels
        self.body["time"][ids] =times


    def estimate_curve(self):
        from sklearn import linear_model
        from sum_regularized_regression import Sum_Regularized_Regression
        import random



        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= 1:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    count -= 1
                    can = []
            return sample



        ###############################################
        clf = linear_model.LogisticRegression(class_weight='balanced')
        sample = np.where(np.array(self.body['code']) != "undetermined")[0]
        clf.fit(self.csr_mat[sample], self.body["code"][sample])

        prob = clf.decision_function(self.csr_mat[:self.newpart])
        prob2 = np.array([[p] for p in prob])
        # prob = clf.apply(self.csr_mat)
        # prob = np.array([[x] for x in prob1])
        # prob = self.csr_mat


        y = np.array([1 if x == 'yes' else 0 for x in self.body['code'][:self.newpart]])
        y0 = np.copy(y)

        all = range(len(y))


        pos_num_last = Counter(y0)[1]
        if pos_num_last==0:
            pos_at = list(clf.classes_).index("yes")
            return sum(clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]), []
        pos_origin = pos_num_last
        old_pos = pos_num_last - Counter(self.body["code"][:self.newpart])["yes"]

        lifes = 1
        life = lifes

        while (True):
            C = pos_num_last / pos_origin
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)
            es.fit(prob2[all], y[all])
            pos_at = list(es.classes_).index(1)
            pre = es.predict_proba(prob2[self.pool])[:, pos_at]

            # es  =Sum_Regularized_Regression()
            # es.fit(prob[all], y[all])
            # pre = es.predict(prob[self.pool])


            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1



            pos_num = Counter(y)[1]

            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num


        esty = pos_num - old_pos
        pre = es.predict_proba(prob2)[:, pos_at]
        # pre = es.predict(prob)


        return esty, pre

    def APFD(self):
        order = np.argsort(self.body["time"][:self.newpart])
        labels = self.body["code"][order]
        n = self.newpart
        m = Counter(self.body["label"][:self.newpart])["yes"]
        apfd = 0
        for i,label in enumerate(labels):
            if label == 'yes':
                apfd += (i+1)
        apfd = 1-float(apfd)/n/m+1/(2*n)
        return apfd

    def get_allpos(self):
        return len([1 for c in self.body["label"][:self.newpart] if c=="yes"])


