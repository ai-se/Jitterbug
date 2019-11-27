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
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


class FASTREAD(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 10
        self.enough = 30
        self.kept=50
        self.atleast=100
        self.enable_est = True
        self.interval = 500000000
        self.ham=False
        self.seed = 0





    def create(self,body):
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

        self.body = body
        self.body['fixed']=[0]*len(self.body)
        self.body['count']=[0]*len(self.body)
        self.body['time']=[0]*len(self.body)
        self.prelabel = np.where(self.body["pre_label"]=="yes")[0]
        self.rest = np.where(self.body["pre_label"]=="no")[0]
        self.pre_yes = Counter(self.body["label"][self.prelabel])["yes"]
        self.preprocess()

        return self





    def preprocess(self):
        content = [x for x in self.body["Abstract"]]



        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore",max_features=4000)
        tfidfer.fit(content)
        self.voc = tfidfer.vocabulary_.keys()




        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                        vocabulary=self.voc,decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat=tfer.fit_transform(content)
        ########################################################
        return



    def get_numbers(self):
        total = len(self.body["code"][self.rest])
        pos = Counter(self.body["code"][self.rest])["yes"]
        neg = Counter(self.body["code"][self.rest])["no"]
        self.pool = self.rest[np.where(np.array(self.body['code'][self.rest]) == "undetermined")[0]]
        self.labeled = list(set(list(self.rest)) - set(self.pool))

        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
        return pos, neg, total


    ## Train model ##
    def train(self,pne=True,weighting=True):

        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(kernel='linear', probability=True)
        clf_pre = tree.DecisionTreeClassifier(class_weight='balanced')
        poses = self.rest[np.where(np.array(self.body['code'][self.rest]) == "yes")[0]]
        negs = self.rest[np.where(np.array(self.body['code'][self.rest]) == "no")[0]]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = self.pool
        try:
            unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),2*len(left),self.atleast)),replace=False)
        except:
            pass

        if not pne:
            unlabeled=[]

        labels=np.array([x if x!='undetermined' else 'no' for x in self.body['code']])
        all_neg=list(negs)+list(unlabeled)
        sample = list(decayed) + list(unlabeled)

        clf_pre.fit(self.csr_mat[sample], labels[sample])


        ## aggressive undersampling ##
        if len(poses)>=self.enough:
            pos_at = list(clf_pre.classes_).index("yes")
            train_dist = clf_pre.predict_proba(self.csr_mat[all_neg])[:,pos_at]
            negs_sel = np.argsort(train_dist)[:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])

        elif pne:
            pos_at = list(clf_pre.classes_).index("yes")
            train_dist = clf_pre.predict_proba(self.csr_mat[unlabeled])[:,pos_at]
            unlabel_sel = np.argsort(train_dist)[:int(len(unlabeled) / 2)]
            sample = list(decayed) + list(np.array(unlabeled)[unlabel_sel])
        clf.fit(self.csr_mat[sample], labels[sample])


        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## Get uncertain ##
    def uncertain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        train_dist = clf.decision_function(self.csr_mat[self.pool])
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        # order = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get certain ##
    def certain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1]
        return np.array(self.pool)[order[:self.step]],np.array(self.pool)[order]


    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)

    def query_pre(self):
        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced')

        clf.fit(self.csr_mat, self.body["pre_label"])

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob


    def APFD(self):
        order = np.argsort(self.body["time"][self.rest])
        labels = self.body["code"][self.rest][order]
        n = len(self.rest)
        m = Counter(self.body["label"][self.rest])["yes"]
        apfd = 0
        for i,label in enumerate(labels):
            if label == 'yes':
                apfd += (i+1)
        apfd = 1-float(apfd)/n/m+1/(2*n)
        return apfd


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
        labels = self.body["label"][ids]
        self.body["code"][ids] = labels
        self.body["time"][ids] = time.time()



    ## Plot ##
    def plot(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        fig = plt.figure()
        plt.plot(self.record['x'], self.record["pos"])
        ### estimation ####
        # if Counter(self.body['code'])['yes']>=self.enough:
        #     est=self.est2[self.pool]
        #     order=np.argsort(est)[::-1]
        #     xx=[self.record["x"][-1]]
        #     yy=[self.record["pos"][-1]]
        #     for x in xrange(int(len(order)/self.step)):
        #         delta = sum(est[order[x*self.step:(x+1)*self.step]])
        #         if delta>=0.1:
        #             yy.append(yy[-1]+delta)
        #             xx.append(xx[-1]+self.step)
        #         else:
        #             break
        #     plt.plot(xx, yy, "-.")
        ####################
        plt.ylabel("Tests Failed")
        plt.xlabel("Tests Run")
        name=self.name+ "_" + str(int(time.time()))+".png"

        dir = "./static/image"
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

        plt.savefig("./static/image/" + name)
        plt.close(fig)
        return name

    def get_allpos(self):
        return len([1 for c in self.body["label"][:self.newpart] if c=="yes"])-self.last_pos

