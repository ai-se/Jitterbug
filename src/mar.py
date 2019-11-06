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


class MAR(object):
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





    def create(self,filename, old_files):
        self.filename=filename
        self.name=self.filename.split(".")[0]
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
        self.old_files = old_files
        self.est_num = 0


        self.loadfile()
        self.create_old()
        self.preprocess()

        return self



    def loadfile(self):
        self.body = pd.read_csv('../data/' + self.filename)

        label = ['no' if l=='WITHOUT_CLASSIFICATION' else 'yes' for l in self.body['classification']]

        self.body['label']=pd.Series(label, index=self.body.index)
        self.body['code']=pd.Series(['undetermined']*len(label), index=self.body.index)
        self.body['fixed']=pd.Series([0]*len(label), index=self.body.index)
        self.body['count']=pd.Series([0]*len(label), index=self.body.index)
        self.body['time']=pd.Series([0]*len(label), index=self.body.index)
        # self.start_time = np.min(self.body['time'])
        self.newpart = len(self.body)
        return

    ### Use previous knowledge, labeled only
    def create_old(self):
        bodies = [self.body]
        for k,file in enumerate(self.old_files):
            body = pd.read_csv('../data/' + str(file))


            label = ['no' if l=='WITHOUT_CLASSIFICATION' else 'yes' for l in body['classification']]
            body['code']=pd.Series(label)
            body['label']=pd.Series(label)
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
        content = [x for x in self.body["commenttext"]]



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




    def estimate_curve(self, clf, num_neg=0, boost=None):
        from sklearn import linear_model
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
                    sample.append(can[0])
                    count = 0
                    can = []
            return sample


        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        if boost:
            prob1 = self.body['yes_vote'].loc[~pd.isnull(self.body['yes_vote'])].values
            prob = np.array([[x] for x in prob1])
        else:
            prob1 = clf.decision_function(self.csr_mat)
            prob = np.array([[x] for x in prob1])
        old_pos = Counter(self.body['code'][self.newpart:])['yes']
        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)


        all = range(len(y))



        pos_num_last = Counter(y0)[1]

        lifes = 1
        life = lifes
        pos_num = Counter(y0)[1]

        while (True):
            C = (Counter(y[all])[1])/ (num_neg)
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C, random_state=self.seed)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)


            pre = es.predict_proba(prob[self.pool])[:, pos_at]



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
        esty = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]


        return esty-old_pos, pre


    ## Train supervised model ##
    def train_supervised(self, learner, seed):
        if learner == 'svm':
            self.clf = svm.SVC(probability=True, class_weight='balanced', random_state=seed)
        elif learner == 'dt':
            self.clf = DecisionTreeClassifier(random_state=seed, class_weight='balanced')
        elif learner == 'nbm':
            self.clf = MultinomialNB(alpha=1)
        elif learner == 'svm_linear':
            self.clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced', random_state=seed)

        labeled = np.array(range(self.newpart,len(self.body['code'])))
        self.clf.fit(self.csr_mat[labeled],self.body['code'][labeled])


    def query_supervised(self):
        pos_at = list(self.clf.classes_).index("yes")
        prob = self.clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1]
        certain_id = self.pool[order]
        if self.enable_est:
            self.est_num, self.est = self.estimate_curve(self.clf, num_neg=Counter(self.body['code'][self.newpart:])['no'])
        return certain_id

    def query_boost(self):
        temp = self.body['yes_vote'].loc[~pd.isnull(self.body['yes_vote'])][self.pool]
        order = np.argsort(temp)[::-1]
        certain_id = self.pool[order]
        if self.enable_est:
            self.est_num, self.est = self.estimate_curve(None,
                                                         num_neg=Counter(self.body['code'][self.newpart:])['no'], boost=True)
        return certain_id

    ## Train model ##
    def train(self,pne=True,weighting=True):

        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(kernel='linear', probability=True)
        clf_pre = tree.DecisionTreeClassifier(class_weight='balanced')
        poses = np.where(np.array(self.body['code'][:self.newpart]) == "yes")[0]
        negs = np.where(np.array(self.body['code'][:self.newpart]) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = self.pool
        try:
            unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),2*len(left),self.atleast)),replace=False)
        except:
            pass

        if not pne:
            unlabeled=[]

        labels=np.array([x if x!='undetermined' else 'no' for x in self.body['code'][:self.newpart]])
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

        ## correct errors with human-machine disagreements ##
        if self.round==self.interval:
            self.round=0
            susp, conf = self.susp(clf)
            return susp, conf, susp, conf
        else:
            self.round = self.round + 1
        #####################################################

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        if self.enable_est:
            self.est_num, self.est = self.estimate_curve(clf, num_neg=len(sample)-len(left))
            return uncertain_id, self.est[uncertain_id], certain_id, self.est[certain_id]
        else:
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

    ## Get certain_time ##
    def certain_time(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]/self.body['est_duration'][self.pool]
        order = np.argsort(prob)[::-1]
        return np.array(self.pool)[order[:self.step]], np.array(self.pool)[order]

    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)


    ## Opt order ##
    def opt(self):
        toorder = list(set(np.where(np.array(self.body['label'][:self.newpart]) == "yes")[0]))

        return np.array(toorder)[np.argsort(self.body['duration'][toorder])]

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
        labels = self.body["label"][ids]
        self.body["code"][ids] = labels
        self.body["time"][ids] = time.time()



    def code_error(self,id,error='none'):
        if error=='circle':
            self.code_circle(id, self.body['label'][id])
        elif error=='random':
            self.code_random(id, self.body['label'][id])
        elif error=='three':
            self.code_three(id, self.body['label'][id])
        else:
            self.code(id, self.body['label'][id])

    # def code_circle(self,id,label):
    #     import random
    #     if random.random()<0.0:
    #         self.body["code"][id] = label
    #     else:
    #         self.body["code"][id] = 'yes' if random.random()<float(self.body['syn_error'][id]) else 'no'
    #     self.body["time"][id] = time.time()

    def code_three(self, id, label):
        self.code_random(id,label)
        self.code_random(id,label)
        if self.body['fixed'][id] == 0:
            self.code_random(id,label)

    def code_random(self,id,label):
        import random
        error_rate = 0.3
        if label=='yes':
            if random.random()<error_rate:
                new = 'no'
            else:
                new = 'yes'
        else:
            if random.random()<error_rate:
                new = 'yes'
            else:
                new = 'no'
        if new == self.body["code"][id]:
            self.body['fixed'][id]=1
        self.body["code"][id] = new
        self.body["time"][id] = time.time()
        self.body["count"][id] = self.body["count"][id] + 1

    ## Get suspecious codes
    def susp(self,clf):
        thres_pos = 1
        thres_neg = 0.5
        length_pos = 10
        length_neg = 10

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        poses = np.array(poses)[np.where(np.array(self.body['fixed'])[poses] == 0)[0]]
        negs = np.array(negs)[np.where(np.array(self.body['fixed'])[negs] == 0)[0]]


        pos_at = list(clf.classes_).index("yes")
        prob_pos = clf.predict_proba(self.csr_mat[poses])[:,pos_at]
        se_pos = np.argsort(prob_pos)[:length_pos]
        se_pos = [s for s in se_pos if prob_pos[s]<thres_pos]
        sel_pos = poses[se_pos]
        # print(np.array(self.body['label'])[sel_pos])

        neg_at = list(clf.classes_).index("no")
        prob_neg = clf.predict_proba(self.csr_mat[negs])[:,neg_at]
        se_neg = np.argsort(prob_neg)[:length_neg]
        se_neg = [s for s in se_neg if prob_neg[s]<thres_neg]
        sel_neg = negs[se_neg]
        # print(np.array(self.body['label'])[sel_neg])


        return sel_pos.tolist() + sel_neg.tolist(), prob_pos[se_pos].tolist() + prob_neg[se_neg].tolist()




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


    ## Get missed relevant docs ##
    def get_rest(self):
        rest=[x for x in xrange(len(self.body['label'])) if self.body['label'][x]=='yes' and self.body['code'][x]!='yes']
        rests={}
        # fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        fields = ["Document Title"]
        for r in rest:
            rests[r]={}
            for f in fields:
                rests[r][f]=self.body[f][r]
        set_trace()
        return rests

    def cache_est(self):
        est = self.est[self.pool]
        order = np.argsort(est)[::-1]
        xx = [self.record["x"][-1]]
        yy = [self.record["pos"][-1]]
        for x in xrange(int(len(order) / self.step)):
            delta = sum(est[order[x * self.step:(x + 1) * self.step]])
            if delta >= 0.1:
                yy.append(yy[-1] + delta)
                xx.append(xx[-1] + self.step)
            else:
                break
        self.xx=xx
        self.yy=yy

        est = self.est2[self.pool]
        order = np.argsort(est)[::-1]
        xx2 = [self.record["x"][-1]]
        yy2 = [self.record["pos"][-1]]
        for x in xrange(int(len(order) / self.step)):
            delta = sum(est[order[x * self.step:(x + 1) * self.step]])
            if delta >= 0.1:
                yy2.append(yy2[-1] + delta)
                xx2.append(xx2[-1] + self.step)
            else:
                break
        self.xx2 = xx2
        self.yy2 = yy2