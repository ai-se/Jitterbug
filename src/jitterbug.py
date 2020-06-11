from __future__ import print_function, division

import matplotlib.pyplot as plt
import time

import pandas as pd

from supervised_models import *
from pdb import set_trace
class Jitterbug(object):
    def __init__(self,data,target):
        self.uncertain_thres = 10
        self.target = target
        self.data = data
        self.rest = self.data.copy()
        self.easy = Easy(self.data,self.target)
        self.easy.preprocess()

    def find_patterns(self):
        self.easy.find_patterns()
        return self.easy.patterns


    def test_patterns(self,output = False):
        self.easy.test_patterns(output=output)
        return self.easy.stats_test


    def easy_code(self):
        content = []
        for target in self.data:
            content += [c.lower() for c in self.data[target]["Abstract"]]
        csr_mat = self.easy.tfer.transform(content)

        indices = {}
        for pattern in self.easy.patterns:
            id = self.easy.voc.tolist().index(pattern)
            indices[pattern] = [i for i in range(csr_mat.shape[0]) if csr_mat[i,id] > 0]
        easy_code = ["no"]*len(content)

        for pattern in self.easy.patterns:
            for i in indices[pattern]:
                easy_code[i]="yes"
        start = 0
        for project in self.data:
            end = len(self.data[project]["label"])+start
            self.data[project]["easy_code"] = easy_code[start:end]
            self.rest[project]=self.data[project][self.data[project]["easy_code"]=="no"]
            start=end

    def output_conflicts(self,output="../new_data/conflicts/"):
        for project in self.data:
            x = self.data[project]
            conflicts = x[x["easy_code"]=="yes"][x["label"]=="no"]
            conflicts.to_csv(output+project+".csv", line_terminator="\r\n", index=False)

    def ML_hard(self, model = "RF", est = False, T_rec = 0.9):
        self.hard=Hard(model=model, est=est)
        self.hard.create(self.rest, self.target)
        step = 10
        while True:
            pos, neg, total = self.hard.get_numbers()
            # try:
            #     print("%d, %d, %d" %(pos,pos+neg, self.hard.est_num))
            # except:
            #     print("%d, %d" %(pos,pos+neg))

            if pos + neg >= total:
                break

            a,b,c,d =self.hard.train()

            if self.hard.est_num>0 and pos >=self.hard.est_num*T_rec:
                break

            if pos<self.uncertain_thres:
                self.hard.code_batch(a[:step])
            else:
                self.hard.code_batch(c[:step])
        return self.hard

    def eval(self):
        stat = Counter(self.data[self.target]['label'])
        t = stat["yes"]
        n = stat["no"]
        order = np.argsort(self.hard.body["time"][:self.hard.newpart])
        tp = self.easy.stats_test['tp']
        fp = self.easy.stats_test['p'] - tp
        tn = n - fp
        fn = t - tp

        # for stopping at target recall
        hard_tp = self.hard.record['pos'][-1]
        hard_p = self.hard.record['x'][-1]
        all_tp = hard_tp+tp
        all_fp = fp+hard_p-hard_tp
        prec = all_tp / float(all_fp+all_tp)
        rec = all_tp / float(t)
        f1 = 2*prec*rec/(prec+rec)
        ######################

        cost = 0
        costs = [cost]
        tps = [tp]
        fps = [fp]
        tns = [tn]
        fns = [fn]

        for label in self.hard.body["label"][order]:
            cost+=1.0
            costs.append(cost)
            if label=="yes":
                tp+=1.0
                fn-=1.0
            else:
                fp+=1.0
                tn-=1.0
            fps.append(fp)
            tps.append(tp)
            tns.append(tn)
            fns.append(fn)

        costs = np.array(costs)
        tps = np.array(tps)
        fps = np.array(fps)
        tns = np.array(tns)
        fns = np.array(fns)

        tpr = tps / (tps+fns)
        fpr = fps / (fps+tns)
        costr = costs / (t+n)

        auc = self.AUC(tpr,fpr)
        apfd = self.AUC(tpr,costr)

        return {"AUC":auc, "APFD":apfd, "TPR":tpr, "CostR":costr, "FPR":fpr, "Precision": prec, "Recall": rec, "F1": f1}

    def AUC(self,ys,xs):
        assert len(ys)==len(xs), "Size must match."
        if type(xs)!=type([]):
            xs=list(xs)
        if type(ys)!=type([]):
            ys=list(ys)
        x_last = 0
        if xs[-1]<1.0:
            xs.append(1.0)
            ys.append(ys[-1])
        auc = 0.0
        for i,x in enumerate(xs):
            y = ys[i]
            auc += y*(x-x_last)
            x_last = x
        return auc



class Hard(object):
    def __init__(self,model="RF", est=False):
        self.step = 10
        self.enable_est = est
        if model=="RF":
            self.model = RandomForestClassifier(class_weight="balanced_subsample")
        elif model=="NB":
            self.model = MultinomialNB()
        elif model == "LR":
            self.model = LogisticRegression(class_weight="balanced")
        elif model == "DT":
            self.model = DecisionTreeClassifier(class_weight="balanced",max_depth=8)
        elif model == "SVM":
            self.model = SGDClassifier(class_weight="balanced")

    def create(self,data,target):
        self.record={"x":[],"pos":[],'est':[]}
        self.body={}
        self.est=[]
        self.est_num = 0
        self.target = target
        self.loadfile(data[target])
        self.create_old(data)
        self.preprocess()

        return self

    def loadfile(self,data):
        self.body = data
        self.body['code']=["undetermined"]*len(self.body)
        self.body['time']=[0.0]*len(self.body)
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
        content0 = self.body["Abstract"][self.newpart:]
        content = self.body["Abstract"]

        tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,sublinear_tf=False,decode_error="ignore")
        tfer.fit(content0)
        self.csr_mat = tfer.transform(content)
        self.voc = np.array(list(tfer.vocabulary_.keys()))[np.argsort(list(tfer.vocabulary_.values()))]

        return

    ## Train model ##
    def train(self):

        sample = np.where(np.array(self.body['code']) != "undetermined")[0]
        self.model.fit(self.csr_mat[sample], self.body["code"][sample])

        if self.enable_est:
            self.est_num, self.est = self.estimate_curve()

        uncertain_id, uncertain_prob = self.uncertain()
        certain_id, certain_prob = self.certain()

        return uncertain_id, uncertain_prob, certain_id, certain_prob

        ## Get uncertain ##

    def uncertain(self):
        pos_at = list(self.model.classes_).index("yes")
        if type(self.model).__name__ == "SGDClassifier":
            prob = self.model.decision_function(self.csr_mat[self.pool])
            order = np.argsort(np.abs(prob))
        else:
            prob = self.model.predict_proba(self.csr_mat[self.pool])[:, pos_at]
            order = np.argsort(np.abs(prob-0.5))
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
        clf = LogisticRegression(penalty='l2', fit_intercept=True, class_weight="balanced")
        sample = np.where(np.array(self.body['code']) != "undetermined")[0]
        clf.fit(self.csr_mat[sample], self.body["code"][sample])

        prob = clf.decision_function(self.csr_mat[:self.newpart])
        prob2 = np.array([[p] for p in prob])

        y = np.array([1 if x == 'yes' else 0 for x in self.body['code'][:self.newpart]])
        y0 = np.copy(y)

        all = range(len(y))

        pos_num_last = Counter(y0)[1]
        if pos_num_last<10:
            return 0, []
        pos_origin = pos_num_last
        old_pos = pos_num_last - Counter(self.body["code"][:self.newpart])["yes"]

        lifes = 1
        life = lifes

        while (True):
            C = pos_num_last / pos_origin
            es = LogisticRegression(penalty='l2', fit_intercept=True, C=1)
            es.fit(prob2[all], y[all])
            pos_at = list(es.classes_).index(1)
            pre = es.predict_proba(prob2[self.pool])[:, pos_at]

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

        return esty, pre

    def get_allpos(self):
        return Counter(self.body["label"][:self.newpart])["yes"]

    def plot(self, T_rec = 0.9):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        t = float(self.get_allpos())
        n = float(len(self.body["label"][:self.newpart]))

        fig = plt.figure()

        costr = np.array(self.record["x"])/n
        tpr = np.array(self.record["pos"])/t
        estr = np.array(self.record["est"])/t

        for i,rec in enumerate(tpr):
            if estr[i]>0 and rec >= estr[i]*T_rec:
                break

        plt.plot(costr[:i+1], tpr[:i+1],label='Recall')
        plt.plot(costr[:i+1], estr[:i+1],'--',label='Estimation')
        plt.plot(costr[:i+1], [1.0]*(i+1),'-.',label='100% Recall')
        plt.plot(costr[:i+1], [T_rec]*(i+1),':',label=str(int(T_rec*100))+'% Recall')

        plt.ylim(0,1.5)
        plt.legend()
        plt.xlabel("Cost")
        plt.savefig("../figures_est/" + self.target + ".png")
        plt.close(fig)
        print(self.target)
        print((tpr[-1],costr[-1]))


class Easy(object):
    def __init__(self,data,target,thres=0.8):
        self.data=data
        self.target=target
        self.thres = thres
        self.stats_test = {'tp':0,'p':0}
        self.x_content = []
        self.x_label = []
        for project in data:
            if project==target:
                continue
            self.x_content += [c.lower() for c in data[project]["Abstract"]]
            self.x_label += [c for c in data[project]["label"]]
        self.x_label = np.array(self.x_label)
        self.y_content = [c.lower() for c in data[target]["Abstract"]]
        self.y_label = [c for c in data[target]["label"]]

    def preprocess(self):

        self.tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                               sublinear_tf=False, decode_error="ignore")
        self.train_data = self.tfer.fit_transform(self.x_content)
        self.test_data = self.tfer.transform(self.y_content)
        self.voc = np.array(list(self.tfer.vocabulary_.keys()))[np.argsort(list(self.tfer.vocabulary_.values()))]

    def find_patterns(self):
        left_train = range(self.train_data.shape[0])
        self.pattern_ids = []
        self.precs = []
        self.train_data[self.train_data.nonzero()]=1
        while True:
            id, fitness = self.get_top_fitness(self.train_data[left_train],self.x_label[left_train])
            left_train,stats_train = self.remove(self.train_data,self.x_label,left_train, id)
            prec = float(stats_train["tp"])/stats_train["p"]
            if prec<self.thres:
                break
            self.pattern_ids.append(id)
            self.precs.append(prec)
        self.patterns = self.voc[self.pattern_ids]


    def get_top_fitness(self,matrix,label):
        poses = np.where(label=="yes")[0]
        count_tp = np.array(np.sum(matrix[poses],axis=0))[0]
        count_p = np.array(np.sum(matrix,axis=0))[0]

        fitness = np.nan_to_num(count_tp*(count_tp/count_p)**3)
        order = np.argsort(fitness)[::-1]
        top_fitness = count_tp[order[0]]/count_p[order[0]]

        print({'tp':count_tp[order[0]], 'fp':count_p[order[0]]-count_tp[order[0]], 'fitness':top_fitness})
        return order[0], top_fitness

    def remove(self, data, label, left, id):
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

    def test_patterns(self,output=False):
        left_test = range(self.test_data.shape[0])
        self.stats_test={"tp":0,"p":0}
        for id in self.pattern_ids:
            left_test,stats_test = self.remove(self.test_data,self.y_label,left_test, id)
            self.stats_test["tp"]+=stats_test["tp"]
            self.stats_test["p"]+=stats_test["p"]
        # save the "hard to find" data
        if output:
            self.rest = self.data[self.target].loc[left_test]
            self.rest.to_csv("../new_data/rest/"+self.target+".csv", line_terminator="\r\n", index=False)
        return self.stats_test


class MAT(Easy):
    def find_patterns(self):
        self.patterns = ["todo","fixme","hack","xxx"]
        self.pattern_ids = [self.voc.tolist().index(x) for x in self.patterns]

class MAT_Two_Step(Jitterbug):
    def __init__(self,data,target):
        self.uncertain_thres = 0
        self.target = target
        self.data = data
        self.rest = self.data.copy()
        self.easy = MAT(self.data,self.target)
        self.easy.preprocess()

    def ML_hard(self, model = "RF"):
        treatments = {"RF":RF,"SVM":SVM,"LR":LR,"NB":NB,"DT":DT,"TM":TM}
        self.hard = treatments[model](self.rest,self.target)
        self.hard.preprocess()
        self.hard.train()
        return self.hard

    def eval(self):
        stat = Counter(self.data[self.target]['label'])
        t = stat["yes"]
        n = stat["no"]
        order = np.argsort(self.hard.probs)[::-1]
        tp = self.easy.stats_test['tp']
        fp = self.easy.stats_test['p'] - tp
        tn = n - fp
        fn = t - tp
        cost = 0
        costs = [cost]
        tps = [tp]
        fps = [fp]
        tns = [tn]
        fns = [fn]

        for label in np.array(self.hard.y_label)[order]:
            cost+=1.0
            costs.append(cost)
            if label=="yes":
                tp+=1.0
                fn-=1.0
            else:
                fp+=1.0
                tn-=1.0
            fps.append(fp)
            tps.append(tp)
            tns.append(tn)
            fns.append(fn)

        costs = np.array(costs)
        tps = np.array(tps)
        fps = np.array(fps)
        tns = np.array(tns)
        fns = np.array(fns)

        tpr = tps / (tps+fns)
        fpr = fps / (fps+tns)
        costr = costs / (t+n)

        auc = self.AUC(tpr,fpr)
        apfd = self.AUC(tpr,costr)

        return {"AUC":auc, "APFD":apfd, "TPR":tpr, "CostR":costr, "FPR":fpr}


class Easy_Two_Step(MAT_Two_Step):
    def __init__(self,data,target):
        self.uncertain_thres = 0
        self.target = target
        self.data = data
        self.rest = self.data.copy()
        self.easy = Easy(self.data,self.target)
        self.easy.preprocess()