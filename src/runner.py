from __future__ import division, print_function

from copy import copy

import numpy as np
from pdb import set_trace

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import util
from demos import cmd
import pickle
import matplotlib.pyplot as plt
from os import listdir
import random

from collections import Counter
import time
from mar import MAR
from sk import rdivDemo
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp

def TEST_AL(filename, old_files = [], stop='est', stopat=1, error='none', interval = 100000, starting =1, seed=0, step =10):
    stopat = float(stopat)
    thres = 0
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename,old_files)
    read.step = step

    read.interval = interval



    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if pos >= target and read.est_num*stopat<= pos:
                break
            for id in c:
                read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    # read.plot()
    return read

def Supervised(filename, old_files = [], stop='', stopat=1, error='none', interval = 100000, starting =1, seed=0,
               step =10, learner='svm_linear', boost=None):
    print("FILENAME: ", filename, "OLDFILES: ", len(old_files))
    stopat = float(stopat)
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename, old_files)
    read.step = step

    read.interval = interval
    read.seed = seed

    if boost:
        util.vote(read, clf_name=boost, seed=seed, all=False, temp=str(seed) + filename)
    return
    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    if boost == None:
        read.train_supervised(learner, seed)
    pos, neg, total = read.get_numbers()

    if boost:
        read.query_boost()
    else:
        read.query_supervised()

    read.record['est'][0] = read.est_num

    while True:
        pos, neg, total = read.get_numbers()

        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        # if pos >= target and (pos+neg) >= total * .22 and read.enable_est and read.est_num*stopat<= pos:
        #     break
        if boost:
            ids = read.query_boost()[:read.step]
        else:
            ids = read.query_supervised()[:read.step]
        read.code_batch(ids)
    return read

'''Boosting Fahid'''
def Boosting(filename, old_files = [], stop='', stopat=1, error='none', interval = 100000, starting =1, seed=0, step =10):
    print("FILENAME: ", filename, "OLDFILES: ", len(old_files))
    stopat = float(stopat)
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename,old_files)
    read.step = step

    read.interval = interval

    util.vote(read)

    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    pos, neg, total = read.get_numbers()


    read.query_boost()
    read.record['est'][0]= read.est_num


    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        if read.enable_est and read.est_num*stopat<= pos:
            break
        for id in read.query_boost()[:read.step]:
            read.code_error(id, error=error)
    return read

def Estimate(filename, old_files = [], n=10, learner='lr'):
    read = MAR()
    read = read.create(filename, old_files)

    result = []
    for i in range(n):
        np.random.seed(i)
        a = read.body[['projectname', 'label']]
        b = a.loc[a['label'] == 'yes']
        total_df = a.groupby(['projectname']).count()
        yes_df = b.groupby(['projectname']).count()
        df = pd.DataFrame()
        df[['total']] = total_df[['label']]
        df[['pos']] = yes_df[['label']]

        test_file = filename.rsplit('.',1)[0]
        test_series = df.loc[test_file]
        train_df = df.drop([test_file])
        x_train = list(train_df.total.values)
        y_train = list(train_df.pos.values)
        if learner == 'lr':
            clf = LogisticRegression(random_state=i)
        elif learner == 'dt':
            clf = DecisionTreeClassifier(random_state=i)
        elif learner == 'svm_linear':
            clf = svm.SVC(kernel='linear', random_state=i)
        elif learner == 'nbm':
            clf = MultinomialNB(alpha=1)

        x_train = np.reshape(x_train, (-1, 1))
        clf.fit(x_train, y_train)
        res = clf.predict(test_series['total'])
        result.append(res[0])
    print(test_file, result)
    return result


def Plot(results,file_save, est_on=True, lbl='svm_linear'):
    font = {'family': 'normal',

                'size': 24}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 22, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': False, 'figure.figsize': (16, 8)}

    plt.rcParams.update(paras)

    fig = plt.figure()
    ax=plt.subplot(111)
    pos=results['true'][0]
    total = results['true'][1]
    colors=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    style = ['+', 'd', 'o', 'v', '^', 's', '.']

    i=0
    for key in results:
        if est_on==True:
            if key == 'true' or 'apfd' in key or 'supervised' != key:
                continue
        else:
            if key == 'true'or 'apfd' in key or 'supervised_' not in key:
                continue
        x = np.array(list(map(float,results[key]['x'])))/total
        y= np.array(list(map(float,results[key]['pos'])))/pos
        if est_on==False:
            label = results[key]['apfd']
            label = round(float(label), 2)
        else:
            label = ''
        if est_on == False:
            if 'svm_linear' in key:
                ax.plot(x, y, color=colors[i], markersize=10, markevery=100, marker=style[i], linestyle = '-', label=key.split('_', 1)[1] + ' ' + str(label))
            else:
                ax.plot(x, y, linewidth=3, color=colors[i], markersize=7, markevery=100, marker=style[i], linestyle='-',
                        label=key.split('_', 1)[1] + ' ' + str(label))
        else:
            ax.plot(x, y, color=colors[i], linestyle='-', label=lbl)
        if len(results[key]['est'])>1 and est_on:
            z= np.array(list(map(float,results[key]['est'])))/pos
            ax.plot(x, z, color=colors[i],linestyle = ':')
        i+=1

    plt.subplots_adjust(top=0.95, left=0.12, bottom=0.2, right=0.75)
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)
    plt.ylabel("Recall", fontweight='bold')
    plt.xlabel("Cost", fontweight='bold')


    plt.savefig('../test_figure/'+file_save+".png")
    plt.savefig('../test_figure/'+file_save+".pdf")
    plt.close(fig)

def sum_stop(results, file, target = 0.9, est_on=True):
    pos = float(results['true'][0])
    total = float(results['true'][1])
    thres = results['supervised']['thres']
    x = results['supervised']['x']
    y = results['supervised']['pos']
    est = results['supervised']['est']
    sup_est = 0
    stop_cost = 0
    twenty_pos = 0
    stop_pos = 0
    stop_est = 0.0001
    for i,cost in enumerate(x):
        if sup_est==0 and cost >= thres:
            sup_pos = y[i]
            sup_est = float(est[i])
            sup_cost = cost
        if stop_cost==0 and est[i]*target <= y[i]:
            stop_pos = y[i]
            stop_cost = cost
            stop_est = float(est[i])
        if twenty_pos==0 and cost>=total*0.2:
            twenty_pos = y[i]
            twenty_est = est[i]
            twenty_cost = cost
        if stop_cost and sup_est and twenty_pos:
            break
    print(file)
    print("Classification:")
    if est_on:
        print("True recall: %.2f, est recall: %.2f, precision: %.2f" %(sup_pos/pos, sup_pos/sup_est, sup_pos/float(sup_cost)))
        print("Stop at %.2f cost:" %0.2)
        print("True recall: %.2f, est recall: %.2f, precision: %.2f" %(twenty_pos/pos, twenty_pos/twenty_est, twenty_pos/float(twenty_cost)))
        print("Stop at %.2f recall:" %target)
        print("True recall: %.2f, est recall: %.2f, cost: %.2f" %(stop_pos/pos, stop_pos/stop_est, stop_cost/total))
        print("")
    else:
        print("True recall: %.2f, precision: %.2f" % (
        sup_pos / pos, sup_pos / float(sup_cost)))
        print("Stop at %.2f cost:" % 0.2)
        print("True recall: %.2f, precision: %.2f" % (
        twenty_pos / pos, twenty_pos / float(twenty_cost)))
        print("Stop at %.2f recall:" % target)
        print("True recall: %.2f, cost: %.2f" % (
        stop_pos / pos, stop_cost / total))
        print("")



### metrics

def APFD(read):
    n = len(read.body["code"][:read.newpart])
    m = Counter(read.body["code"][:read.newpart])["yes"]
    order = np.argsort(read.body['time'][:read.newpart])
    time = 0
    apfd = 0
    apfdc = 0
    num = 0
    time_total = sum(read.body['duration'][:read.newpart])
    for id in order:
        if read.body["code"][id] == 'undetermined':
            continue
        time += read.body["duration"][id]
        num+=1
        if read.body["code"][id] == 'yes':
            apfd += (num)
            apfdc += time_total - time + read.body["duration"][id] / 2

    apfd = 1-float(apfd)/n/m+1/(2*n)
    apfdc = apfdc / time_total / m
    return apfd, apfdc

def CostRecall(read, recall):
    m = Counter(read.body["code"][:read.newpart])["yes"]
    order = np.argsort(read.body['time'][:read.newpart])
    time_total = sum(read.body['duration'][:read.newpart])
    target = recall * m
    time = 0
    pos = 0
    for id in order:
        if read.body["code"][id] == 'undetermined':
            continue
        time += read.body["duration"][id]
        if read.body["code"][id] == 'yes':
            pos += 1
        if pos>=target:
            return time/time_total





def metrics(read, results, treatment, cost=0):
    results["APFD"][treatment], results["APFDc"][treatment] = APFD(read)
    results["X50"][treatment] = CostRecall(read, 0.5)
    results["FIRSTFAIL"][treatment] = CostRecall(read, 0.00000000001)
    results["ALLFAIL"][treatment] = CostRecall(read, .99999999999)
    results["runtime"][treatment] = cost
    return results




### exp

def exp_HPC(i , repeat=1, learner='nbm', boost=None, train_project=None, input = '../data/', dest_path='../temp/'):
    ori_files = listdir(input)
    file = ori_files[i]
    ori_files.remove(file)
    results = {}



    for repeat_counter in range(repeat):
        print("REPEAT: ", repeat_counter)
        files = copy(ori_files)
        if train_project:
            random.seed(repeat_counter)
            files = random.sample(files, train_project)
            print(files)
        read = Supervised(file, files, learner=learner, seed=repeat_counter, boost=boost, stop='est', stopat=.9)
        continue
        pos = Counter(read.body['label'][:read.newpart])['yes']
        total = read.newpart

        results['true'] = [pos, total]
        results['supervised' + str(repeat_counter)] = read.record
        if boost==None:
            thres = Counter(read.clf.predict(read.csr_mat[:read.newpart]))['yes']
            results['supervised' + str(repeat_counter)]['thres'] = thres

        with open(dest_path + '.'.join(file.split('.')[:-1]) + str(repeat_counter) + '.pkl', "wb") as h:
            pickle.dump(results, h)
    with open(dest_path + '.'.join(file.split('.')[:-1]) + '.pkl',"wb") as handle:
        pickle.dump(results, handle)

def exp_HPC_Parallel(i , repeat=10, learner='nbm', boost=None, input = '../data/', dest_path='../dump/'):

    files = listdir(input)
    file = files[i]
    files.remove(file)
    print('HPC', i)
    jobs = []
    for i in range(repeat):
        p = mp.Process(target=parallel_repeat, args=(file, files, i,))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    #Parallel(n_jobs=num_cpu)(delayed(parallel_repeat)(file, files, i) for i in range(repeat))

def parallel_repeat(file, files , repeat_counter, input = '../data/', dest_path='../dump/'):
    results = {}
    print(file + " REPEAT: " + str(repeat_counter))
    read = Supervised(file, files, learner='svm_linear', seed=repeat_counter, boost=None, stop='est', stopat=.05)
    pos = Counter(read.body['label'][:read.newpart])['yes']
    total = read.newpart

    results['true'] = [pos, total]
    results['supervised' + str(repeat_counter)] = read.record
    thres = Counter(read.clf.predict(read.csr_mat[:read.newpart]))['yes']
    results['supervised' + str(repeat_counter)]['thres'] = thres
    with open(dest_path + '.'.join(file.split('.')[:-1]) + str(repeat_counter) + '.pkl',"wb") as handle:
        pickle.dump(results, handle)

def exp_HPC_Estimate(repeat=10, learner='nbm', input = '../data/', dest_path='../dump_est/'):
    files = listdir(input)

    results = {}
    for file in files:
        current_files = listdir(input)
        current_files.remove(file)
        result = Estimate(file, current_files, n=repeat, learner=learner)
        results.setdefault(file, result)
    df = pd.DataFrame(results)
    df.to_csv(dest_path + 'est_' + learner + '.csv')

def plot_HPC(input = '../dump/'):
    files = listdir(input)
    for file in files:
        with open("../dump/"+file,"rb") as handle:
            result = pickle.load(handle)
        Plot(result,'.'.join(file.split('.')[:-1]), est_on=True)
        #sum_stop(result,'.'.join(file.split('.')[:-1]),target=0.95, est_on=True)

def collect_HPC(path = "./dump/"):
    files = listdir(path)
    results = {}
    for i, file in enumerate(files):
        with open(path+file,"r") as handle:
            one_run = pickle.load(handle)
        if i==0:
            for metrics in one_run:
                results[metrics] = {}
                for treatment in one_run[metrics]:
                    results[metrics][treatment] = [one_run[metrics][treatment]]
        else:
            for metrics in one_run:
                for treatment in one_run[metrics]:
                    results[metrics][treatment].append(one_run[metrics][treatment])

    summary = {}
    for metrics in results:
        summary[metrics] = {}
        for t in results[metrics]:
            try:
                summary[metrics][t]={'median': np.median(results[metrics][t]), 'iqr': np.percentile(results[metrics][t],75)-np.percentile(results[metrics][t],25)}
            except:
                set_trace()
    print("summary:")
    print(summary)
    with open("./result/result.pickle","w") as handle:
        pickle.dump(results,handle)

def sum_HPC():
    with open("./result/result.pickle","r") as handle:
        results = pickle.load(handle)
    summary = {}
    for metrics in results:
        summary[metrics] = {}
        for t in results[metrics]:
            summary[metrics][t]={'median': np.median(results[metrics][t]), 'iqr': np.percentile(results[metrics][t],75)-np.percentile(results[metrics][t],25)}
    print("summary:")
    print(summary)

    for metrics in results:
        print(metrics)
        if metrics == "X50":
            rdivDemo(results[metrics],bigger_better=False)
        else:
            rdivDemo(results[metrics],bigger_better=False)

def sum_relative():
    with open("./dump/result.pickle","r") as handle:
        results = pickle.load(handle)

    result = {}
    for metrics in results:
        result[metrics] = {}
        for t in results[metrics]:
            if t=='A2':
                continue
            result[metrics][t] = np.array(results[metrics][t]) / np.array(results[metrics]['A2'])
    results = result
    summary = {}
    for metrics in results:
        summary[metrics] = {}
        for t in results[metrics]:
            summary[metrics][t]={'median': np.median(results[metrics][t]), 'iqr': np.percentile(results[metrics][t],75)-np.percentile(results[metrics][t],25)}
    print("summary:")
    print(summary)

    for metrics in results:
        print(metrics)
        if metrics == "X50":
            rdivDemo(results[metrics],bigger_better=False)
        else:
            rdivDemo(results[metrics],bigger_better=False)


def exp_target(input = '../data/',target='apache-ant-1.7.0.csv'):
    files = listdir(input)
    files.remove(target)
    try:
        files.remove('.DS_Store')
    except:
        pass
    read = TEST_AL(target)
    set_trace()
    est = read.record['est'][0]
    print(target+": "+str(est)+" / "+ str(read.record['pos'][-1])+" / "+str(read.record['pos'][-1]))
    print(str(read.record['x'][-1])+" / "+ str(read.newpart))


def exp_all(input = '../data/'):
    files = listdir(input)
    for file in files:
        exp_target(input=input,target=file)




def get_charts_from_pickles():
    results_by_file = util.combine_pickles_for_plot_supervised()
    for i, x in results_by_file.items():
        Plot(x, i, est_on=False)


def multiprocessing(n):
    jobs = []
    for i in range(n):
        p = mp.Process(target=exp_HPC, args=(i, 10, 'svm_linear', None,))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()


def test():
    ori_files = listdir('../data/')
    file = ori_files[0]
    ori_files.remove(file)
    results = {}

    read = Supervised(file, ori_files, learner='nbm', seed=0, boost='nbm', stop='est', stopat=.9)
    pos = Counter(read.body['label'][:read.newpart])['yes']
    total = read.newpart

    results['true'] = [pos, total]
    results['supervised' + str(0)] = read.record
    thres = Counter(read.clf.predict(read.csr_mat[:read.newpart]))['yes']
    results['supervised' + str(0)]['thres'] = thres

    with open('../temp/' + '.'.join(file.split('.')[:-1]) + str(0) + '.pkl', "wb") as h:
        pickle.dump(results, h)


if __name__ == "__main__":

    num_cpu = mp.cpu_count()
    print("core:", num_cpu)
    #test()
    Parallel(n_jobs=num_cpu-2)(delayed(exp_HPC)(i, 10, 'dt', 'dt') for i in range(10))

    #multiprocessing(10)
    #util.combine_n_runs(n=10)
    #util.combine_n_runs_for_median_only(n=10)
    #plot_HPC()
    #get_charts_from_pickles()
    #eval(cmd())
    #modify_pickles()
    #exp_HPC_Estimate(learner='nbm')
    #get_charts_from_pickles()
    print('hi')

