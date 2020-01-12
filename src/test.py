from pdb import set_trace
from error import *
from sum_regularized_regression import Sum_Regularized_Regression


def exp_est():
    data = load_rest()
    data2 = load_new()
    results={}
    for target in data:
        x_content = []
        x_content2 = []
        x_label_old = []
        x_label_old2 = []
        y_label = []
        y_content = []
        y_content2 = []
        for project in data:
            if project==target:
                y_label += data[target]["label"].tolist()
                y_content += [c.decode("utf8","ignore") for c in data[target]["Abstract"]]
                y_content2 += [c.decode("utf8", "ignore") for c in data2[target]["Abstract"]]
            else:
                x_content += [c.decode("utf8","ignore") for c in data[project]["Abstract"]]
                x_content2 += [c.decode("utf8", "ignore") for c in data2[project]["Abstract"]]
                x_label_old += [c for c in data[project]["label"]]
                for i in range(len(data2[project]["label"])):
                    if data2[project]["pre_label"][i]=="yes":
                        x_label_old2.append("yes")
                    else:
                        x_label_old2.append(data2[project]["label"][i])

        t = Counter(y_label)["yes"]
        x_label = np.array([1 if x=="yes" else 0 for x in x_label_old])
        x_label2 = np.array([1 if x == "yes" else 0 for x in x_label_old2])

        treatment = LR(x_content,y_content)
        treatment.preprocess()
        treatment.train(x_label_old)
        x_test = treatment.model.decision_function(treatment.test_data)
        x_train = treatment.model.decision_function(treatment.train_data)
        est = Sum_Regularized_Regression()
        est.fit(x_train,x_label)
        y=est.predict(x_test)
        e1 = sum(y)

        treatment2 = LR(x_content2, y_content2)
        treatment2.preprocess()
        treatment2.train(x_label_old2)
        x_test2 = treatment2.model.decision_function(treatment2.test_data)
        x_train2 = treatment2.model.decision_function(treatment2.train_data)
        est2 = Sum_Regularized_Regression()
        est2.fit(x_train2, x_label2)
        y2 = est2.predict(x_test2)
        e2 = sum(y2)-Counter(data2[target]["pre_label"])["yes"]

        treatment3 = LR(x_content2, y_content)
        treatment3.preprocess()
        treatment3.train(x_label_old2)
        x_test3 = treatment3.model.decision_function(treatment3.test_data)
        x_train3 = treatment3.model.decision_function(treatment3.train_data)
        est3 = Sum_Regularized_Regression()
        est3.fit(x_train3, x_label2)
        y3 = est3.predict(x_test3)
        e3 = sum(y3)

        results[target]=(t,e1,e2,e3)

    print(results)
    set_trace()

def exp_mix():
    data = load_rest()
    rate=0.5
    content = []
    label = []
    for project in data:
        content += [c.decode("utf8","ignore") for c in data[project]["Abstract"]]
        label += [c for c in data[project]["label"]]
    content = np.array(content)
    label = np.array(label)
    n = len(label)
    select = int(rate*n)

    train = np.random.choice(range(n),select,replace=False)
    test = list(set(range(n))-set(train.tolist()))

    t = Counter(label[test])["yes"]

    x_label = np.array([1 if x=="yes" else 0 for x in label[train]])

    treatment = LR(content[train],content[test])
    treatment.preprocess()
    treatment.train(label[train])

    x_test = treatment.model.decision_function(treatment.test_data)
    x_train = treatment.model.decision_function(treatment.train_data)
    est = Sum_Regularized_Regression()
    est.fit(x_train,x_label)
    y=est.predict(x_test)
    print(t,sum(y))
    set_trace()

def exp_separate():
    data = load_rest()
    rate=0.5
    result = {}
    for project in data:
        content = [c.decode("utf8","ignore") for c in data[project]["Abstract"]]
        label = [c for c in data[project]["label"]]
        content = np.array(content)
        label = np.array(label)
        n = len(label)
        select = int(rate*n)

        train = np.random.choice(range(n),select,replace=False)
        test = list(set(range(n))-set(train.tolist()))

        t = Counter(label[test])["yes"]

        x_label = np.array([1 if x=="yes" else 0 for x in label[train]])

        treatment = LR(content[train],content[test])
        treatment.preprocess()
        treatment.train(label[train])

        x_test = treatment.model.decision_function(treatment.test_data)
        x_train = treatment.model.decision_function(treatment.train_data)
        est = Sum_Regularized_Regression()
        est.fit(x_train,x_label)
        y=est.predict(x_test)
        result[project] = (t,sum(y))
    print(result)
    set_trace()

if __name__ == "__main__":
    eval(cmd())