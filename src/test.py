from pdb import set_trace
from error import *
from sum_regularized_regression import Sum_Regularized_Regression


def exp_est():
    data = load_rest()
    results={}
    for target in data:
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
        t = Counter(y_label)["yes"]
        x_label = np.array([1 if x=="yes" else 0 for x in x_label_old])
        treatment = SVM(x_content,y_content)
        treatment.preprocess()
        treatment.train(x_label_old)
        x_test = treatment.model.decision_function(treatment.test_data)
        x_train = treatment.model.decision_function(treatment.train_data)
        est = Sum_Regularized_Regression()
        est.fit(x_train,x_label)
        y=est.predict(x_test)
        results[target]=(t,sum(y))

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