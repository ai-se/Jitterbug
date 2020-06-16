# [Identifying Self-Admitted Technical Debts with Jitterbug: A Two-step Approach](https://arxiv.org/abs/2002.11049)

Cite as:
``` 
@misc{yu2020identifying,
    title={Identifying Self-Admitted Technical Debts with Jitterbug: A Two-step Approach},
    author={Zhe Yu and Fahmid Morshed Fahid and Huy Tu and Tim Menzies},
    year={2020},
    eprint={2002.11049},
    archivePrefix={arXiv},
    primaryClass={cs.SE}
}
```

## Data
 - [Original](https://github.com/ai-se/tech-debt/tree/master/data) from Maldonado and Shihab "Detecting and quantifying different types of self-admitted  technical  debt," in 2015 IEEE 7th InternationalWorkshop on Managing Technical Debt (MTD). IEEE, 2015, pp. 9–15.
 - [Corrected](https://github.com/ai-se/tech-debt/tree/master/new_data/corrected): 439 labels checked, 431 labels corrected.
 
## Experiments
### Setup
```
Jitterbug$ pip install -r requirements.txt
Jitterbug$ python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
Jitterbug$ cd src
```
### RQ1: How to find the strong patterns of the "easy to find" SATDs in Step 1?
 - Prepare data:
 ```
 src$ python main.py parse
 ```
 - Find patterns with Easy (target project = apache-ant-1.7.0):
 ```
 src$ python main.py find_patterns
 
 {'fp': 367.0, 'tp': 2493.0, 'fitness': 0.8716783216783217}
 {'fp': 53.0, 'tp': 330.0, 'fitness': 0.8616187989556136}
 {'fp': 7.0, 'tp': 87.0, 'fitness': 0.925531914893617}
 {'fp': 3.0, 'tp': 46.0, 'fitness': 0.9387755102040817}
 {'fp': 28.0, 'tp': 61.0, 'fitness': 0.6853932584269663}
 Patterns:
 [u'todo' u'fixme' u'hack' u'workaround']
 Precisions on training set:
 {u'fixme': 0.8616187989556136, u'todo': 0.8716783216783217, u'workaround': 0.9387755102040817, u'hack': 0.925531914893617}
 ```
 - Test Easy on every target project, save output as [step1_Easy_original.csv](https://github.com/ai-se/tech-debt/tree/master/results/step1_Easy_original.csv):
 ```
 src$ python main.py Easy_results original
 ```
 - Test MAT on every target project, save output as [step1_MAT_original.csv](https://github.com/ai-se/tech-debt/tree/master/results/step1_MAT_original.csv):
 ```
 src$ python main.py MAT_results original
 ```
#### Can the ground truth be wrong?
 - Find conflicting labels (GT=no AND Easy=yes), save as csv files under the [conflicts](https://github.com/ai-se/tech-debt/tree/master/new_data/conflicts) directory:
 ```
 src$ python main.py validate_ground_truth
 ```
 - Validate the conflicting labels manually, results are under the [validate](https://github.com/ai-se/tech-debt/tree/master/new_data/validate) directory.
 - Summarize validation results and save as [validate_sum.csv](https://github.com/ai-se/tech-debt/tree/master/results/validate_sum.csv):
 ```
 src$ python main.py summarize_validate
 ```
 - Correct ground truth labels with the validation results, new data saved under [corrected](https://github.com/ai-se/tech-debt/tree/master/new_data/corrected) directory:
 ```
 src$ python main.py correct_ground_truth
 ```
 - Test Easy on every target project with corrected labels, save output as [step1_Easy_corrected.csv](https://github.com/ai-se/tech-debt/tree/master/results/step1_Easy_corrected.csv), also output the data with the "easy to find" SATDs removed to the [rest](https://github.com/ai-se/tech-debt/tree/master/new_data/rest) directory:
 ```
 src$ python main.py Easy_results corrected
 ```
 - Test MAT on every target project with corrected labels, save output as [step1_MAT_corrected.csv](https://github.com/ai-se/tech-debt/tree/master/results/step1_MAT_corrected.csv):
 ```
 src$ python main.py MAT_results corrected
 ```
### RQ2: How to better find the "hard to find" SATDs with less human effort in Step 2?
 - Test Hard, TM, and other supervised learners on every target project with "easy to find" SATDs removed, save results (rest_\*.csv) to the [results](https://github.com/ai-se/tech-debt/tree/master/results/) directory, and dump results as [rest_result.pickle](https://github.com/ai-se/tech-debt/tree/master/dump/rest_result.pickle):
 ```
 src$ python main.py rest_results
 ```
 - Plot recall-cost curves of Step2 experiments to [figures_rest](https://github.com/ai-se/tech-debt/tree/master/figures_rest) directory:
 ```
 src$ python main.py plot_recall_cost rest
 ```
#### When to stop Hard in Step 2?
 - Use estimator to stop at 90% recall, plot curves to [figures_est](https://github.com/ai-se/tech-debt/tree/master/figures_est) directory:
 ```
 src$ python main.py estimate_results
 ```
### RQ3: Overall how does Jitterbug perform?
 - Test Jitterbug, Easy+RF, Hard, MAT+RF, TM, RF on every target project, save APFD results as [overall_APFD.csv](https://github.com/ai-se/tech-debt/tree/master/results/overall_APFD.csv), AUC results as [overall_AUC.csv](https://github.com/ai-se/tech-debt/tree/master/results/overall_AUC.csv), and dump results as [overall_result.pickle](https://github.com/ai-se/tech-debt/tree/master/dump/overall_result.pickle):
 ```
 src$ python main.py overall_results
 ```
 - Plot overall recall-cost curves to [figures_overall](https://github.com/ai-se/tech-debt/tree/master/figures_overall) directory:
 ```
 src$ python main.py plot_recall_cost overall
 ```
 
 
