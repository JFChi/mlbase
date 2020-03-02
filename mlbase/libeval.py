import json
import os
import sys
import glob
import yaml

from typing import Set, List

from joblib import Parallel, delayed

from collections import Counter

from tqdm import tqdm
from tqdm import tqdm_notebook
from termcolor import cprint

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sn

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.utils import shuffle

from msbase.utils import load_json, sha256sum

import pickle

precision_recall_fscore_cols = ['precision', 'recall', 'fscore', 'support']

class Configuration(object):
    def __init__(self):
        self.labels = None
        self.vectors_dir = None

        self.model = "rf" # default: random forest
        self.gamma_svc = "auto"
        self.random_state = 37
        self.linearsvc_C = 1

        self.output_suffix = None
        self.secondary_conf = None

    def get_labels_index(self):
        return list(range(len(self.labels)))

    def get_output_suffix(self):
        return self.output_suffix

    def load_vectors(self):
        self.DX = [] # features
        self.DY = [] # label
        self.DEXTRA = [] # extra info

        with open(self.vectors_dir + '/DX.pickle', "rb") as f:
            DX = pickle.load(f)
        with open(self.vectors_dir + '/DY.pickle', "rb") as f:
            DY = pickle.load(f)
        with open(self.vectors_dir + '/DEXTRA.pickle', "rb") as f:
            DEXTRA = pickle.load(f)

        label_index = { label: i for i, label in enumerate(self.labels) }

        for i, X in enumerate(DX):
            label = DY[i]
            extra = DEXTRA[i]
            self.DX.append(X)
            self.DY.append(label_index[label])
            self.DEXTRA.append(extra)

        self.DY = np.array(self.DY)
        self.DEXTRA = np.array(self.DEXTRA)
        if isinstance(self.DX[0], dict):
            print("Use DictVectorizer")
            v = DictVectorizer(sparse=False)
            self.DX = v.fit_transform(self.DX)
            self.feature_names = v.get_feature_names()
        elif isinstance(self.DX[0], str):
            print("Use TfidfVectorizer")
            v = TfidfVectorizer(tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
            self.DX = v.fit_transform(self.DX).todense()
            self.feature_names = v.get_feature_names()
        elif isinstance(self.DX[0], np.ndarray):
            print("Use arrays directly")
            self.DX = np.array(self.DX)
            with open(self.vectors_dir + '/features.pickle', "rb") as f:
                self.feature_names = pickle.load(f)
        self.DY_EXTRAs = pd.DataFrame([[self.labels[y] for y in self.DY], self.DEXTRA], index=["true-label", "EXTRA"]).T

        n_samples, n_dim = self.DX.shape
        print("Number of samples: %s" % n_samples)
        print("Number of features: %s" % n_dim)
        print(self.DY_EXTRAs.groupby("true-label").count())

    def precision_recall_fscore_df(self, y_true, y_pred):
        return pd.DataFrame(list(precision_recall_fscore_support(y_true, y_pred, labels=self.get_labels_index())),
                            index=precision_recall_fscore_cols, columns=self.labels).T

    def run_eval_repeated(self, N: int, n_fold: int = 10):
        cprint("repeated cross validation", "blue")

        eval_result_all = []
        for _ in range(N):
            _, y_true, y_pred, _, _ = self.avg_eval(n_fold=n_fold)
            eval_result_all.append(self.precision_recall_fscore_df(y_true, y_pred))

        return (sum(eval_result_all) / len(eval_result_all)).round(2)

    def run_eval(self, MAX_FEAT_IMP=10):
        result = self.avg_eval()
        feature_importances_s = result["feature_importances_s"]
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        EXTRAs_test = result["EXTRAs_test"]

        eval_result = self.precision_recall_fscore_df(y_true, y_pred)

        if len(feature_importances_s) and feature_importances_s[0] is not None and not feature_importances_s[0].empty:
            fi_sum = feature_importances_s[0][:MAX_FEAT_IMP]
            for j in range(1, len(feature_importances_s)):
                fi_sum = feature_importances_s[j][:MAX_FEAT_IMP].add(fi_sum, fill_value=0)
            fi_sum = fi_sum.sort_values(by="importance", ascending=False)
        else:
            fi_sum = None

        return dict(result, **{
               "eval_result": eval_result,
               "feature_importances": fi_sum,
        })

    def heatmap(self, eval_results, output_prefix: str):
        y_true = eval_results["y_true"]
        y_pred = eval_results["y_pred"]
        size_array = np.zeros(len(self.get_labels_index()))
        for k, v in Counter(y_true).items():
            size_array[k] = v
        confusion_mat = confusion_matrix(y_true, y_pred, labels=self.get_labels_index())
        mat = confusion_mat.T
        sn.heatmap((mat / size_array),
                xticklabels=self.labels,
                yticklabels=self.labels,
                cmap="YlGnBu")
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.title("Classification Result Heatmap")
        plt.tight_layout()
        plt.savefig(output_prefix + "prediction-heatmap" + self.get_output_suffix() + ".pdf")

    def analyze_feature(self, feature_name: str, label: str):
        feat_i = self.feature_names.index(feature_name)
        positive_features = self.DX[self.DY == self.labels.index(label), feat_i]
        negative_features = self.DX[self.DY != self.labels.index(label), feat_i]
        print("%.2f of %s samples has %s" % (positive_features.mean(), label, feature_name))
        print("%.2f of non-%s samples has %s" % (negative_features.mean(), label, feature_name))
        positive_samples = self.DEXTRA[self.DY == self.labels.index(label)]
        negative_samples = self.DEXTRA[self.DY != self.labels.index(label)]
        neg_label_samples = positive_samples[positive_features == 0]
        pos_non_label_samples = negative_samples[negative_features >= 1]
        return neg_label_samples, pos_non_label_samples

    def select_EXTRAs(self, EXTRAs: Set[str]):
        return np.isin(self.DEXTRA, EXTRAs)

    def unique_EXTRAs(self):
        return np.array(list(set(self.DEXTRA)))

    def avg_eval(self, n_fold: int = 10):
        feature_importances_s = []
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        EXTRAs_test_all = []

        cprint("cross validation: train/test = %s/%s" % (n_fold - 1, n_fold), "blue")
        kf = KFold(n_splits=n_fold, shuffle=True)
        EXTRAs = self.unique_EXTRAs()
        results = Parallel(n_jobs=-1)(delayed(self.classify_fold)(train_EXTRAs=EXTRAs[train_index],
                                                                  test_EXTRAs=EXTRAs[test_index])
                                      for train_index, test_index in kf.split(EXTRAs))

        for result in results:
            feature_importances_s.append(result["feature_importances"])
            y_true_all.append(result["y_true"])
            y_pred_all.append(result["y_pred"])
            if result["y_pred_proba"] is not None:
                y_pred_proba_all.append(result["y_pred_proba"])
            EXTRAs_test_all += list(result["test_EXTRAs"])

        if y_pred_proba_all:
            y_pred_proba_all = np.concatenate(y_pred_proba_all)
        else:
            y_pred_proba_all = None

        return {
            "feature_importances_s": feature_importances_s,
            "y_true": np.concatenate(y_true_all),
            "y_pred": np.concatenate(y_pred_all),
            "y_pred_proba": y_pred_proba_all,
            "EXTRAs_test": EXTRAs_test_all
        }

    def classify_fold(self, train_EXTRAs, test_EXTRAs):
        return dict(self.classify(train_EXTRAs, test_EXTRAs), test_EXTRAs=test_EXTRAs)

    def classify_random_split(self, test_size):
        EXTRAs = self.unique_EXTRAs()
        train_EXTRAs, test_EXTRAs = train_test_split(EXTRAs, test_size=test_size)
        return self.classify(train_EXTRAs, test_EXTRAs)

    def eval_random_split(self, test_size=0.3):
        result = self.classify_random_split(test_size)
        return self.precision_recall_fscore_df(y_true=result["y_true"], y_pred=result["y_pred"])

    def eval_random_split_repeated(self, N: int, test_size=0.3):
        print("%s.eval_random_split_repeated: N=%s, test_size=%s" % (self.__class__, N, test_size))
        eval_result_all = Parallel(n_jobs=-1)(delayed(self.eval_random_split)(test_size=test_size) for _ in range(N))
        return (sum(eval_result_all) / len(eval_result_all)).round(2)

    def classify(self, train_EXTRAs, test_EXTRAs):
        train_idx = self.select_EXTRAs(train_EXTRAs)
        test_idx = self.select_EXTRAs(test_EXTRAs)
        train_X, train_Y, test_X, test_Y = self.DX[train_idx], self.DY[train_idx], self.DX[test_idx], self.DY[test_idx]
        if self.model == "svc":
            classifier = SVC(gamma=self.gamma_svc)
        elif self.model == "linearsvc":
            params = { 'C' : self.linearsvc_C }
            classifier = LinearSVC(**params)
        elif self.model == "svm":
            classifier = svm.SVC(kernel="rbf")
        elif self.model == "3nn":
            classifier = KNeighborsClassifier(n_neighbors=3)
        elif self.model == "rf":
            classifier = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators,
                                                class_weight="balanced",
                                                max_features=self.max_features, n_jobs=-1)
        classifier.fit(train_X, train_Y)
        pred_Y = classifier.predict(test_X)
        if self.model in ["rf"]:
            y_pred_proba = classifier.predict_proba(test_X)
            for i in range(y_pred_proba.shape[1], len(self.get_labels_index())):
                y_pred_proba = np.insert(y_pred_proba, i, 0, axis=1)
            y_pred_proba2 = y_pred_proba.copy()
            y_pred_proba2[np.arange(y_pred_proba.shape[0]), np.argmax(y_pred_proba, axis=1)] = 0
            y_pred2 = np.argmax(y_pred_proba2, axis=1)
            feature_importances = pd.DataFrame(classifier.feature_importances_,
                                    index = self.feature_names,
                                    columns=['importance']).sort_values('importance',ascending=False)
        else:
            y_pred_proba = None
            y_pred_proba2 = None
            y_pred2 = None
            feature_importances = None
        if self.secondary_conf is not None:
            result = self.secondary_conf.classify(train_EXTRAs, test_EXTRAs)
            df = pd.DataFrame([y_pred_proba.max(axis=1) - y_pred_proba2.max(axis=1),
                               result["y_pred_proba"].max(axis=1) - result["y_pred_proba2"].max(axis=1)],
                              index=["pred-proba", "pred-proba-prime"]).T
            df["pred"] = pred_Y
            df["true"] = test_Y
            df["pred-prime"] = result["y_pred"]
            indexer = df["pred-proba-prime"] > 0.8
            indexer = indexer & (df["pred-proba"] < 0.01)
            open("log.txt", "a").write(str(df.loc[indexer]) + "\n\n")
            # df.loc[indexer, "pred"] = df.loc[indexer, "pred-prime"]
            pred_Y[indexer] = result["y_pred"][indexer]
            if indexer.empty():
                assert pred_Y == df["pred"]
        return {
            "feature_importances": feature_importances,
            "y_true" : test_Y,
            "y_pred" : pred_Y,
            "y_pred2" : y_pred2,
            "y_pred_proba" : y_pred_proba,
            "y_pred_proba2" : y_pred_proba2
        }

def combine_results(good_labels, results):
    combined = pd.concat([result.rename(columns={ n: n + " (" + name + ")" for n in precision_recall_fscore_cols })
                            for name, result in results.items() ],
                            axis=1, sort=False)
    return combined.loc[good_labels]

def compare_eval_results_table(names, combined, output_suffix: str, output_prefix: str):
    combined2 = combined.copy()
    combined2.loc['mean'] = combined.mean().round(2)
    all_names = "/".join(names)
    for col in precision_recall_fscore_cols:
        s0 = combined2[col + " (" + names[0] + ")"].map(nan_str)
        for name in names[1:]:
            s0 = s0 + "/" + combined2[col + " (" + name + ")"].map(nan_str)
        combined2[col + " (" +  all_names + ")"] = s0
    combined_final = combined2[[ col + " (" + all_names + ")" for col in precision_recall_fscore_cols]].drop(columns=["support (" + all_names + ")", "fscore (" + all_names + ")"])
    open(output_prefix + "eval-results" + output_suffix + ".tex", "w").write(combined_final.to_latex())
    return combined_final

def compare_eval_results_barcharts(combined, names, all_config, output_suffix: str, alpha:float, label_suffix: str = ""):
    combined = combined.dropna()
    categories = tuple(combined.index)
    x_pos = np.arange(len(categories))

    score_sg_avgs = { name: np.average(combined["fscore (" + name + ")"]) for name in names }

    bar_width = 0.5 / len(score_sg_avgs)
    plt.xticks(x_pos, categories)

    for i, name in enumerate(score_sg_avgs.keys()):
        score_sg_avg = score_sg_avgs[name]
        plt.bar(x_pos - 0.125 + i * bar_width, combined["fscore (" + name + ")"], width=bar_width,
                color=all_config[name]["color"], align='center', hatch=all_config[name]["hatch"],
                alpha=alpha, label=all_config[name]["name"] + label_suffix)
        print("score_sg_avg: %s, name: %s" % (score_sg_avg, name))

def nan_str(x):
    ret = str(x)
    if ret == "nan":
        return "-"
    return ret

def precision_recall_fscore_df(labels: List[str], y_true, y_pred, suffix=""):
    index = precision_recall_fscore_cols
    index = [ i + suffix for i in index ]
    return pd.DataFrame(list(precision_recall_fscore_support(y_true, y_pred, labels=labels)),
                        index=index, columns=labels).T.round(3)