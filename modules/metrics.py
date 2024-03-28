import json
import os
import sqlite3
import sys
from numpy import average
import pandas as pd
from rouge_score import rouge_scorer, scoring


class CodeRouge:
    def __init__(self, rouge_types=['rougeL','rougeLsum'],) -> None:
        self.rouge_types = rouge_types
        self.rouge = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        self.scores = []
        self.avgs = {}
    
    def calc_averages(self) -> dict:
        if len(self.scores) == 0:
            raise ValueError("Can't calculate avg of emptry scores.")
        scores_pd = pd.DataFrame(self.scores)
        
        for r in self.rouge_types:
            self.avgs[f"avg_{r}"] = scoring.Score(**{
                "precision": average([x.precision for x in scores_pd[r].tolist()]),
                "recall": average([x.recall for x in scores_pd[r].tolist()]), 
                "fmeasure": average([x.fmeasure for x in scores_pd[r].tolist()])
            })
        
        return self.avgs
    
    def rouge_type_to_list(self, type) -> list:
        rouge_type_list = []
        for i, score in enumerate(self.scores):
            rouge_type_list.append(
                {
                    'dataset_id': i,
                    "type": type,
                    "precision": score[type].precision,
                    "recall": score[type].recall,
                    "fmeasure": score[type].fmeasure,
                }
            )
        return rouge_type_list
                
    def compute(self, predictions: list, references: list) -> None:
        if len(predictions) != len(references):
            raise RuntimeError("Can't compare lists with different lengths")
        for i in range(len(predictions)):
            self.scores.append(
                self.rouge.score(
                    target=references[i],
                    prediction=predictions[i],
                )
            )