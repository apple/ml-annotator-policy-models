#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from aix360.algorithms.rbm.BRCG import BRCGExplainer
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG
import pandas as pd

def convert_to_multiindex(df, include_nots=False):
    """Convert binary DataFrame to MultiIndex format for BooleanRuleCG"""
    if include_nots:
        new_columns = pd.MultiIndex.from_tuples(
            [(col, '==', "present") for col in df.columns]+[(col, '!=', "present") for col in df.columns],
            names=['feature', 'operator', 'threshold']
        )
        df_multiindex = pd.concat([df.copy(), 1-df.copy()], axis=1)
        df_multiindex.columns = new_columns
    else:
        new_columns = pd.MultiIndex.from_tuples(
            [(col, '==', "present") for col in df.columns],
            names=['feature', 'operator', 'threshold']
        )
        df_multiindex = df.copy()
        df_multiindex.columns = new_columns
    
    return df_multiindex.sort_index()

class DNF:
    def __init__(self, features, include_nots=False, **model_kwargs):
        self.features=features
        self.include_nots = include_nots
        brcg = BooleanRuleCG(
            **model_kwargs
        )
        self.clf = BRCGExplainer(brcg)
        self.props = []

    def fit(self, X, y):
        X_df = convert_to_multiindex(pd.DataFrame(X, columns=self.features), include_nots=self.include_nots)
        self.clf.fit(X_df, y)
        explanation = self.clf.explain()
        for clause in explanation["rules"]:
            terms =  [(condition.split(" == ")[0], "==", "present") for condition in clause.split(" AND ")]
            try:
                len_satisfying = (X_df[terms] == 1).all(axis=1).sum()
            except:
                print(terms)
                len_satisfying = 0
            prop = len_satisfying/len(X_df)
            self.props.append(prop)

    
    def predict(self, X):
        X_df = convert_to_multiindex(pd.DataFrame(X, columns=self.features), include_nots=self.include_nots)
        return self.clf.predict(X_df)

    def score(self, X, y):
        X_df = convert_to_multiindex(pd.DataFrame(X, columns=self.features), include_nots=self.include_nots)
        return (self.clf.predict(X_df) == y).mean()
    
    def get_string_representation(self):
        explanation = self.clf.explain()
        cleaned = []
        for i, clause in enumerate(explanation["rules"]):
            clean = []
            for condition in clause.split(" AND "):
                if condition.endswith(" != present"):
                    clean.append("!"+condition.replace(" != present", ""))
                else:
                    clean.append(condition.replace(" == present", ""))
            cleaned.append(" AND ".join(clean) + f" {self.props[i]:.1%}")
        return " OR\n".join(cleaned)

    def get_features_used(self):
        features_used = set()
        explanation = self.clf.explain()
        for clause in explanation['rules']:
            for condition in clause.split(" AND "):
                features_used.add(condition.replace(" == present", ""))
        feature_indices = []
        for feat in features_used:
            feature_indices.append(self.features.index(feat))
        return feature_indices
