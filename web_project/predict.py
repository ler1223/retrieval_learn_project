import pandas as pd
import numpy as np
import string
import os
from rank_bm25 import BM25Okapi
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens


def normalize_scores(scores):
    scores_array = np.array(scores)
    if len(scores_array) == 0:
        return scores_array

    min_score = np.min(scores_array)
    max_score = np.max(scores_array)

    if max_score == min_score:
        return np.full_like(scores_array, 0.5)
    return (scores_array - min_score) / (max_score - min_score)


def assign_relevance_batch(scores_name, scores_main_cat, scores_sub_cat, ratings):
    text_scores = 0.5 * scores_name + 0.2 * scores_main_cat + 0.2 * scores_sub_cat
    quality_scores = 0.1 * (ratings / 100.0)
    total_scores = text_scores + quality_scores

    conditions = [
        total_scores >= 0.65,
        (total_scores >= 0.3) & (total_scores < 0.65),
        total_scores < 0.3
    ]
    choices = [2, 1, 0]  # 2=Полностью релевантен, 1=Частично, 0=Не релевантен

    # return np.select(conditions, choices)
    return total_scores


class Model:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.df = pd.read_csv("data.csv")

            self.model = nn.Sequential(
                nn.Linear(4, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 3)
            )
            self.model.load_state_dict(torch.load("model_state_dict.pth", weights_only=True))
            self.model.eval()
            print("Model loaded successfully!")

            tokenized_products = []
            for tokens_str in self.df['tokenized_name']:
                if isinstance(tokens_str, str):
                    tokenized_products.append(eval(tokens_str))
                else:
                    tokenized_products.append([])

            tokenized_main_category = []
            for tokens_str in self.df['tokenized_main_category']:
                if isinstance(tokens_str, str):
                    tokenized_main_category.append(eval(tokens_str))
                else:
                    tokenized_main_category.append([])

            tokenized_sub_category = []
            for tokens_str in self.df['tokenized_sub_category']:
                if isinstance(tokens_str, str):
                    tokenized_sub_category.append(eval(tokens_str))
                else:
                    tokenized_sub_category.append([])
            self.bm25_name = BM25Okapi(tokenized_products)
            self.bm25_main_category = BM25Okapi(tokenized_main_category)
            self.bm25_sub_category = BM25Okapi(tokenized_sub_category)

            self._initialized = True

    def predict(self, query, mode="cpu", n_best=1000):
        tokenized_query = preprocess_text(query)
        # print(tokenized_query)

        scores_name = self.bm25_name.get_scores(tokenized_query)
        scores_main_cat = self.bm25_main_category.get_scores(tokenized_query)
        scores_sub_cat = self.bm25_sub_category.get_scores(tokenized_query)

        score_name_norm = normalize_scores(scores_name)
        score_main_cat_norm = normalize_scores(scores_main_cat)
        score_sub_cat_norm = normalize_scores(scores_sub_cat)
        ratings = self.df['ratings'].values

        features = np.column_stack([score_name_norm, score_main_cat_norm, score_sub_cat_norm, ratings])

        labels = assign_relevance_batch(score_name_norm, score_main_cat_norm, score_sub_cat_norm, ratings)

        top_indices = np.argsort(labels)[::-1][:n_best]
        features_top1000 = features[top_indices]
        result_df = self.df.iloc[top_indices].copy()
        result_df['ranking_score'] = labels[top_indices]

        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_top1000).to(mode)
            outputs = self.model(features_tensor).cpu()
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        result_df['predicted_class'] = predictions.numpy()
        result_df['predicted_prob_class_0'] = probabilities.numpy()[:, 0]
        result_df['predicted_prob_class_1'] = probabilities.numpy()[:, 1]
        result_df['predicted_prob_class_2'] = probabilities.numpy()[:, 2]
        result_df['ranking_score'] = result_df['predicted_prob_class_2'] + result_df['predicted_prob_class_1'] - result_df['predicted_prob_class_0']

        # return result_df.sort_values('ranking_score', ascending=False)
        # top_results = result_df.sort_values('ranking_score', ascending=False).head(5)
        top_results = result_df.sort_values('ranking_score', ascending=False).head(5)
        products = []

        for idx, row in top_results.iterrows():
            product = {
                "id": idx + 1,
                "name": row['name'],
                "price": f"{row['discount_price']} ₽" if pd.notna(
                    row['discount_price']) else f"{row['actual_price']} ₽",
                "rating": float(row['ratings']) if pd.notna(row['ratings']) else 0.0,
                "image": row['image'],
                "description": f"{row['main_category']} - {row['sub_category']}"
            }
            products.append(product)
        return products
