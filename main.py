import pandas as pd
import numpy as np
import dask.dataframe as dd
import dill
import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

from classes import SimpleAE, SimpleTensorDataset



def delete_cols(df):
    columns_to_drop = ['pre_loans_total_overdue', 'pre_loans5', 'pre_loans530', 'pre_loans3060',
                       'pre_loans6090', 'pre_loans90', 'enc_loans_account_cur']
    df = df.drop(columns=columns_to_drop, axis=1).copy()
    return df



class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.target_maps = {}

    def fit(self, X, y=None):
        df = X.copy()
        features = [col for col in df.columns if col not in ['id', 'rn', 'flag']]
        for column in features:
            self.target_maps[column] = df.groupby(column)['flag'].mean().to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        for column, mapping in self.target_maps.items():
            df[column] = df[column].map(mapping).fillna(0.0)
        return df


class AutoEncoderEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, T_max=15, latent_dim=48, batch_size=512):
        self.T_max = T_max
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_cols = None
        self.model_weights = None
        self.model_class = SimpleAE
        self._SimpleTensorDataset = SimpleTensorDataset
        self._DataLoader = DataLoader
        self.model_path = 'autoencoder_model_weights.pt'
        self._torch = torch
        self._np = np
        self._pd = pd

    def _prepare_data_by_id(self, df):
        data = {}
        for id_, group in df.groupby("id"):
            X = group.sort_values("rn")[self.feature_cols].values
            T, D = X.shape
            if T >= self.T_max:
                X = X[:self.T_max]
            else:
                pad = self._np.zeros((self.T_max - T, D))
                X = self._np.vstack([X, pad])
            data[id_] = X
        return data

    def fit(self, X, y=None):
        self.feature_cols = [col for col in X.columns if col not in ["id", "rn", "flag"]]
        input_dim = self.T_max * len(self.feature_cols)
        self.model = self.model_class(input_dim=input_dim, latent_dim=self.latent_dim).to(self.device)

        prepared_data = self._prepare_data_by_id(X)
        dataset = self._SimpleTensorDataset(prepared_data)
        loader = self._DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = self._torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = self._torch.nn.MSELoss()

        for epoch in range(10):
            self.model.train()
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                z, x_hat = self.model(x_batch)
                loss = loss_fn(x_hat, x_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        self.model_weights = self.model.state_dict()
        self._torch.save(self.model.state_dict(), self.model_path)

        return self

    def transform(self, X):
        if self.feature_cols is None or self.model_weights is None:
            raise ValueError("Model has not been fitted!")

        self.model = self.model_class(input_dim=self.T_max * len(self.feature_cols), latent_dim=self.latent_dim).to(self.device)
        self.model.load_state_dict(self._torch.load(self.model_path))
        self.model.eval()

        prepared_data = self._prepare_data_by_id(X)
        dataset = self._SimpleTensorDataset(prepared_data)
        loader = self._DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        z_list = []
        id_list = []

        with self._torch.no_grad():
            for x_batch, id_batch in loader:
                x_batch = x_batch.to(self.device)
                z, _ = self.model(x_batch)
                z_list.append(z.cpu())
                id_list.extend(id_batch)

        z_all = self._torch.cat(z_list).numpy()
        df_z = self._pd.DataFrame(z_all, columns=[f"z_{i}" for i in range(self.latent_dim)])
        df_z["id"] = id_list


        df_rn = self._pd.DataFrame(X.groupby('id')['rn'].max().reset_index())
        df_z['id'] = df_z['id'].astype(int)
        df_z = self._pd.merge(df_z, df_rn, how='left', on='id')

        if 'flag' in X.columns:
            df_flag = self._pd.DataFrame(X.groupby('id')['flag'].max().reset_index())
            df_z = self._pd.merge(df_z, df_flag, how='left', on='id')
        else:
            print("The 'flag' column is missing, skipping it.")

        return df_z



class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features_inp = [
            'pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm', 'pre_till_pclose',
            'pre_till_fclose', 'pre_loans_credit_limit', 'pre_loans_next_pay_summ', 'pre_loans_outstanding',
            'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate', 'is_zero_loans5', 'is_zero_loans530',
            'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90', 'pre_util', 'pre_over2limit',
            'pre_maxover2limit', 'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit', 'enc_paym_0',
            'enc_paym_1', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24', 'enc_loans_account_holder_type',
            'enc_loans_credit_status', 'enc_loans_credit_type', 'pclose_flag', 'fclose_flag'
        ]
        self.all_columns = None
        self._pd = pd

    def fit(self, X, y=None):
        df = X.copy()
        df_for_freqs = self._pd.DataFrame(df[['id']].drop_duplicates())

        for name in self.features_inp:
            df_freq = df.groupby(['id', name]).size().reset_index(name='cnt')
            df_freq['total'] = df_freq.groupby('id')['cnt'].transform('sum')
            df_freq['percent'] = (df_freq['cnt'] / df_freq['total']).round(3).astype('float32')
            df_pivot = df_freq.pivot(index='id', columns=name, values='percent').fillna(0)
            df_pivot.columns = [f"{name}_{col}" for col in df_pivot.columns]
            df_for_freqs = self._pd.merge(df_for_freqs, df_pivot, on='id', how='left')

        self.all_columns = set(df_for_freqs.columns)
        return self

    def transform(self, X):
        df = X.copy()
        df_for_freqs = self._pd.DataFrame(df[['id']].drop_duplicates())

        for name in self.features_inp:
            df_freq = df.groupby(['id', name]).size().reset_index(name='cnt')
            df_freq['total'] = df_freq.groupby('id')['cnt'].transform('sum')
            df_freq['percent'] = (df_freq['cnt'] / df_freq['total']).round(3).astype('float32')
            df_pivot = df_freq.pivot(index='id', columns=name, values='percent').fillna(0)
            df_pivot.columns = [f"{name}_{col}" for col in df_pivot.columns]
            df_for_freqs = self._pd.merge(df_for_freqs, df_pivot, on='id', how='left')

        missing_cols = self.all_columns - set(df_for_freqs.columns)
        for col in missing_cols:
            df_for_freqs[col] = 0.0

        return df_for_freqs



class MergePipelines(BaseEstimator, TransformerMixin):
    def __init__(self, pipe1, pipe2):
        self.pipe1 = pipe1
        self.pipe2 = pipe2
        self._pd = pd
    def fit(self, X, y=None):
        self.pipe1.fit(X, y)
        self.pipe2.fit(X, y)
        return self
    def transform(self, X):
        df1 = self.pipe1.transform(X)
        df2 = self.pipe2.transform(X)
        df_merged = self._pd.merge(
            df1,
            df2.drop(columns=[col for col in df2.columns if col in df1.columns and col != 'id'], errors='ignore'),
            on='id',
            how='left'
        )
        df_merged = df_merged.drop(columns='id')
        return df_merged



class PipelineDefaultPrediction:
    def __init__(self, preprocessor, model, target_column='flag', id_column='id'):
        self.preprocessor = preprocessor
        self.model = model
        self.target_column = target_column
        self.id_column = id_column

    def fit(self, X_raw, y_raw):
     
        X_processed = self.preprocessor.fit_transform(X_raw)

        print (X_processed.columns)

        X_final = X_processed.drop(columns=[self.target_column])
        y_final = X_processed[self.target_column]

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        roc_auc_scores = []
        n_splits = kf.get_n_splits(X_final, y_final)

        for fold_num, (train_index, val_index) in enumerate(kf.split(X_final, y_final)):
            X_train, X_test = X_final.iloc[train_index], X_final.iloc[val_index]
            y_train, y_test = y_final.iloc[train_index], y_final.iloc[val_index]

            X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
                X_train, y_train, random_state=42, stratify=y_train
            )

            self.model.fit(X_train_val, y_train_val, eval_set=(X_test_val, y_test_val))

            y_pred = self.model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            roc_auc_scores.append(auc)
            print(f"Fold {fold_num + 1} AUC: {auc:.4f}")

            if fold_num == n_splits - 1:
                results_df = pd.DataFrame({
                    "true_flag": y_test.values,
                    "proba": y_pred
                }).sort_values(by="proba", ascending=False).reset_index(drop=True)
                results_df.to_csv("predictions.csv", index=False)

        print(f'Mean AUC: {np.mean(roc_auc_scores):.4f}, Std: {np.std(roc_auc_scores):.4f}')

        self.model.fit(X_final, y_final)
        return self


def main():
    data_path = '/your_path/'
    taget_path = '/your_path/'
    data_train = dd.read_parquet(data_path)
    data_train = data_train.compute()
    data_target = pd.read_csv(taget_path)
    full_data = data_train.merge(data_target, how='left', on='id')


    pipe1 = Pipeline(steps=[
        ('delete_cols', FunctionTransformer(delete_cols)),
        # ('target_encoding', FunctionTransformer(target_encoding)),
        ('target_encoding', TargetEncoder()),
        # ('embeddings', FunctionTransformer(embeddings)),
        ('embeddings', AutoEncoderEmbedding()),

    ])
    pipe2 = Pipeline(steps=[
        # ('frequency', FunctionTransformer(frequency)),
        ('frequency', FrequencyEncoder()),
    ])

    preprocessor = MergePipelines(pipe1, pipe2)

    model = CatBoostClassifier(
        bagging_temperature=0.9699098521619943,
        border_count=236,
        depth=9,
        iterations=1000,
        l2_leaf_reg=2,
        learning_rate=0.06454749016213018,
        random_strength=0.18340450985343382,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42
    )

    # sampled_ids = data_target['id'].sample(frac=0.0001, random_state=42)
    #
    # small_sample_data = full_data[full_data['id'].isin(sampled_ids)]
    # small_sample_target = data_target[data_target['id'].isin(sampled_ids)]
    #
    # main_pipe = PipelineDefaultPrediction(preprocessor, model)
    # main_pipe.fit(small_sample_data, small_sample_target)

    main_pipe = PipelineDefaultPrediction(preprocessor, model)
    main_pipe.fit(full_data, data_target)


    with open('default_prediction_model.pkl', 'wb') as file:
        dill.dump({
            'model': main_pipe,
            'metadata': {
                'name': 'default prediction',
                'author': 'Artjom Zaicev',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(main_pipe).__name__,
                # 'ROC AUC score': np.mean(roc_auc_scores)
            }
        }, file)

if __name__ == '__main__':
    main()