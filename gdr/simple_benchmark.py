from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import config_preprocessing as config_preprocessing
from gdr import GDR


def benchmark_gdr(dataset_name,
                  target_name,
                  fold,
                  random_state,
                  with_original_features,
                  generate_features,
                  generations_number,
                  population_size,
                  feature_select_number,
                  terminal_selection,
                  mutation_prob,
                  crossover_prob,
                  tournsize,
                  sample_size,
                  head_length,
                  experiment_name,
                  save_transformed_data,
                  save_fitness_diagrams,
                  save_time_logs,
                  save_feature_viz,
                  save_loss_curves
                  ):
    data = pd.read_csv(f"data/{dataset_name}")
    data_X = data.drop(columns=[target_name], axis=1)
    data_y = data[target_name]

    dataset_metadata = config_preprocessing.config_preprocessing.get(
                                                            dataset_name)
    if dataset_metadata is None:
        numerical_feature_names = data_X.columns
        categorical_feature_names = []
    else:
        numerical_feature_names = dataset_metadata.get('numericals')
        categorical_feature_names = dataset_metadata.get('non_numericals')

    for cat_feat in categorical_feature_names:
        data_X[cat_feat] = LabelEncoder().fit_transform(
                            data_X[cat_feat])
    for num_feat in numerical_feature_names:
        data_X[num_feat] = StandardScaler().fit_transform(
                            data_X[num_feat].to_numpy().reshape(-1, 1))

    data_y = LabelEncoder().fit_transform(data_y)

    train_X, test_X, train_y, test_y = train_test_split(
                                        data_X,
                                        data_y,
                                        test_size=0.2,
                                        train_size=0.8,
                                        random_state=random_state)

    numerical_train_X = train_X[numerical_feature_names]
    numerical_test_X = test_X[numerical_feature_names]

    def get_scores(model,
                   train_X,
                   train_y,
                   test_X,
                   test_y,
                   prefix='') -> Tuple[Dict, Any]:
        model = model.fit(train_X, train_y)
        results = dict()

        predict_proba = model.predict_proba(test_X)

        predict_y = model.predict(test_X)

        metric_logloss = log_loss(test_y, predict_proba)
        results.update({f'{prefix}LogLoss': metric_logloss})

        metric_rocauc = roc_auc_score(test_y, predict_y)
        results.update({f'{prefix}RocAuc': metric_rocauc})

        metric_acc = accuracy_score(test_y, predict_y)
        results.update({f'{prefix}Acc': metric_acc})

        metric_balanced_acc = balanced_accuracy_score(test_y, predict_y)
        results.update({f'{prefix}BalAcc': metric_balanced_acc})

        metric_f1 = f1_score(test_y, predict_y)
        results.update({f'{prefix}F1': metric_f1})

        metric_precision = precision_score(test_y, predict_y)
        results.update({f'{prefix}Prec': metric_precision})

        metric_recall = recall_score(test_y, predict_y)
        results.update({f'{prefix}Recall': metric_recall})

        return results, predict_proba

    pre_metrics = dict()
    lgbm = LGBMClassifier()
    pre_metrics.update({'lgbm': get_scores(
                                lgbm,
                                train_X,
                                train_y,
                                test_X,
                                test_y,
                                'Pre')[0]})
    logreg = LogisticRegression()
    pre_metrics.update({'logreg': get_scores(
                                logreg,
                                train_X,
                                train_y,
                                test_X,
                                test_y,
                                'Pre')[0]})
    dt = DecisionTreeClassifier()
    pre_metrics.update({'dt': get_scores(
                                dt,
                                train_X,
                                train_y,
                                test_X,
                                test_y,
                                'Pre')[0]})
    rf = RandomForestClassifier()
    pre_metrics.update({'rf': get_scores(
                                rf,
                                train_X,
                                train_y,
                                test_X,
                                test_y,
                                'Pre')[0]})
    knn = KNeighborsClassifier()
    pre_metrics.update({'knn': get_scores(
                                knn,
                                train_X,
                                train_y,
                                test_X,
                                test_y,
                                'Pre')[0]})

    gdr = GDR(generations_number=generations_number,
              population_size=population_size,
              generate_features_num=generate_features,
              feature_select_number=feature_select_number,
              head_length=head_length,
              terminal_selection=terminal_selection,
              mutation_prob=mutation_prob,
              crossover_prob=crossover_prob,
              tournsize=tournsize,
              sample_size=sample_size,
              random_state=random_state)
    gdr.fit(numerical_train_X, train_y, numerical_test_X, test_y)
    gdr_train_X = gdr.transform(numerical_train_X)
    gdr_test_X = gdr.transform(numerical_test_X)

    if with_original_features == 'all':
        if len(gdr_train_X.columns) > 0:
            gdr_train_X = gdr_train_X.join(train_X.reset_index(drop=True))
            gdr_test_X = gdr_test_X.join(test_X.reset_index(drop=True))
        else:
            gdr_train_X = train_X
            gdr_test_X = test_X
        print(gdr_train_X.head())
        print(gdr_test_X.head())
    elif with_original_features == 'categoricals':
        if len(gdr_train_X.columns) > 0:
            gdr_train_X = gdr_train_X.join(
                train_X[categorical_feature_names].reset_index(drop=True))
            gdr_test_X = gdr_test_X.join(
                test_X[categorical_feature_names].reset_index(drop=True))
        else:
            gdr_train_X = train_X[categorical_feature_names]
            gdr_test_X = test_X[categorical_feature_names]
        print(gdr_train_X.head())
        print(gdr_test_X.head())

    if save_transformed_data:
        transformed_original_data = gdr.transform(data_X)
        transformed_original_data['target'] = data_y
        save_data_path = 'experiments/'
        save_data_path += f'{experiment_name}/'
        save_data_path += 'transformed_data/'
        save_data_path += f'{dataset_name[:-4]}_{fold}.csv'
        transformed_original_data.to_csv(
            save_data_path,
            index=False)  # slice in dataset name removes '.csv'

    post_metrics = dict()

    lgbm = LGBMClassifier()
    logreg = LogisticRegression()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier()

    post_metrics.update({'lgbm': get_scores(
     lgbm,
     gdr_train_X,
     train_y,
     gdr_test_X,
     test_y,
     'Post')[0]})
    post_metrics.update({'logreg': get_scores(
     logreg,
     gdr_train_X,
     train_y,
     gdr_test_X,
     test_y,
     'Post')[0]})
    post_metrics.update({'dt': get_scores(
     dt,
     gdr_train_X,
     train_y,
     gdr_test_X,
     test_y,
     'Post')[0]})
    post_metrics.update({'rf': get_scores(
     rf,
     gdr_train_X,
     train_y,
     gdr_test_X,
     test_y,
     'Post')[0]})
    post_metrics.update({'knn': get_scores(
     knn,
     gdr_train_X,
     train_y,
     gdr_test_X,
     test_y,
     'Post')[0]})

    if not gdr._is_id:
        if save_feature_viz:
            gdr.save_hof_visualization(
                f'experiments/{experiment_name}/pics/',
                dataset_name=dataset_name[:-4],
                fold=fold)
        if save_fitness_diagrams:
            gdr.save_logs_visualization(
                f'experiments/{experiment_name}/diagrams/',
                dataset_name=dataset_name[:-4],
                fold=fold)
        if save_time_logs:
            gdr.save_time_logs(
                f'experiments/{experiment_name}/time_logs/',
                dataset_name=dataset_name[:-4],
                fold=fold)
        if save_loss_curves:
            gdr.save_loss_curves(
                f'experiments/{experiment_name}/loss_curves/',
                dataset_name=dataset_name[:-4],
                fold=fold)
    return pre_metrics, post_metrics
