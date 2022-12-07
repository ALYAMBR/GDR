import os
import shutil
import pandas as pd
from simple_benchmark import benchmark_gdr
import clean_experiment_data
from experiment_config import experiment_config as config
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


# the second tuple element - target column name
all_datasets = [
    ('bank_marketing.csv', 'Class'),
    ('bioresponse.csv', 'target'),
    ('blood_trans.csv', 'Class'),
    ('breast_cancer_wisconsin.csv', 'diagnosis'),
    ('christine.csv', 'class'),
    ('credit_g.csv', 'class'),
    ('diabetes.csv', 'Outcome'),
    ('guillermo.csv', 'class'),
    ('hyperplane.csv', 'class'),
    ('ionosphere.csv', 'class'),
    ('madelon.csv', 'Class'),
    ('sonar.csv', 'Class'),
]

save_transformed_data = config.get('save_transformed_data')
save_fitness_diagrams = config.get('save_fitness_diagrams')
save_time_logs = config.get('save_time_logs')
save_feature_viz = config.get('save_feature_viz')
save_loss_curves = config.get('save_loss_curves')
num_folds = config.get('folds')
datasets_list = config.get('datasets')
original_experiment_name = config.get('experiment_name')

experiment_name = original_experiment_name

clean_experiment_data.clean_experiment_data(experiment_name,
                                            save_transformed_data,
                                            save_fitness_diagrams,
                                            save_time_logs,
                                            save_feature_viz,
                                            save_loss_curves)

shutil.copyfile("gdr/experiment_config.py",
                f"experiments/{experiment_name}/experiment_config.py")

answer_df = pd.DataFrame().reset_index()

for dataset in datasets_list:
    for fold in range(num_folds):
        random_state = config.get('random_state') + fold
        pre_metrics, post_metrics = benchmark_gdr(
                        dataset_name=dataset[0],
                        target_name=dataset[1],
                        fold=fold,
                        random_state=random_state,
                        with_original_features=config.get('merge_mode'),
                        generate_features=config.get('generate_features'),
                        generations_number=config.get('generations'),
                        population_size=config.get('population'),
                        feature_select_number=config.get('base_feats'),
                        terminal_selection=config.get('terminal_selection'),
                        mutation_prob=config.get('mutation_prob'),
                        crossover_prob=config.get('crossover_prob'),
                        tournsize=config.get('tournsize'),
                        sample_size=config.get('sample_size'),
                        head_length=config.get('head_len'),
                        experiment_name=experiment_name,
                        save_transformed_data=save_transformed_data,
                        save_fitness_diagrams=save_fitness_diagrams,
                        save_time_logs=save_time_logs,
                        save_feature_viz=save_feature_viz,
                        save_loss_curves=save_loss_curves,
                        )
        print(f'Dataset {dataset[0]} fold #{fold+1}')
        for model, metrics in pre_metrics.items():
            print(f'\t\t\t{model} logloss before dim. red.:',
                  f' {metrics.get("PreLogLoss")},',
                  f'rocauc: {metrics.get("PreRocAuc")}')
        print('-' * 72)
        for model, metrics in post_metrics.items():
            print(f'\t\t\t{model} logloss after dim. red.:',
                  f' {metrics.get("PostLogLoss")},',
                  f' rocauc: {metrics.get("PostRocAuc")}')

        for model in pre_metrics.keys():
            temp_dict = dict({'Dataset': dataset[0],
                              'Fold': fold+1,
                              'Model': model})
            temp_dict.update(pre_metrics.get(model))
            temp_dict.update(post_metrics.get(model))
            answer_df = answer_df.append(temp_dict, ignore_index=True)
        answer_df.drop('index', axis=1).to_csv(
            f'experiments/{experiment_name}/temp_results_gdr.csv')

    print(answer_df)

answer_df.to_csv(f'experiments/{experiment_name}/results_gdr.csv')
os.remove(f'experiments/{experiment_name}/temp_results_gdr.csv')

print(42*"-")
print("Experiment is finished!",
      " Pack it into archive and send to Nikita. Thank you!")
print(42*"-")
