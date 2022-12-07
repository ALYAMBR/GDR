import os
import shutil


def clean_experiment_data(experiment_name,
                          save_transformed_data,
                          save_fitness_diagrams,
                          save_time_logs,
                          save_feature_viz,
                          save_loss_curves):
    shutil.rmtree(f"experiments/{experiment_name}", ignore_errors=True)
    os.mkdir(f"experiments/{experiment_name}")
    if save_fitness_diagrams:
        os.mkdir(f"experiments/{experiment_name}/diagrams")
    if save_feature_viz:
        os.mkdir(f"experiments/{experiment_name}/pics")
    if save_time_logs:
        os.mkdir(f"experiments/{experiment_name}/time_logs")
    if save_transformed_data:
        os.mkdir(f"experiments/{experiment_name}/transformed_data")
    if save_loss_curves:
        os.mkdir(f"experiments/{experiment_name}/loss_curves")
