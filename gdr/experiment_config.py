"""
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
"""

experiment_config = {
    'experiment_name': 'GDR_loss_curves_testrun',
    'datasets': [
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
    ],
    'folds': 10,
    'random_state': 72,
    'merge_mode': 'categoricals',  # categoricals, all
    'generate_features': 3,
    'generations': 21,
    'population': 1600,
    'terminal_selection': 'all',  # important, all, random
    'mutation_prob': 0.25,
    'crossover_prob': 0.25,
    'tournsize': 4,
    'sample_size': 8192,
    'base_feats': 8,
    'head_len': 8,
    'save_transformed_data': False,
    'save_time_logs': False,
    'save_feature_viz': True,
    'save_fitness_diagrams': False,
    'save_loss_curves': True,
}