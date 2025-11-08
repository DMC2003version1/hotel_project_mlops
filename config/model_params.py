from scipy.stats import randint, uniform

LIGHTGBM_PARAMS = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 100),
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'min_data_in_leaf': randint(10, 50),
    'min_gain_to_split': uniform(0.0, 0.1)
}


RANDOM_FOREST_PARAMS = {
    'n_iter': 4,
    'cv': 3,
    'n_jobs': -1,
    'verbose': 2,
    'random_state': 42,
    'scoring': 'accuracy'
}