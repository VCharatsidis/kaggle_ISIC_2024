#Best LGBM trial: FrozenTrial(number=38, state=1, values=[0.17404115603048945], datetime_start=datetime.datetime(2024, 8, 17, 16, 33, 55, 488180), datetime_complete=datetime.datetime(2024, 8, 17, 16, 34, 14, 950991), params={'lambda_l1': 0.0010065663960964343, 'lambda_l2': 0.09252696729873518, 'learning_rate': 0.016930699845968775, 'max_depth': 7, 'num_leaves': 230, 'colsample_bytree': 0.9552177571289984, 'colsample_bynode': 0.46735369119377135, 'bagging_fraction': 0.9997820711118077, 'bagging_freq': 4, 'min_data_in_leaf': 37, 'scale_pos_weight': 2.512129229660625}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'lambda_l1': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'lambda_l2': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'learning_rate': FloatDistribution(high=0.1, log=True, low=0.01, step=None), 'max_depth': IntDistribution(high=8, log=False, low=4, step=1), 'num_leaves': IntDistribution(high=256, log=False, low=16, step=1), 'colsample_bytree': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'colsample_bynode': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'bagging_fraction': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'bagging_freq': IntDistribution(high=7, log=False, low=1, step=1), 'min_data_in_leaf': IntDistribution(high=100, log=False, low=5, step=1), 'scale_pos_weight': FloatDistribution(high=4.0, log=False, low=0.8, step=None)}, trial_id=38, value=None)
#Best CatBoost trial: FrozenTrial(number=190, state=1, values=[0.17135366117624146], datetime_start=datetime.datetime(2024, 8, 17, 20, 22, 9, 335), datetime_complete=datetime.datetime(2024, 8, 17, 20, 23, 23, 668461), params={'learning_rate': 0.08740625888900079, 'max_depth': 8, 'l2_leaf_reg': 7.009298427887963, 'subsample': 0.5486542540911756, 'colsample_bylevel': 0.8564779094591841, 'min_data_in_leaf': 95, 'scale_pos_weight': 2.523027737460909}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'learning_rate': FloatDistribution(high=0.1, log=True, low=0.01, step=None), 'max_depth': IntDistribution(high=8, log=False, low=4, step=1), 'l2_leaf_reg': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'subsample': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'colsample_bylevel': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'min_data_in_leaf': IntDistribution(high=100, log=False, low=5, step=1), 'scale_pos_weight': FloatDistribution(high=4.0, log=False, low=0.8, step=None)}, trial_id=190, value=None)
#Best XGBoost trial: FrozenTrial(number=142, state=1, values=[0.17323329842181906], datetime_start=datetime.datetime(2024, 8, 17, 21, 44, 44, 972570), datetime_complete=datetime.datetime(2024, 8, 17, 21, 45, 18, 369675), params={'learning_rate': 0.04185796781487898, 'max_depth': 8, 'lambda': 0.02650828214464622, 'alpha': 0.08008440537616693, 'subsample': 0.621532884902385, 'colsample_bytree': 0.9553048021746551, 'colsample_bynode': 0.5773607670935514, 'scale_pos_weight': 3.5283758941665857}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'learning_rate': FloatDistribution(high=0.1, log=True, low=0.01, step=None), 'max_depth': IntDistribution(high=8, log=False, low=4, step=1), 'lambda': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'alpha': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'subsample': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'colsample_bytree': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'colsample_bynode': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'scale_pos_weight': FloatDistribution(high=4.0, log=False, low=0.8, step=None)}, trial_id=142, value=None)


# 836 features
#[I 2024-08-20 00:24:07,469] Trial 185 finished with value: 0.1801107965855093 and parameters:
# {'lambda_l1': 0.04148730013039774, 'lambda_l2': 0.5393934436213434, 'learning_rate': 0.050914843897483957, 'max_depth': 7, 'num_leaves': 84, 'colsample_bytree': 0.9314408907527532, 'colsample_bynode': 0.7106603583755883, 'bagging_fraction': 0.699679069417054, 'bagging_freq': 5, 'min_data_in_leaf': 22, 'scale_pos_weight': 2.7898278216285863}
#[I 2024-08-19 11:46:34,008] Trial 70 finished with value: 0.1786092266659516 and parameters: {'lambda_l1': 0.09401596685127875, 'lambda_l2': 0.005582109823099, 'learning_rate': 0.048847450637279785, 'max_depth': 6, 'num_leaves': 27, 'colsample_bytree': 0.7688795670811674, 'colsample_bynode': 0.6476025200809796, 'bagging_fraction': 0.8571714069569365, 'bagging_freq': 6, 'min_data_in_leaf': 78, 'scale_pos_weight': 1.3028488403195542}. Best is trial 70 with value: 0.1786092266659516.

# Best CatBoost trial: FrozenTrial(number=26, state=1, values=[0.17762763230066564], datetime_start=datetime.datetime(2024, 8, 19, 13, 31, 17, 605446), datetime_complete=datetime.datetime(2024, 8, 19, 13, 33, 33, 594114),
# params={'learning_rate': 0.03043524505667575, 'max_depth': 8, 'l2_leaf_reg': 5.075613524576857, 'subsample': 0.9429242011916316, 'colsample_bylevel': 0.4953651517214425, 'min_data_in_leaf': 66, 'scale_pos_weight': 3.6465805422915385}
# , user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'learning_rate': FloatDistribution(high=0.1, log=True, low=0.01, step=None), 'max_depth': IntDistribution(high=9, log=False, low=6, step=1), 'l2_leaf_reg': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'subsample': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'colsample_bylevel': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'min_data_in_leaf': IntDistribution(high=100, log=False, low=5, step=1), 'scale_pos_weight': FloatDistribution(high=4.0, log=False, low=0.8, step=None)}, trial_id=26, value=None)
# Best XGBoost trial: FrozenTrial(number=28, state=1, values=[0.1792281469050735], datetime_start=datetime.datetime(2024, 8, 19, 18, 50, 2, 163187), datetime_complete=datetime.datetime(2024, 8, 19, 18, 51, 19, 161927),
# params={'learning_rate': 0.04177826592780814, 'max_depth': 6, 'lambda': 0.016425354519466015, 'alpha': 0.002110796069354656, 'subsample': 0.6502571154031597, 'colsample_bytree': 0.9945865982720925, 'colsample_bynode': 0.6264467992167548, 'scale_pos_weight': 3.306461727445813}
# , user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'learning_rate': FloatDistribution(high=0.1, log=True, low=0.01, step=None), 'max_depth': IntDistribution(high=9, log=False, low=6, step=1), 'lambda': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'alpha': FloatDistribution(high=10.0, log=True, low=0.001, step=None), 'subsample': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'colsample_bytree': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'colsample_bynode': FloatDistribution(high=1.0, log=False, low=0.4, step=None), 'scale_pos_weight': FloatDistribution(high=4.0, log=False, low=0.8, step=None)}, trial_id=28, value=None)
