import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
import itertools
from timeit import default_timer as timer
import pickle

"""
Create object that contains functions to train an XGB Regressor model for queue estimation. Requires json control file
and csv for processed data that includes BSM, occupancy and flow data. 
"""

class QueueEstimatorTrainer(object):
    
    def __init__(self, control_file, parameters_file, train_share = 0.75):
        """
        Define the queue estimator trainer with the given json control file, 
        """
        with open(control_file) as in_f:
            self.control = json.load(in_f)

  
        # Read json for parameters to be grid searched
        with open(parameters_file) as in_f:
            self.params = json.load(in_f)

        # Load in Xy DataFrame for ML
        try:
            self.df_Xy = self.read_Xy_data(self.control['data_for_ML_filepath'])
        except (TypeError, KeyError):
            print('Static data files could not be properly loaded.')

        if (train_share < 0) or (train_share > 1):
            raise Exception("Train share must be between 0 and 1")
        else:
            self.train_share = train_share

        # Get train and test splits, as well as X and y splits
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test_X_y()


    def modify_train_share(self, new_train_share):
        '''
        Modifies trains share if user wants to change after starting instance
        '''
        if (new_train_share < 0) or (new_train_share > 1):
            raise Exception("Train share must be between 0 and 1")
        else:
            self.train_share = new_train_share

    def read_Xy_data(self, df_Xy_filepath):
        '''
        Reads in data for ML training based on file path given in control file
        '''
        df_Xy = pd.read_csv(df_Xy_filepath)
        df_Xy.time = pd.to_datetime(df_Xy.time)
        return df_Xy

    def split_train_test_X_y(self):
        # preparing X and y
        col_lst = ['queue_count_max', 'queue_len_max', 'queue_indicator', 'queue_count_binned','time']
        X = self.df_Xy.loc[:,~self.df_Xy.columns.isin(col_lst)] #.to_numpy()
        y = self.df_Xy['queue_count_max']

        # Splitting to train and test
        time_steps = list(set(X.time_float))
        time_steps.sort()
        train_index = int(len(time_steps) * self.train_share)
        train_time_steps = time_steps[:train_index]
        test_time_steps = time_steps[train_index:]

        # Coming up with indexes for train and test data in main df
        train_mask = [x in train_time_steps for x in X.time_float]
        test_mask = [x in test_time_steps for x in X.time_float]

        # Subset main dfs
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask]
        y_test = y[test_mask]

        return X_train, X_test, y_train, y_test

    def temporal_validation(self, df_X, arr_y, model=xgb.XGBRegressor(), folds = 4, measure = 'mse', non_categorical=False):
        '''
        Implements a temporal validation (as opposed to cross-validation) model validation approach that 
        avoids leakage by splitting the data into folds as cross-validation, and only validates on data 
        that occurs after the training data. This approach is more accurate because it better represents 
        the operational setting, where only data from the past are available to train.

        Citation: https://textbook.coleridgeinitiative.org/chap-ml.html
        '''
        
        time_span = list(set(df_X.time_float))
        time_span.sort()
        n_times = len(time_span)
        fold_size = int(n_times / folds)

        ## Loop through the folds to train data
        train_index = fold_size
        val_index = train_index + fold_size
        score_list = []
        mse_list = []

        for i in range(folds-1):
            # Get indeces for training and test folds
            train_times = time_span[0:train_index]
            val_times = time_span[train_index:val_index]

            # Subset dataframes into the temporary train and validation sets
            X_sub_train = df_X[[True if x in train_times else False for x in df_X.time_float]]
            y_sub_train = arr_y[np.array([1 if x in train_times else 0 for x in df_X.time_float]).astype(bool)]
            X_sub_val = df_X[[True if x in val_times else False for x in df_X.time_float]]
            y_sub_val = arr_y[np.array([1 if x in val_times else 0 for x in df_X.time_float]).astype(bool)]

            # Train model
            model.fit(X_sub_train, y_sub_train)
            
            # Regression prediction (for MSE)
            y_reg_pred = model.predict(X_sub_val)
            # Cannot be zero
            y_reg_pred = [0 if x < 0 else x for x in y_reg_pred]
            # Add to MSE list
            mse_list.append(mean_squared_error(y_sub_val, y_reg_pred))

            # Get predictions for classification metric
            if non_categorical == False:
                y_pred = np.rint(y_reg_pred)
                score_list.append(accuracy_score(y_sub_val, y_pred))
    
            # Accounting for if the range of times is not neatly divisible by the number of folds
            if len(time_span) % folds != 0:
                remainder = len(time_span) % folds
                if (folds - (i+1)) == remainder:
                    val_index +=1
                    fold_size+=1
                    
            # Reset indeces for next loop iteration
            train_index += fold_size
            val_index += fold_size
        
        return score_list, mse_list

    def product_dict(self, **kwargs):
        '''
        Helper function to get cartesian product of all parameters
        from: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        '''
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
                yield dict(zip(keys, instance))

    def xgboost_temporal_grid_search(self, folds = 4, measure = 'mse', non_categorical=False, stop_condition=0):
        '''
        Takes a dictionary of XGBRegressor parameters and runs and exhaustive grid search across them using
        temporal validation.
        
        Measure can be 'accuracy' or 'mse' depending on how you want to select best model. 
        Always returns both score (accuracy) and MSE, but parameter determines for which measure to optimize.
        '''
        best_acc = 0
        best_mse = 1e100
        best_model = xgb.XGBRegressor()
        best_parameters = []
        
        # Get all combinations (cartesian product) of parameters
        cart_prod_params_list = list(self.product_dict(**self.params))
        
        # Loop through all combinations of parameters
        # Use temporal validation to get scores and save best one
        i=0
        start = timer()
        stop_count = 0

        for comb in cart_prod_params_list:
            i+=1
            time_update = timer()
            print('Combination ' + str(i) + ' / ' + str(len(cart_prod_params_list)) + ', ' + str(int((time_update-start)//60)) + 'min, ' + 
                str(int((time_update-start)%60)) + 'sec')
            temp_model = xgb.XGBRegressor(**comb)
            score_list, mse_list = self.temporal_validation(model = temp_model, df_X=self.X_train, arr_y=self.y_train, 
                                                                                                 folds=folds, measure = measure, non_categorical=non_categorical)
            avg_acc = 0
            if non_categorical == False:
                avg_acc = sum(score_list)/len(score_list)
            avg_mse = sum(mse_list)/len(mse_list)
            
            if measure == 'accuracy':
                if avg_acc > best_acc:
                    best_mse = avg_mse
                    best_acc = avg_acc
                    best_model = temp_model
                    best_parameters = comb
                    stop_count = 0
                print('Accuracy: ' + str(best_acc))
                            
            elif measure == 'mse':
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_acc = avg_acc
                    best_model = temp_model
                    best_parameters = comb
                    stop_count = 0
                print('MSE: ' + str(best_mse))
            
            # Condition to break out of loop if same best metric for too many cycles in a row
            if stop_condition > 0:
                if stop_count >= stop_condition:
                    break
            stop_count += 1

        self.best_accuracy = best_acc
        self.best_mse = best_mse
        self.best_model = best_model
        self.best_parameters = best_parameters
        
        return self.best_accuracy, self.best_mse, self.best_model, self.best_parameters

    def validate_model(self):
        ''' 
        Validates best model using test data
        '''

        try:
            y_pred = [0 if y<0 else y for y in self.best_model.predict(self.X_test)]
            self.y_pred = y_pred
            mse_val = mean_squared_error(self.y_test, y_pred)
            acc_val = accuracy_score(self.y_test, np.rint(y_pred))
        except:
            print('Must have trained model to access this function')


        return mse_val, acc_val

    def train_best_parameters_on_full_set(self):
        '''
        After a grid search has been run, trains the model on full set of data
        '''
        try:
            full_X = pd.concat([self.X_train, self.X_test], axis=0).reset_index(drop=True)
            full_y = pd.concat([self.y_train, self.y_test])
            self.best_model.fit(full_X, full_y)
        except:
            print('Must have trained model to access this function')

    def save_best_model(self, filepath):
        '''
        Saves best version of a model through a pickle file
        '''
        try:
            pickle.dump(self.best_model, open(filepath, 'wb'))
        except:
            print('Must have trained model to access this function')







