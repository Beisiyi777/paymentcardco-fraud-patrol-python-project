import joblib
import pandas as pd

class Patrol:
    """
    A class to return an action in relation to whether a given transaction is fraudulent.

    Attributes
    ----------
    model_path : str
        Path to pre-trained model that has predict_proba() method available.
    
    data : pd.DataFrame
        DataFrame containing transaction_id and all corresponding variables necessary for model calculation.
    
    feature_list : list
        List of column names in  provided DataFrame with which predictions will be made.
    
    threshold : float
        Probability threshold at which a transaction is classified as fraudulent.

    Methods
    -------
    set_model_path(model_path)
        Updates the model path used to generate the probability that given transaction is fraudulent.
    
    set_data(data)
        Updates the DataFrame containing the given transaction including its id and all corresponding variables necessary to return an action.
    
    set_feature_list(feature_list)
        Updates the list of columns in the DataFrame used to predict whether the given transaction is fraudulent.
    
    set_threshold(threshold)
        Updates the threshold at which a transaction is classified as fraudulent.
    
    check_transaction(transaction_id)
        Returns 'PASS' if given transaction is not fraudulent and 'LOCK' if fraudulent.
    """    
    # Class variables shared by all instances
    default_model_path = 'hypertuned_xgb_model.pkl'
    default_feature_list = ['tr_type', 'tr_state', 'tr_amount_gbp', 'tr_currency', 'user_country', 'user_age', 'account_age', 'tr_day', 'tr_weekday', 'tr_hour']
    default_threshold = 0.2810381
    
    def __init__(self, model_path=default_model_path, data=None, feature_list=default_feature_list, threshold=default_threshold):
        """
        Initialize the Patrol class with provided or default model path, data, feature list, and threshold.

        Parameters
        ----------
        model_path : str, optional, default='hypertuned_xgb_model.pkl'
            Path to pre-trained model that has predict_proba() method available.
        
        data : pd.DataFrame, optional, default=None
            DataFrame containing transaction_id and all corresponding variables necessary for model calculation.
        
        feature_list : list, optional, default=['tr_type', 'tr_state', 'tr_amount_gbp', 'tr_currency', 'user_country', 'user_age', 'account_age', 'tr_day', 'tr_weekday', 'tr_hour']
            List of column names in  provided DataFrame with which predictions are made.
        
        threshold : float, optional, default=0.2810381
            Probability threshold at which a transaction is classified as fraudulent.
        """         
        #print('Initializing Patrol...')
        self.set_model_path(model_path)
        self.set_data(data)
        self.set_feature_list(feature_list)
        self.set_threshold(threshold)   
  
    def set_model_path(self, model_path):
        """
        Update path to a pre-trained model that has the .predict_proba() method available.

        Parameters
        ----------
        model_path : str
            A new path to the model

        Raises
        ------
        AttributeError:
            If the model referenced in model_path does not have a predict_proba method.
        """
        #print('Setting Model...')
        self.model = joblib.load(model_path)
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError('The loaded model does not have a predict_proba method and cannot be used.')
            
    def set_data(self, data):
        """
        Update DataFrame containing transaction_id and all corresponding variables necessary for model calculation.

        Parameters
        ----------
        data : pd.DataFrame
            A new DataFrame

        Raises
        ------
        ValueError:
            If data is not provided.
        
        TypeError:
            If data is not a pandas DataFrame.        
        """
        #print('Setting Data...')
        if data is None:
            raise ValueError('A DataFrame containing transaction_id and all corresponding variables must be provided.')
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Transaction data must be a pandas DataFrame.')

        #print(f'Data received: {data.head()}')
        self.data = data
    
    def set_feature_list(self, feature_list):
        """
        Update the list of column names with which predictions are made.

        Parameters
        ----------
        feature_list : list
            A new list of column names
        
        Raises
        ------
        ValueError: 
            If not all columns in feature list are included in data.
        """
        #print('Setting Feature List...')
        self.feature_list = feature_list
        
        if not all(feature in self.data.columns for feature in self.feature_list):
            raise ValueError('The provided data does not include all required features.')        
         
    def set_threshold(self, threshold):
        """
        Update probability threshold at which to determine that a transaction is fraudulent.

        Parameters
        ----------
        threshold : float
            A new threshold

        Raises
        ------
        TypeError:
            If new threshold is not of type float
        
        ValueError:
            If new threshold is not between 0 and 1.
        """
        #print('Setting Threshold...')
        if not isinstance(threshold, float):
            raise TypeError('Threshold must be a float.')

        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1.')

        self.threshold = threshold

    def check_transaction(self, transaction_id):
        """
        Determine action for given transaction through the following steps:      
        1) calculate probability of fraud for given transaction id based on pre-loaded features in the pre-loaded data
        2) compare probability to pre-loaded threshold
        3) return 'LOCK' (if probability is at or above threshold) or 'PASS' (if probability is below threshold)

        This function checks each transaction separately and returns an action, regardless of previous checks on other transactions by the same user.

        Parameters
        ----------
        transaction_id : str
            Transaction_id for transaction in pre-loaded DataFrame to be checked.

        Returns
        -------
        'LOCK' : 
            if predicted probabilty for given transaction is at or above pre-loaded threshold
        
        'PASS': 
            if predicted probability for given transaction is below pre-loaded threshold                        
        """
        #print('Checking Transaction...')
        try:
            # ensure data is pre-loaded
            if self.data is None:
                raise ValueError('Data has not been loaded.  Please load the DataFrame containing the data for transaction_id using set_data method.')
                
            # ensure transaction_id exists in data             
            if not self.data['tr_id'].str.contains(transaction_id).any():
                raise ValueError(f'Transaction ID {transaction_id} not found in data.')

            # extract features for given transaction_id
            transaction_features = self.data[self.data['tr_id'] == transaction_id][self.feature_list]

            # ensure transaction_id is unique
            if transaction_features.shape[0] > 1:
                raise ValueError('Multiple transactions found for Transaction ID.')

            # ensure number of extracted features match number of expected features
            num_transaction_features = transaction_features.shape[1]
            num_expected_features = len(self.feature_list)
            
            if num_transaction_features != num_expected_features:
                raise ValueError(f'Mismatch on number of features, got: {num_transaction_features}, expected: {num_expected_features}')

            # ensure extracted features match expected features
            extracted_features = set(transaction_features.columns)
            expected_features = set(self.feature_list)
            
            if extracted_features != expected_features:
                raise ValueError(f'Mismatch on features, got: {extracted_features}, expected: {expected_features}')

            # generate probability
            probability = self.model.predict_proba(transaction_features)[:, 1]

            # compare probability to threshold
            if probability >= self.threshold:
                result = 'LOCK'
            else:
                result = 'PASS'
                
            return result
        
        except ValueError as ve:
            # error handling for specific errors
            print(f'Error {ve}')
            return 'Action cannot be generated.'

        except Exception as e:
            # capture any unforeseen errors
            print(f'An unexpected error occurred: {e}')
            return 'Action cannot be generated.'

        finally:
            print(f'Check for ID {transaction_id} completed: {result}')