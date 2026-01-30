import numpy as np

class Node:
    """
    This class is the basic building block of the Decision tree. 
    Each node stores the information needed to make a split or predictions
    feature_index - Index of the feature to split on
    threshold - Threshold value for the split
    left - Left child node
    right - Right child node
    value - mean value for leaf nodes - for Regression, mean of the target values of the Node
    """
    def __init__(self,feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressor:
    """
    This class is the core which applies the logic for building a decision tree
    1. Initialization of parameters
    2. Function mse is used for calculating how label y is deviated from its mean 
    3. Finding best split with minimum mean squared error 
    4. building tree with the given X and y
    5. Fit the regressor to the labeled data
    6. Make predictions of a batch
    7. Evaluation metrics
    """
    def __init__(self, min_samples_split=2, max_depth=float("inf")):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def mse_split(self,y):
        """
        This is used for determine the best split at a node 
        """
        if len(y)==0:
            return 0
        else:
            return np.mean((y - np.mean(y)) ** 2)

    def find_best_split(self,X,y,m,n):
        """
        This function returns information of the best split - 
        1. index of the column  
        2. Threshold value of that column
        3. Indices of the dataset of the left child
        4. Indices of the dataset of the right child
        """
        n_samples, n_features = m,n
        best_mse = float("inf")
        best_split = None
        # current_mse = self.mse(y)

        for idx in range(n_features):
            unique_values=np.unique(X[:,idx])
            for thr in unique_values:
                left_idx=np.where(X[:,idx]<=thr)[0]
                right_idx=np.where(X[:,idx]>thr)[0]
                if len(left_idx)==0 or len(right_idx)==0:
                    continue
                y_left, y_right=y[left_idx], y[right_idx]
                if len(y_left)==0 or len(y_right)==0:
                    continue
                mse_left, mse_right=self.mse_split(y_left), self.mse_split(y_right)

                weighted_mse= (len(y_left) * mse_left + len(y_right) * mse_right) / n_samples
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_split = {
                        "feature_index": idx,
                        "threshold": thr,
                        "left_indices": left_idx,
                        "right_indices": right_idx
                    }
        return best_split


    def build_tree(self,X,y,m,n,depth=0):
        """
        This function performs the recursion and build the decision tree with best split at each node
        This returns a built tree, which we will be using to traverse during the prediction
        """
        n_samples, n_features = m,n
        #Stopping criteria logic
        #Current depth has reached the maximum depth
        if self.max_depth is not None and depth>= self.max_depth:
            return Node(value=np.mean(y))
        #current number of samples has reached minimum samples split
        if n_samples < self.min_samples_split:
            return Node(value=np.mean(y))
        #all target values are the same
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        
        best_split=self.find_best_split(X,y,m,n)
        if best_split is None:
            return Node(value=np.mean(y))
        feature_index=best_split["feature_index"]
        threshold=best_split["threshold"]
        left_indices=best_split["left_indices"]
        right_indices=best_split["right_indices"]

        left_child=self.build_tree(X[left_indices], y[left_indices], len(left_indices), n, depth+1)
        right_child=self.build_tree(X[right_indices], y[right_indices], len(right_indices), n, depth+1)
        
        return Node(feature_index, threshold, left_child, right_child)


    def fit(self, X, y):
        """
        This functions fits the tree to the data
        We will be interacting with this function on the front-end while using this algorithm
        This assign the built tree in the above fuction to root
        """
        m,n=X.shape
        X_t=X.to_numpy()
        y_t=y.to_numpy().reshape(-1,1)
        self.root=self.build_tree(X_t, y_t, m, n)

    def predict_sample(self, x, node):
        """
        Used to make prediction for a single sample at a time
        The test data traverse through the decision tree algorithm and its prediction is made
        """
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x,node.left)
        else:
            return self.predict_sample(x,node.right)


    def predict(self, X):
        """
        This takes the batch of test or validation data to make predictions
        Returns the predictions
        """
        X_t=X.to_numpy()
        predictions=[self.predict_sample(x, self.root) for x in X_t]
        predictions=np.array([float(y) for y in predictions])
        return predictions
    
    def mean_absolute_error(self,y_true,y_pred):
        """
        This returns average absolute prediction error
        """
        assert len(y_true) == len(y_pred)
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(y_true_np)
        mae=0
        for i in range(m):
            mae+=np.abs(y_true_np[i]-y_pred[i])
        mae/=m
        return mae
            

    def mean_squared_error(self,y_true,y_pred):
        """
        This returns average squared error
        """
        assert len(y_pred) == len(y_true)
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(y_true_np)
        error=0
        for i in range(m):
            error+=(y_true_np[i] - y_pred[i]) ** 2
        error/=m
        return error
    
    def rmse(self,y_true,y_pred):
        """
        This return root mean square error of the predictions
        """
        mse=self.mean_squared_error(y_true,y_pred)
        rmse=np.sqrt(mse)
        return rmse
    
    def r_squared(self,X,y_true,y_pred):
        """
        This are R^2 score and adjusted R^2 - Coefficient of Determination
        This tells us how much variance is explained by the model
        """
        X_np=X.to_numpy()
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(X_np)
        n=X_np.shape[1]
        p=n+1
        y_mean=np.mean(y_true_np)
        rss=0
        tss=0
        for i in range(m):
            rss+=(y_true_np[i]-y_pred[i])**2
            tss+=(y_true_np[i]-y_mean)**2
        r_squared = 1 - (rss/tss)
        adj_r_squared = 1 - (((1-(r_squared)**2)*(m-1))/(m-p-1))
        return r_squared, adj_r_squared

    def mbe(self, y_true, y_pred):
        """
        This return mean bias error. 
        It is a metric that tells you the average difference between predicted values and actual values.
        """
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(y_true_np)
        mbe=0
        for i in range(m):
            mbe+=(y_pred[i]-y_true_np[i])
        mbe/=m
        return mbe
    
    def mape(self, y_true,y_pred):
        """
        Mean absolute percentage error
        This is very interpretable
        """
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(y_true_np)
        error=0
        for i in range(m):
            error+=np.abs((y_true_np[i] - y_pred[i]) / y_true_np[i])
        error=error * 100/m
        return error
    
    def smape(self,y_true,y_pred):
        """
        Symmetric Mean absolute percentage error
        This is a metric used to evaluate forecasting/regression models, especially in time series.
        """
        y_true_np=y_true.to_numpy().reshape(-1,1)
        m=len(y_true_np)
        error=0
        for i in range(m):
            error+= np.abs(y_true_np[i]-y_pred[i]) /(np.abs(y_true_np[i]) + np.abs(y_pred[i]))
        error*=2
        return error
    
    def explained_var_score(self,y_true,y_pred):
        """
        It measures how much of the variance in the target variable is explained by your model.
        """
        y_true_np=np.array(y_true)
        y_pred_np=np.array(y_pred)
        m=len(y_true_np)
        residuals=y_true_np-y_pred_np
        var_res=np.var(residuals)
        var_true=np.var(y_true_np)
        return var_res/var_true
    