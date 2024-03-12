# sklearn implementation for shallow rAD using the second risk estimator
# written by LTK Hien, 2023
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils._array_api import get_namespace
import time
import matplotlib.pyplot as plt
import sys
import io
class rAD:
    """Anomaly Detection based on risk estimator.

    Parameters
    ----------
    C : float, default=0.1
        Regularization parameter. Currently we use $l_2$ penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='linear'        
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - if ``gamma='scale'" then gamma= 1 / (n_features * X.var()),
        - if ``gamma='auto'", then gamma= 1 / n_features.
    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    loss : {'modified_huber','squared_error','hinge','log_loss', 'log'}   
    prob_positive: float, defaut = 0.8 
            probability of positive class
    a: float, defaut = 0.5
          coeficient of the risk estimator         
    n_components: dimension of the transformed feature space
    reg_coef : float, default=0.01
        Regularization parameter. Currently we use $l_2$ penalty.
    """
    def __init__(self, 
                 kernel='linear',
                 degree=3,
                 gamma='scale',
                 coef0=1,
                 loss='modified_huber',
                 prob_positive=0.8,
                 a=0.5,
                 n_components=200,                 
                 reg_coef=0.01,
                 draw_loss='no'):
        
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma       
        self.coef0 = coef0
        self.loss  = loss
        self.prob_positive = prob_positive
        self.a=a
        self.n_components=n_components
        self.reg_coef=reg_coef
        self.draw_loss=draw_loss
        self.X_unlabel = None
        self.X_label = None
        self.y_label = None
        self.n_samples=None
        self.coef_w=None
        self.coef_b=None
        self.y_predict= None
        
        self.running_time = None
       
          
    #get kernel and compute the new feature space 
    def feature_transorm(self,X):
        # check gamma
        if self.gamma == 0:
            raise ValueError(
                "The gamma value of 0.0 is invalid. Use 'auto' to set"
                " gamma to a value of 1 / n_features.") 
        if self.gamma is None:
            self.gamma = 'scale'
        if self.gamma == 'scale':
               X_var = X.var()
               gamma = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
        elif self.gamma == 'auto':   
               gamma = 1.0 / X.shape[1]
        else:
            raise ValueError(
                   "'gamma' should be either 'scale' or 'auto'.")  
        
        if self.kernel=='rbf':
            rbf_feature= RBFSampler(gamma=gamma,n_components=self.n_components, random_state=42)
            X_new = rbf_feature.fit_transform(X)
        elif self.kernel=='poly': 
            poly_ts = PolynomialCountSketch(degree=self.degree, gamma=gamma,n_components=self.n_components, random_state=42)
            X_new= poly_ts.fit_transform(X)
        elif self.kernel=="sigmoid":
            nystroem_feature = Nystroem(kernel='sigmoid', gamma=gamma,n_components=self.n_components, random_state=42)
            X_new = nystroem_feature.fit_transform(X)  
        elif self.kernel=="linear":
            X_new =X
        else:   
            raise ValueError(
                   "'kernel' should be one of {'linear', sigmoid', 'poly', 'rbf' }.") 
        return X_new  

   


    def fit(self,X_unlabel,X_label,y_label):
     # use SGDClassifier to solve a regularized weighted regression problem  
        start_time = time.time()
        self.X_unlabel = X_unlabel
        self.X_label = X_label
        self.y_label = y_label
        n_unlabel=X_unlabel.shape[0] 
        n_positive= np.count_nonzero(y_label==1)
        n_negative=np.count_nonzero(y_label==-1)
        self.n_samples=n_unlabel + n_positive + n_negative

        if n_negative==0:
            raise ValueError("There are no negative samples in the training data. rAD currently does not support this cases")
        if self.kernel=="linear":
            self.n_components=self.n_samples
        if self.n_components > self.n_samples: 
            raise ValueError(
                 "There are more features than number of samples in the transformed feature space. rAD may not work well. Decrease n_components")
        
        X=np.vstack(( X_unlabel, X_label)) 
        X=self.feature_transorm(X)
        prob_negative=1-self.prob_positive
        
        y=np.concatenate((np.ones(n_unlabel),y_label))              
        dup_negative=X[np.where(y_label==-1)]       
        X_train=np.vstack(( X, dup_negative)) 
        y_train=np.concatenate((y,np.ones(n_negative)))
                    
        sample_weight=(1-self.a)*self.prob_positive/n_positive*np.ones(n_positive+n_negative)
        sample_weight[np.where(y_label==-1)]=prob_negative/n_negative
          
        sample_weight=np.concatenate((self.a/n_unlabel*np.ones(n_unlabel),sample_weight))
        sample_weight=np.concatenate((sample_weight,-self.a*prob_negative/n_negative*np.ones(n_negative)))
        verbose=0
        if self.draw_loss=='yes':
            verbose=1

        clf = SGDClassifier(loss=self.loss, penalty='l2', alpha=self.reg_coef, max_iter=10000, tol=0.001,verbose=verbose)
        clf.fit(X_train,y_train,sample_weight=sample_weight)
        if self.draw_loss=='yes':
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            sys.stdout = old_stdout
            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if(len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.savefig("loss_curve.png")
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.close()
            

        
        self.coef_w=clf.coef_
        self.coef_b=clf.intercept_            
            
        end_time = time.time()
        
        self.running_time = end_time - start_time
        
            
        return self 
    
    def decision_function(self,X):
         X = self.feature_transorm(X)
         xp, _ = get_namespace(X)
         scores = safe_sparse_dot(X, self.coef_w.T, dense_output=True) + self.coef_b
        # scores = X.dot(self.coef_w.T)+ self.coef_b 
         return xp.reshape(scores, -1) if scores.shape[1] == 1 else scores #np.reshape(scores, -1)
    
    def predict(self, X):
        scores=self.decision_function(X)
        self.y_predict=np.ones(X.shape[0])
        self.y_predict[np.where(scores<0)]=-1
        return self.y_predict 


        
