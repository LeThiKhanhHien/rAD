import numpy as np
import time
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.rAD_shallow import rAD
from src.PU import PUlearning
from sklearn.metrics import roc_curve, auc
#from src.BaseSVDD import BaseSVDD
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM
from sklearn import svm
import json 
from pathlib import Path
import random
import pickle

def simple_run():
    txt_file='result.txt'

    with open(txt_file, 'a') as f:
        f.write("Results of  tests on classical benchmark AD dataset  \n")
        f.write('dataset (d,n,pi_n) & rAD_quad &  rAD_hinge & rAD_huber & PU_quad & PU_hinge PU_huber &OC-SVM & semi OCSVM \n' )

    for dataset in ['thyroid','pendigits']: 
               
        with open(txt_file, 'a') as f:
            f.write(dataset)
        
        data=np.load(dataset+'.npz')       
        X=data['X']
        y=data['y']
        y=y*2-1
        y=-y
        n_samples= len(y)
        print('n_samples',n_samples)
        n_feature = X.shape[1]
        pi_negative=np.count_nonzero(y==-1)/n_samples     
        print('n_feature',n_feature)
        print('pi_negative',pi_negative)

        num_train=2 # number of trials
        sqrtn=np.sqrt(num_train)
        auc_PU_quad=[]
        auc_PU_hinge=[]
        auc_PU_huber=[]
        auc_rAD_quad=[]
        auc_rAD_hinge=[]
        auc_rAD_huber=[]
        auc_ocsvm=[]
        auc_s_ocsvm=[]
        
       
        for i in range(num_train): #num_train random split 
         
            have_negative=0
            while have_negative==0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                if np.count_nonzero(y_train==-1)>0:
                    have_negative=1
            
            print("number of negative samples",np.count_nonzero(y_train==-1)) 
            print("size of train set", len(y_train)) 

            #split X_train to get label/unlabel data, we suppose unlabel samples are larger
            have_negative=0 #to guarantee there are negative examples in the labeled data
            while have_negative==0:
                X_unlabel, X_label, y_unlabel, y_label = train_test_split(X_train, y_train, test_size=0.05, random_state=None)
                if np.count_nonzero(y_label==-1)>0:
                    have_negative=1
                    

            #form train data from X_unlabel, X_label, y_unlabel, y_label  for semi-SVDD and semi-SVM
            X_train_semi=np.vstack((X_unlabel,X_label))
            y_train_semi=np.vstack((np.ones((len(y_unlabel),1)), np.reshape(y_label,(len(y_label),1))) )   

            #form train data for PUlearning 
            X_positive=X_label[np.where(y_label==1)]
            X_unlabel_PU=np.vstack(( X_unlabel, X_label[np.where(y_label==-1)])) 
                
            #train PUlearning
            print("...running PUlearning...")
            for loss in ['squared_error','hinge','modified_huber']:
                PU_model=PUlearning(kernel='linear', gamma='scale', loss=loss, prob_positive=0.8)
                PU_model.fit(X_unlabel_PU,X_positive)
                y_score=PU_model.decision_function(X_test)                
                            
                fpr_PU, tpr_PU, _ = roc_curve(y_test, y_score)
                roc_auc_PU = auc(fpr_PU, tpr_PU)
                if loss=='squared_error':
                    auc_PU_quad.append(roc_auc_PU)
                if loss=='hinge':
                    auc_PU_hinge.append(roc_auc_PU)
                if loss=='modified_huber':
                    auc_PU_huber.append(roc_auc_PU)


            # train rAD 
            print("...running rAD, no kernel...")
            for loss in ['squared_error','hinge','modified_huber']:
                rAD_model=rAD( kernel='linear', gamma='scale', loss=loss, prob_positive=0.8, a=0.5)                
                rAD_model.fit(X_unlabel,X_label,y_label)
                y_score=rAD_model.decision_function(X_test)                
                                
                fpr_rAD, tpr_rAD, _ = roc_curve(y_test, y_score)
                roc_auc_rAD = auc(fpr_rAD, tpr_rAD)
                if loss=='squared_error':
                    auc_rAD_quad.append(roc_auc_rAD)
                if loss=='hinge':
                    auc_rAD_hinge.append(roc_auc_rAD)
                if loss=='modified_huber':
                    auc_rAD_huber.append(roc_auc_rAD)

            # Train OCSVM model
            print("...running OCSVM...")                       
            one_class_svm = OneClassSVM(nu=0.1, kernel = 'rbf', gamma = 'auto').fit(X_train)
            y_score_svm = one_class_svm.score_samples(X_test)             
            fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
            roc_auc_svm = auc(fpr_svm, tpr_svm)
            auc_ocsvm.append(roc_auc_svm)

            #semisupervised SVM
            print("...running semi-supervised SVM ...")          
            svm_model= svm.SVC(kernel='rbf')
            svm_model.fit(X_train_semi, np.reshape(y_train_semi,-1))
            y_score_svm_semi = svm_model.decision_function(X_test)   
            fpr_svm_semi, tpr_svm_semi, _ = roc_curve(y_test, y_score_svm_semi)
            roc_auc_svm_semi = auc(fpr_svm_semi, tpr_svm_semi)
            auc_s_ocsvm.append(roc_auc_svm_semi)         

        
        with open(txt_file, 'a') as f:
            f.write('(' + str(n_feature)+', '+ str(n_samples) + ', ' + str(pi_negative) +') & ')
            f.write(str(np.mean(auc_rAD_quad))+ '('+ str(np.std(auc_rAD_quad)/sqrtn)+ ') &')
            f.write(str(np.mean(auc_rAD_hinge))+ '('+ str(np.std(auc_rAD_hinge)/sqrtn)+ ') &')
            f.write(str(np.mean(auc_rAD_huber))+ '('+ str(np.std(auc_rAD_huber)/sqrtn)+ ') & ')
                    
            f.write(str(np.mean(auc_PU_quad))+ '('+ str(np.std(auc_PU_quad)/sqrtn)+ ') &')
            f.write(str(np.mean(auc_PU_hinge))+ '('+ str(np.std(auc_PU_hinge)/sqrtn)+ ') &')
            f.write(str(np.mean(auc_PU_huber))+ '('+ str(np.std(auc_PU_huber)/sqrtn)+ ') & ')
            f.write(str(np.mean(auc_ocsvm))+ '('+ str(np.std(auc_ocsvm)/sqrtn)+ ') & ')
            f.write(str(np.mean(auc_s_ocsvm))+ '('+ str(np.std(auc_s_ocsvm)/sqrtn)+ ') & \n')


if __name__ == '__main__':
    simple_run()
