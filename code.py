
# coding: utf-8

# In[1]:


import nibabel as nib
import nilearn
from nilearn import image
from nilearn import masking

import scipy.io as io
import numpy as np

import cv2
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[2]:


test_filename = './sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii' 
retest_filename= './sub-01/ses-retest/func/sub-01_ses-retest_task-fingerfootlips_bold.nii'
labels = io.loadmat('label.mat') #load labels
labels = labels['label'].flatten() #list of labels


# In[89]:


# This function takes a file path, load the file with nibabel, 
# get the brain mask and returns the extracted brain region for each picture
def get_masked(filename,thr):
    test_file = nib.load(filename)   #loading the file
    mask_nilearn = masking.compute_brain_mask(test_file, threshold=thr)  #getting the mask
    masked_nilearn = masking.apply_mask(test_file, mask_nilearn) #extract the brain region with the mask
    return np.array(masked_nilearn)


# In[90]:


#This function takes in the image vectors and the number of features we want to extract from these vectors
#Applies the pca model to do the feature selection and return vectors of these features
#The number of components that yields best results of 0.857 include 78,80,83,84,86,87
def extract_features(masked_nilearn,n):
    pca = PCA(n_components=n, random_state = 0) #pca model
    pca.fit(masked_nilearn)
    
    x_reduced = []
    x_reduced = pca.transform(masked_nilearn)  #features
    return x_reduced


# In[180]:


#This function will return the accuracy rate after using pca
#input c and kernels feeds into the svc model
#splits specifies the parameter for stratified k fold cross validation
#returns the accuracy rate for the testing group

def svm_with_pca(x_reduced, labels, c, kernel, splits):  #SVC(C=1.5, kernel='sigmoid', gamma='auto')
    
    clf = make_pipeline(StandardScaler(), SVC(C=c, kernel=kernel, gamma='auto', random_state = 0))
    clf.fit(x_reduced, labels)
    predict = clf.predict(x_reduced)
    
    Str_kf = StratifiedKFold(n_splits = splits, random_state = 0)  #stratified k fold n_splits = 20
    
    for train_idx, test_idx in Str_kf.split(X = x_reduced, y = labels): #get score only for test data
    
        train_data_x = [x_reduced[i] for i in train_idx]
        train_data_y = [labels[i] for i in train_idx]
    
        test_data_x = [x_reduced[j] for j in test_idx]
        test_data_y = [labels[j] for j in test_idx]
    
        clf.fit(train_data_x, train_data_y) #fit the model
        predicted_y = clf.predict(test_data_x)
        
        #a = clf.predict(train_data_x)
        #a = np.array(a)
        
        test_data_y = np.array(test_data_y)
        predicted_y = np.array(predicted_y)

        length = len(test_data_y) # length of the tst data
        L =  len(train_data_y)
        
        count_correct = sum(test_data_y == predicted_y) #count the number of correct prediction
        accuracy_w_pca = count_correct/length
        
        #tr = sum(train_data_y == a)/L
        
    return accuracy_w_pca


# In[114]:


#brain region extraction for test and retest data
test_masked = get_masked(test_filename, 0.4)   


# In[181]:


#get results for test data set
test_features = extract_features(test_masked, 78)  #test task dimentional reduction with pca
test_svm_w_pca = svm_with_pca(test_features, labels, 1.5, 'linear', 19) #
print('For the test data the best accuracy rate is:', test_svm_w_pca, 'with brain threshold to be 0.4, pca pararmeter set to be 78, svm using linear kernel and 19 stratified folds')


# In[182]:


#get results for retest data set
retest_masked = get_masked(retest_filename,0.8)
retest_features = extract_features(retest_masked, 90)
retest_svm_w_pca = svm_with_pca(retest_features, labels, 1.5, 'linear', 16)

print('For the retest data the best accuracy rate is:', retest_svm_w_pca, 'with brain threshold to be 0.8, pcapararmeter set to be 90, svm using linear kernel, and 16 stratified folds')


# In[183]:


#This function will return the accuracy rate without using pca
#input c and kernels feeds into the svc model
#splits specifies the parameter for stratified k fold cross validation
#returns the accuracy rate for the testing group

def svm_wth_pca(masked_nilearn, labels, c, kernel, splits):  #SVC(C=1.5, kernel='sigmoid', gamma='auto')
    
    clf = SVC(C=c, kernel=kernel, gamma='auto', random_state = 0)
    clf.fit(masked_nilearn, labels)
    Str_kf = StratifiedKFold(n_splits = splits, random_state =0) #n_splits = 20
    
    for train_idx_wt, test_idx_wt in Str_kf.split(X = masked_nilearn, y = labels): #accuracy score only for test data
    
        train_data_x_wt = [masked_nilearn[i] for i in train_idx_wt]  #training brain data without using pca
        train_data_y_wt = [labels[i] for i in train_idx_wt]

        test_data_x_wt = [masked_nilearn[j] for j in test_idx_wt]  #testing brain data without using pca
        test_data_y_wt = [labels[j] for j in test_idx_wt]

        clf.fit(train_data_x_wt, train_data_y_wt)  #fit the model
        predicted_y_wt = clf.predict(test_data_x_wt)

        test_data_y_wt = np.array(test_data_y_wt) 
        predicted_y_wt = np.array(predicted_y_wt)  #predicted for the test data group

        length_wt = len(test_data_y_wt) #test data

        count_correct_wt = sum(test_data_y_wt == predicted_y_wt) 

        accuracy_wt = count_correct_wt/length_wt #accuary rate only for test data
    
    return accuracy_wt


# In[184]:


#Compare the above results with following results that produced without using pca
test_svm_wth_pca = svm_wth_pca(test_masked, labels, 1.5, 'sigmoid', 19)
print('Without dimension reduction, the accuracy rate for test data becomes:', test_svm_wth_pca)

retest_svm_wth_pca = svm_wth_pca(retest_masked, labels, 1.5, 'linear', 16)
print('Without dimension reduction, the accuracy rate for retest data becomes:', retest_svm_wth_pca)

