from flask import Flask, render_template, request, jsonify
import pandas as pd
from werkzeug import secure_filename
from flask_table import Table, Col
import time
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score
from pandas_ml import ConfusionMatrix

app = Flask(__name__)
global y_true
global y_pred

# @ signifies  a decorator - way to wrap a function and modifying its behaviour
@app.route('/')
def index():
    return 'this is the homepage'


# @ file uploader
@app.route('/upload')
def upload_file():
    return render_template('LandingMain.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        textual = request.form['datatype']
        print(textual)
        threshold = request.form['thresh']
        featurecount = request.form['features']
        f = request.files['file']
        if not threshold:
            threshold = 0.5
        print(threshold)
        f.save(secure_filename(f.filename))
        outtext = 'file uploaded successfully, model building under process.....'
        #return show_churn()
        if(textual =="Train"):
            return modelbuild(f.filename,featurecount,textual,False )
        else:
            return invokePredict(f.filename, threshold,textual,False)




def invokePredict(filename, threshold,callType,DEBUG):


    preprocessed_pred = readFileAndPreprocess(filename,callType,DEBUG)
    ###############################################################
    #   Conversion to matrix and data split
    ###############################################################
    X_features_all = preprocessed_pred[0]
    X_features_all = X_features_all.astype(float)
    Y_labels_dummy = preprocessed_pred[1]
    with open('SelectedFeatures_'+ filename.split('.')[0][:-8]+ '.pickle', 'rb') as handle1:
        loaded_features = pickle.load(handle1)
        X_features = X_features_all[loaded_features]
    print('The shape of the predicting data set is...')
    print(X_features.shape)
    X_mat = X_features.as_matrix()

   # Xmat_train, Xmat_test, Ymat_train, Ymat_test = train_test_split(X_mat, Y_mat, train_size=.8, random_state=8)

    ######  Standardize the data
    sc = StandardScaler()
    sc.fit(X_mat)
    X_std = sc.transform(X_mat)

   ###Retrieve saved model from the disk
    savedModel = 'LogisticRegModel_' + filename.split('.')[0][:-8] + '.pickle'
    with open(savedModel, 'rb') as handle:
        logreg_loaded = pickle.load(handle)
        probs = logreg_loaded.predict_proba(X_std)
        y_pred_class = logreg_loaded.predict(X_std)
    print("Show pred class raw")
    print(y_pred_class[0:10])
    y_pred_class_threshold = userThreshold(probs, threshold)
    print("Show pred class after threshold method call")
    print(y_pred_class_threshold[0:10])
    df_out = pd.DataFrame(data=probs, index=X_features.index, columns = ['Zero', 'One'])
    #df_out['Class'] = y_pred_class
    df_out['Class'] = y_pred_class_threshold
    df_out.insert(loc=0, column='MonthlyCharges', value=X_features_all['MonthlyCharges'])

    df_out= df_out.sort_values(['Class','One'], ascending=False)
    #print(df_out)
    #df_coef = pd.DataFrame(data=X_features.columns, columns=['Features'])

    #df_coef['Coefficients'] = logreg_loaded.coef_[0, :]
    #df_coef = df_coef.sort_values(by=['Coefficients'], ascending=False)

    predChurnRevenue = df_out[df_out['Class'] ==1]['MonthlyCharges'].sum()
    print("Going for the styling")
    dfs = df_out.style.apply(highlight_churnclass, class_val=1, column='Class', axis=1).set_table_attributes('class="churn"')
    print("After styling")
    #print(predChurnRevenue)
   # print(list(X_features.columns.values))
    #print(list(logreg_loaded.coef_[0, :]))

    if(DEBUG):
        print((X_features_all['Churn']).values)
        print(list(y_pred_class_threshold))
        from pandas_ml import ConfusionMatrix
        from sklearn.metrics import accuracy_score
        confusion_matrix = ConfusionMatrix(X_features_all['Churn'].values, y_pred_class_threshold)
        print("Confusion matrix:\n%s" % confusion_matrix)
        print("Test Accuracy  :: ", accuracy_score(X_features_all['Churn'].values, y_pred_class_threshold))

    #_repr_html_
    #tables = [df_out.to_html(classes='churn')]
    #tables=[df_out._repr_html_()]
    #nlist = [1,2,3,4,5]
    #style = '<style> body{background-color:orange;font-family: "Arial", "Helvetica", sans-serif} h2{margin-left: 5px;} </style>'
    #return style + '<p><h2>Prediction result to be diplayed' + '</h2></p>'"{0:.2f}".format(predChurnRevenue*12)  - predChurnRevenue*12
    return render_template('PredictResponse.html', tables=[dfs.render()], titles=['Churn Info'], churnRev = "{0:.2f}".format(predChurnRevenue*12), features = list(X_features.columns.values),coefval = list(logreg_loaded.coef_[0, :]))

def highlight_churnclass(s, class_val, column):
    #print("Coming inside the highlight method")
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] == class_val
    return ['background-color: LightSalmon' if is_max.any() else 'background-color: LightSteelBlue' for v in is_max]

############Function to handle selective threshold
def userThreshold(probas, threshold):
    #print('in the function...')
   # print(probas[1:10])
    #temp = probas[1:10]
    pred_class = np.zeros(len(probas))
    for i in range(len(probas)):
        if probas[i][1] >= float(threshold):
            pred_class[i] = 1
        else:
            pred_class[i] = 0
    return pred_class.astype(int)

def modelbuild(filename, featurecount,callType,DEBUG):
    preprocessed = readFileAndPreprocess(filename,callType,DEBUG)
    global g_Ymat_test
    global g_y_probs
    global g_all_probs
    X_features = preprocessed[0]
    Y_labels = preprocessed[1]

###############################################################
    #   Feature selection, to process with reduced features
###############################################################
    ##Select the N important features on the standardized data
    #print('Display shape after pre-processing')
    #print(X_features.shape)
    selector = SelectKBest(f_classif, k=int(featurecount))
    selector.fit(X_features.as_matrix(),Y_labels.as_matrix())
    indxs_selected = selector.get_support(indices=True)
    features_dataframe_red = X_features.columns[indxs_selected]
    X_features_red = X_features[features_dataframe_red]
    print('The features selected for processing after collinear feature drops....')
    print(features_dataframe_red)
    #print(X_features_red.shape)

##Save the features list, to be used for prediction
    with open('SelectedFeatures_'+ filename.split('.')[0][:-6] +'.pickle', 'wb') as handle:
        pickle.dump(features_dataframe_red, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_mat = X_features_red.as_matrix()
    Y_mat = Y_labels.as_matrix()
    ##Split data intpo Train and Test set
    Xmat_train, Xmat_test, Ymat_train, Ymat_test = train_test_split(X_mat, Y_mat, train_size = .8, random_state=8)

######  Standardize the data
    sc = StandardScaler()
    sc.fit(Xmat_train)
    X_train_std = sc.transform(Xmat_train)
    X_test_std = sc.transform(Xmat_test)


    ############ Logistic regression for reduced features
    logreg_red = LogisticRegression()
    logreg_red.fit(X_train_std, Ymat_train)
    y_pred_class = logreg_red.predict(X_test_std)
    all_probs = logreg_red.predict_proba(X_test_std)
    probs = logreg_red.predict_proba(X_test_std)[:,1]
    print("Show the orig probs shape")
    print(probs.shape)
    thresholdClass = userThreshold(all_probs, 0.5)
    # calculate accuracy
    g_all_probs = all_probs
    g_Ymat_test = Ymat_test
    g_y_probs= probs
    y_true = pd.Series(Ymat_test, name="Actual")
    y_pred = pd.Series(thresholdClass, name="Predicted")
    df_confusion = pd.crosstab(y_true, y_pred)
    #print(np.unique(y_pred_class,return_counts=True ))
    ##accuracy_val = metrics.accuracy_score(Ymat_test, y_pred_class)
    #print(accuracy_val)
    scores = cross_val_score(logreg_red, X_mat, Y_mat, cv=10)
    cv_score = scores.mean()
    #ls_file = filename.split('.')
    #For ROC Curve we give the actual test labels and the predicted probabilities of the positive class
    fpr, tpr, threshout = metrics.roc_curve(Ymat_test, probs)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    #### Save the regression model as a pickle file to be used later for prediction
    with open('LogisticRegModel_'+ filename.split('.')[0][:-6] +'.pickle', 'wb') as handle:
        pickle.dump(logreg_red, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #style = '<style> body{background-color:orange;font-family: "Arial", "Helvetica", sans-serif} h2{margin-left: 5px;} </style>'
    return render_template('TrainResponse.html',cfmtrx=[df_confusion.to_html(classes='cfm')], roc_data = [fpr.tolist(), tpr.tolist()],cvs = cv_score*100,rocAcc =roc_auc )

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#    Function to read and preprocess the file
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def readFileAndPreprocess(filename,callType,DEBUG):
    try:
        df = pd.read_excel(filename, index_col=0)
        print(df.shape)
        df = df.astype(str)
    ####Make blank values as nan and drop such rows
        df = df.apply(lambda x: x.str.strip()).replace('', np.nan)
        df = df.dropna()

###########################################################
#       Handle the columns with No_internet_service value
###########################################################
        df = df.replace(['No_internet_service'],[None])
        d = {'Yes': 1, 'No': 0 , 'None': 0}
        df['StreamingMovies'] = df['StreamingMovies'].map(d).fillna(0)
        df['StreamingTV'] = df['StreamingTV'].map(d).fillna(0)
        df['TechSupport'] = df['TechSupport'].map(d).fillna(0)
        df['DeviceProtection'] = df['DeviceProtection'].map(d).fillna(0)
        df['OnlineBackup'] = df['OnlineBackup'].map(d).fillna(0)
        df['OnlineSecurity'] = df['OnlineSecurity'].map(d).fillna(0)
        if(DEBUG):
            df['Churn'] = df['Churn'].map(d).fillna(0)

#############################################################
#     Handle the multiple value columns by creating new dummy variables
#############################################################
        df = pd.get_dummies(df, columns=["Contract", "PaymentMethod","InternetService", "MultipleLines", "gender"], prefix=["Contract", "PayMethod","Internet","Lines", "Gender"])


#############################################################
##  Convert Yes/No to 1 and 0
#############################################################

        d = {'Yes': 1, 'No': 0}
        df['Partner'] = df['Partner'].map(d)
        df['Dependents'] = df['Dependents'].map(d)
        df['PhoneService'] = df['PhoneService'].map(d)
        df['PaperlessBilling'] = df['PaperlessBilling'].map(d)

        X=df
        Y=''
###############################################################
#separate the label and save to disk
###############################################################

        if callType == 'Train':
            df['Churn'] = df['Churn'].map(d)
            df_churn = df['Churn']
            df = df.drop('Churn', axis =1)
            Y=df_churn

            X= dropCollinearFeatures(df, 0.7) ##Drop collinear features
            writer = pd.ExcelWriter('PreProcessed_data_ult.xlsx')
            X.to_excel(writer,'Sheet1')
            df_churn.to_excel(writer,'Sheet2')
            writer.save()
        print("Coming out of read and preprocess")
        return [X,Y]
    #-----------------------end of function
    except Exception as e:
     return (str(e))

def dropCollinearFeatures(dataset, threshold):
    dataset = dataset.astype(float)
    col_tupl = set()
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        #print("the value of i is {}".format(i))
        for j in range(i):
            #print("the value of j is {}".format(j))
            if corr_matrix.iloc[i, j] >= threshold:

                coli = corr_matrix.columns[i]  # getting the name of column
                colj = corr_matrix.columns[j]
                col_tupl.add((coli, colj))
                # getting the name of column
                if (coli in col_corr):
                    if coli in dataset.columns:
                        del dataset[coli]
                        print('deleting i column {}'.format(coli))

                if (colj in col_corr):
                    if colj in dataset.columns:
                        del dataset[colj]
                        print('deleting j column {}'.format(coli))
                col_corr.add(coli)
                col_corr.add(colj)

                if corr_matrix.iloc[i, j] >= 0.8:
                    if coli in dataset.columns:
                        print('deleting >0.8 column {}'.format(coli))
                        del dataset[coli]

    #print(col_corr)
    return dataset

# @ Display dynamic Confusion Matrix
@app.route('/display_CM', methods=['GET', 'POST'])
def display_CM():
    print("coming inside the AJAX called")
    print(g_all_probs)
    print("DBG1")
    CM_thresh = request.form['CM_threshold']
    print("DBG1")
    print(CM_thresh)
    print(g_all_probs.shape)
    dyn_class = userThreshold(g_all_probs, CM_thresh)
    confusion_matrix = ConfusionMatrix(g_Ymat_test, dyn_class)
    print("Confusion matrix:\n%s" % confusion_matrix)

    from sklearn.metrics import confusion_matrix
    cfm = confusion_matrix(g_Ymat_test, dyn_class)
    print(cfm)
    print(cfm[0][0])
    print(cfm[0][1])

    return jsonify({"C00":str(cfm[0][0]), "C01":str(cfm[0][1]), "C10":str(cfm[1][0]), "C11":str(cfm[1][1])})

if __name__ == "__main__":
    app.run(debug=True)