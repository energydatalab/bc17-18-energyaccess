import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
import seaborn as sns
from seaborn import heatmap
from matplotlib import pyplot as plt

def pca_transform_all(dfs, labels):
    pca_dfall = pca_transform(dfs[0], labels[0])
    pca_df100 = pca_transform(dfs[1], labels[1])
    pca_df400 = pca_transform(dfs[2], labels[2])
    return pca_dfall, pca_df100, pca_df400

def pca_transform(data, labels):
    n_components = 104
    pca = PCA(n_components=n_components)
    return pd.DataFrame(pca.fit_transform(data, labels))

info_columns = ['Percentage Electrified', 
                'Number of Electrified Households', 
                'Unnamed: 0', 
                'Census 2011 ID', 
                'Number_of_Households', 
                'eH', 'HH', 'eHr', 'CEN_2011']

def delete_useless(df):
    for col_name in info_columns:
        try:
            del df[col_name]
        except:
            print('Column %s does not exist' % (col_name))
            continue
    
    for col_name in df.columns:
        if df[col_name].dtype == "object":
            try:
                del df[col_name]
            except:
                print('Failed to delete column %s' % (col_name))
                continue

def nan_detect(df):
    print('NaN left:', df.columns[df.isna().any()].tolist())

def scale_df(df):
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(df)

def data_clean(file_path, labels=True, pca_transform=None):
    print('Begin data cleaning.')
    try:
        df = pd.read_csv(file_path)
        print('Read in data from %s sucessfully.' % (file_path))
    except:
        raise NameError('Incorrect file path.')
    
    df = df.dropna(how = 'all')
    df = df[~np.isnan(df.eH)]
    df_all = df.assign(eHr = lambda x: 100. * x.eH / x.HH).fillna(-1)
    try:
        # df_all = df.rename(columns={'Number of Households': 'Number_of_Households'})
        df_100 = df_all.query('HH > 100')
        df_400 = df_all.query('HH > 400')
        df_list = [df_all, df_100, df_400]
        labels_list = [(df['eHr'] >= 10).tolist() for df in df_list]
        print('All dataframes and labels generated.')
    except:
        raise ValueError('Failed to process dataframes from %s.' % (file_path))
    
    try:
        for df in df_list:
            delete_useless(df)
        print('Deleting useless and predictive informative columns')
    except:
        raise ValueError('Failed to delete useless and predictive informative columns')
    
    try:
        if pca_transform:
            df_list = pca_transform_all(df_list, labels_list)
            print('PCA transformation successful.')
    except:
        raise ValueError('Failed to execute PCA transformation.')
    
    print('Data cleaning successful.')
    try:
        for df in df_list:
            nan_detect(df)
    except:
        raise ValueError('NaN fatal errors.')

    return {'df':df_list, 'label':labels_list}

def plot_pca(data_list, label_list, max_conponents, save_path=None):
    plt.figure(figsize=(20,20))
    result = {}
    for data, label in zip(data_list, label_list):
        PCA_components = []
        score = []
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data)
        for i in range(1,min(max_conponents, len(data.columns))):
            pca = PCA(n_components=i)
            PCA_components.append(i)
            pca.fit(data_transformed, label)
            var_values = pca.explained_variance_ratio_
            score.append(sum(var_values))
            if i not in result:
                result[i] = []
            result[i].append(sum(var_values))
        plt.plot(PCA_components, score)
    plt.ylabel("Expalined variance", fontsize=18)
    plt.xlabel("number of features employed", fontsize=18)
    plt.title("Principle Component Analysis", fontsize=22)
    
    if save_path==None:
        print('Dispalying the plot.')
        plt.show()
    else:
        print('Saving to %s' % (save_path))
        plt.savefig(save_path)
        plt.close()
        
    return result

def gdb_machine(df_list, label_list, penalty=1, scale=False):
    random_state = np.random.RandomState(20180213)
    gdb_results = {'prediction':[], 'probaility':[], 'y_test':[], 'y_score':[]}
    try:
        if scale:
            df_list = [scale_df(df) for df in df_list]
            print('DF Scaling successful.')
    except:
        raise ValueError('Failed to execute DF Scaling.')

    for x, y in zip(df_list, label_list):
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=random_state)
        except:
            raise ValueError('Train/Test split failed.')
        
        #df = pd.DataFrame(x_train).assign(outcome=y_train)
        #df = pd.concat([df[df.outcome==1].sample(n=5 * sum(np.array(y_train)==0)), df[df.outcome==0]])
        #x_train = df.drop(['outcome'])
        #y_train = df.outcome
        #del df
        gdb = GradientBoostingClassifier(n_estimators=7, 
                                         max_depth=6,
                                         min_samples_split=1780, 
                                         min_samples_leaf=1,
                                         random_state=20180320,
                                         max_features=310, 
                                         subsample=0.8, 
                                         learning_rate=0.11)
        
        weighting = lambda x:1 if x else penalty
        gdb.fit(x_train, y_train, sample_weight=[weighting(i) for i in y_train])
        
        gdb_results['y_test'].append(y_test)
        gdb_results['prediction'].append(gdb.predict(x_test))
        gdb_results['probaility'].append(gdb.predict_proba(x_test)[::,1])
        gdb_results['y_score'].append(gdb.decision_function(x_test))
    return gdb_results

def voting_process(df_list, label_list, scale=False):
    random_state = np.random.RandomState(20180213)
    vt_results = {'prediction':[], 'probaility':[], 'y_test':[], 'y_score':[]}
    try:
        if scale:
            df_list = [scale_df(df) for df in df_list]
            print('DF Scaling successful.')
    except:
        raise ValueError('Failed to execute DF Scaling.')
        
    for x, y in zip(df_list, label_list):
        
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=random_state)
        except:
            raise ValueError('Train/Test split failed.')
        vt = VotingClassifier(estimators=[('basic_log', LogisticRegression()),
                                          ('et', ExtraTreesClassifier()), 
                                          ('ada', AdaBoostClassifier()), 
                                          ('rf', RandomForestClassifier()), 
                                          ('gbm', GradientBoostingClassifier(n_estimators=100, 
                                                                             max_depth=5, 
                                                                             learning_rate=0.1))], 
                              voting='soft')
        weighting = lambda x:1 if x else 50
        vt.fit(x_train, y_train, sample_weight=[weighting(i) for i in y_train])
        
        vt_results['y_test'].append(y_test)
        vt_results['prediction'].append(vt.predict(x_test))
        vt_results['probaility'].append(vt.predict_proba(x_test)[::,1])
        try:
            vt_results['y_score'].append(vt.decision_function(x_test))
        except:
            vt_results['y_score'].append(vt.predict_proba(x_test)[::,1])
    return vt_results

def plot_roc_curve(pred_dict, filter_list, save_path=None):
    print('Plotting ROC/AUC plot.')
    plt.figure(figsize=(15,10))
    if filter_list == None or len(filter_list) != len(pred_dict['y_test']):
        filter_list = [str(i) for i in range(len(pred_dict['y_test']))]

    for test, prob, filter_title in zip(pred_dict['y_test'], pred_dict['probaility'], filter_list):
        fpr, tpr, thresholds = roc_curve(test, prob)
        plt.plot(fpr, tpr, label=str(filter_title)+"/AUC ="+str(roc_auc_score(test, prob).round(3)))
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Pure Chance")
    plt.legend(loc=4, fontsize=18)
    plt.xlabel('False Alarm Rate', fontsize=18)
    plt.ylabel('Correct Detection Rate', fontsize=18)
    plt.title('Gradient Boosting Classification ROC Curve', fontsize=26)
    if save_path==None:
        print('Dispalying the plot.')
        plt.show()
    else:
        print('Saving to %s' % (save_path))
        plt.savefig(save_path)
        plt.close()
    
def plot_confusion_matrix(pred, truth, title, save_path=None): 
    print('Plotting confusion matrix.')
    sns.set(font_scale=1.4)
    try:
        conf_matrix = confusion_matrix(truth, pred)
    except:
        raise ValueError('Confuction matrix generation failed.')
    plt.figure(figsize=(15,15))
    ticks = ['unelectrified', 'electrified']
    heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=ticks, yticklabels=ticks,
            annot_kws={"size":17}, square=True)
    plt.xlabel('Prediction', fontsize=18)
    plt.ylabel('Truth', fontsize=18)
    plt.title(title, fontsize=20)
    if save_path==None:
        print('Dispalying the plot.')
        plt.show()
    else:
        print('Saving to %s' % (save_path))
        plt.savefig(save_path)
        plt.close()
    
def plot_correlation_matrix(df, save_path=None):
    print('Plotting correlation matrix.')
    corr_matrix = df.corr()
    plt.figure(figsize=(20,15))
    heatmap(corr_matrix)
    plt.title("Pearson Correlation matrix", fontsize=20)
    if save_path==None:
        print('Dispalying the plot.')
        plt.show()
    else:
        print('Saving to %s' % (save_path))
        plt.savefig(save_path)
        plt.close()
    
def plot_precision_recall_curve(y_test, y_score, save_path=None):
    print('Plotting Precision-Recall Curve.')
    
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision), fontsize=26)
    if save_path==None:
        print('Dispalying the plot.')
        plt.show()
    else:
        print('Saving to %s' % (save_path))
        plt.savefig(save_path)
        plt.close()

if __name__=='__main__':
    df_label = data_clean('./interpolated_full.csv')
    pickle.dump(df_label, open('./df_label.pickle', 'wb'))
    plot_correlation_matrix(df_label['df'][0], './corr.png')
    plot_confusion_matrix(df_label['label'][0], 
                          sorted(df_label['label'][0]), 
                          'confusion matrix',
                          './confusion.png')
    plot_roc_curve(df_label['df'], df_label['label'], 
                   ["All households", "100+ households", "400+ households"], 
                   './roc.png')