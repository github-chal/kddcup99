"""
    1.导入包，设置随机种子保证论文可复现
    2.以类的方式定义训练参数，并实例对象获取所有参数
    3.定义CallBack类（重写 keras.callbacks.Callback）
    4.定义处理数据集的函数
    5.定义自己的模型,设置loss，优化器等，开始训练
    6.绘图
    7.预测
"""
import random
import time
from collections import Counter
from datetime import datetime
import tensorflow as tf
import keras.callbacks
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from keras import Sequential
from keras.layers import LSTM,Dense,Dropout,Bidirectional
from keras import optimizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    classification_report,confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from keras.utils import plot_model
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import seaborn as sns

class Args:
    def __init__(self):
        self.patience=5
        self.batch_size = 128
        self.hidden_size = 128
        self.time_steps=1
        self.epochs = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = '../../data/kddcup99_15'
        self.model_path= '../../model/keras_lstm_best_15.keras'

def preprocess():
    data = args.data
    df = pd.read_csv(data, header=None)
    df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                  'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                  'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                  'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                  'num_access_files', 'num_outbound_cmds', 'is_host_login',
                  'is_guest_login', 'count', 'srv_count', 'serror_rate',
                  'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                  'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                  'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                  'dst_host_same_src_port_rate',
                  'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                  'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                  'dst_host_srv_rerror_rate', 'label']  # 42个特征列名
    df = df[['service','src_bytes','dst_bytes','duration','dst_host_count','dst_host_srv_diff_host_rate','dst_host_srv_count','hot','dst_host_rerror_rate','dst_host_diff_srv_rate',
'protocol_type','count','dst_host_same_srv_rate','flag','rerror_rate','dst_host_serror_rate','num_compromised','srv_count','diff_srv_rate','num_file_creations',
             'label']]
    attack_type = {
        'normal': 'normal',
        'back': 'dos',
        'buffer_overflow': 'u2r',
        'ftp_write': 'r2l',
        'guess_passwd': 'r2l',
        'imap': 'r2l',
        'ipsweep': 'probe',
        'land': 'dos',
        'loadmodule': 'u2r',
        'multihop': 'r2l',
        'neptune': 'dos',
        'nmap': 'probe',
        'perl': 'u2r',
        'phf': 'r2l',
        'pod': 'dos',
        'portsweep': 'probe',
        'rootkit': 'u2r',
        'satan': 'probe',
        'smurf': 'dos',
        'spy': 'r2l',
        'teardrop': 'dos',
        'warezclient': 'r2l',
        'warezmaster': 'r2l',
        'mscan': 'probe',
        'saint': 'probe',
        'apache2': 'dos',
        'mailbomb': 'dos',
        'processtable': 'dos',
        'udpstorm': 'dos',
        'httptunnel': 'u2r',
        'ps': 'u2r',
        'sqlattack': 'u2r',
        'xterm': 'u2r',
        'named': 'r2l',
        'sendmail': 'r2l',
        'snmpgetattack': 'r2l',
        'snmpguess': 'r2l',
        'worm': 'r2l',
        'xlock': 'r2l',
        'xsnoop': 'r2l'
    }
    # df['label'] = df['label'].apply(lambda r: attack_type[r[:-1]])
    # 划分成 2类,即 normal和 abnormal
    # df['label'] = df['label'].apply(lambda r: r[:] if r[:] == 'normal' else 'abnormal')
    # 数值化非数值型的特征
    le = LabelEncoder()
    df['protocol_type'] = le.fit_transform(df['protocol_type'])
    df['service'] = le.fit_transform(df['service'])
    df['flag'] = le.fit_transform(df['flag'])
    df['label'] = le.fit_transform(df['label'])
    # 独热编码处理
    proc=pd.DataFrame(keras.utils.to_categorical(df['protocol_type'],num_classes=3))
    service=pd.DataFrame(keras.utils.to_categorical(df['service'],num_classes=70)) ## 实际上只有66种
    flag=pd.DataFrame(keras.utils.to_categorical(df['flag'],num_classes=11))
    df=pd.concat([df,proc,service,flag],axis=1)
    df=df.drop(labels=['protocol_type','flag','service'],axis=1)
    # df=pd.concat([df,service],axis=1)
    # df=df.drop(labels=['service'],axis=1)
    # 划分数据集
    Y=df['label']
    X=df.drop('label',axis=1)

    label = X.shape[1]
    X.columns=list(range(label))
    # 归一化：归一到(0,1)的范围
    for feature_name in range(label-1):
        X[feature_name] = MinMaxScaler().fit_transform(X[feature_name].values.reshape(-1,1))
    # 标准化：
    # for feature_name in range(label-1):
    #     X[feature_name]=StandardScaler().fit_transform(X[feature_name].values.reshape(-1,1))

    # 需要划分训练集和验证集
    x_train, x_test, y_train, y_test = train_test_split(X.values,Y.values,test_size=0.2,random_state=seed)
    print('y_train类别的分布',Counter(y_train))
    print('y_test类别的分布',Counter(y_test))
    x_train=np.reshape(x_train,(-1,args.time_steps,x_train.shape[1]))# input_shape=[sample,timesteps,feature]
    x_test=np.reshape(x_test,(-1,args.time_steps,x_test.shape[1]))
    y_train=keras.utils.to_categorical(y_train)
    # y_test=keras.utils.to_categorical(y_test,num_classes=5)
    return x_train, x_test, y_train, y_test

class model:
    def __init__(self):
        self.net=None
    def train(self,x_train,y_train):
        self.net = Sequential()
        self.net.add(LSTM(args.hidden_size,return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.net.add(Dropout(0.2))
        self.net.add(Dense(y_train.shape[1],activation='softmax'))
        optimizer=optimizers.Adam(learning_rate=0.007)
        self.net.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=optimizer)
        # 二分类
        # self.net.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer=optimizer)

        self.net.summary()
        history = self.net.fit(x_train,
                               y_train,
                               validation_split=0.2,
                               epochs=args.epochs,
                               verbose=1,
                               batch_size=args.batch_size,
                               callbacks=callbacks_list,
                               shuffle=True,
                            ).history
        """===============================plot==========================="""
        # plt.plot(history['loss'], linewidth=2, label='Train')
        # plt.plot(history['val_loss'], linewidth=2, label='Valid')
        # plt.legend(loc='upper right')
        # plt.title('epoch-loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.show()
        """画模型图"""
        # plot_model(self.net, to_file='../../图表/model-LSTM+BiLSTM.png', show_shapes = True)
    def test(self,x_test,y_test):
        yy_pred=self.net.predict(x_test,batch_size=args.batch_size,verbose=0)
        y_pred=np.argmax(yy_pred,axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro',zero_division=1)
        recall = recall_score(y_test, y_pred, average='micro',zero_division=1)
        f1 = f1_score(y_test, y_pred, average='micro',zero_division=1)
        print("使用微平均的PRF")
        print('Accuracy:{:.4f}'.format(accuracy))
        print('Precision:{:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('F1-measure: {:.4f}'.format(f1))

        # 两分类
        # print(classification_report(y_test, y_pred, digits=4, target_names=["abnormal", "normal"]))
        print(classification_report(y_test, y_pred, digits=4,target_names=["dos","normal","probe","r2l","u2r"]))
        # plt.title("Proposed method")
        # cm=confusion_matrix(y_test,y_pred)
        # cm=pd.DataFrame(cm,columns=["dos","normal","probe","r2l","u2r"],index=["dos","normal","probe","r2l","u2r"])
        # sns.heatmap(cm,fmt="d",cmap="OrRd",annot=True)
        # 两分类
        # cm = pd.DataFrame(cm, columns=["abnormal", "normal"], index=["abnormal", "normal"])
        # sns.heatmap(cm, fmt="d", cmap="Blues", annot=True)
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.show()

if __name__=='__main__':
    """设置随机种子保证论文可复现"""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    # 获取所有训练参数
    args = Args()
    # 定义回调函数
    callbacks_list=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience # 5轮内不改善就中断
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            monitor='val_loss',
            save_best_only=True # 只保存最佳模型
        )
    ]
    # 对原始数据集进行处理
    x_train, x_test, y_train, y_test=preprocess()
    # 实例化模型，开始训练
    model=model()
    start_time=time.time()
    model.train(x_train,y_train)
    # 获取测试集
    print(f"训练时间：{time.time()-start_time}")
    print('要测试的模型：',args.model_path)
    model.test(x_test,y_test)
