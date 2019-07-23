
__author__ = 'Xin'
'''
Reference:
Multi-class AdaBoosted Decision Trees:
http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html
'''
import numpy 
#############randome seed:
#seed = 100
seed = 50
numpy.random.seed(seed)
#TensorFlow has its own random number generator
from tensorflow import set_random_seed
set_random_seed(seed)
####################
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
#from multi_AdaBoost import AdaBoostClassifier
from multi_AdaBoost import AdaBoostClassifier as Ada


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#####deep lCNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D

from sklearn.preprocessing import LabelBinarizer

#theano doesn't need any seed because it uses numpy.random.seed
#######function def:
def train_CNN(X_train=None, y_train=None, epochs=None, batch_size=None, X_test=None, y_test=None, n_features =10, seed =100):
    ######ranome seed
    numpy.random.seed(seed)
    set_random_seed(seed)
    
    model = baseline_model(n_features, seed)
    #reshape imput matrig to be compatibel to CNN
    newshape=X_train.shape
    newshape = list(newshape)
    newshape.append(1)
    newshape = tuple(newshape)
    X_train_r = numpy.reshape(X_train, newshape)#reshat the trainig data to (2300, 10, 1) for CNN
    #binarize labes:
    lb=LabelBinarizer()
    y_train_b = lb.fit_transform(y_train)
    #train CNN
    numpy.random.seed(seed)
    set_random_seed(seed)
    model.fit(X_train_r, y_train_b, epochs=epochs, batch_size=batch_size)
    
    #####################reshap test data and evaluate:
    newshape = X_test.shape
    newshape = list(newshape)
    newshape.append(1)
    newshape = tuple(newshape)
    X_test_r = numpy.reshape(X_test, newshape)
    #bibarize lables:
    lb=LabelBinarizer()
    y_test_b = lb.fit_transform(y_test)
    
    yp=model.evaluate(X_train_r, y_train_b)
    print('\nSingle CNN evaluation on training data, [loss, test_accuracy]:')
    print(yp)

    
    yp=model.evaluate(X_test_r, y_test_b)
    print('\nSingle CNN evaluation on testing data, [loss, test_accuracy]:')
    print(yp)
    ########################
#####deep CNN
def baseline_model(n_features=10, seed=100):
    numpy.random.seed(seed)
    set_random_seed(seed)
	# create model
    model = Sequential()
    model.add(Conv1D(32, 3, padding = "same", input_shape=(n_features, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
#    model.add(Conv1D(32, 3, border_mode='valid',  activation='relu'))
#    model.add(MaxPooling1D(pool_size=(2, 2)))
#    model.add(Dropout(0.2))#

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.add(Dropout(0.2))#
    model.add(Dense(64, activation='relu'))#

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
	# Compile model
    numpy.random.seed(seed)
    set_random_seed(seed)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    
    print (model.summary())
    return model
#from keras import backend as K
#K.set_image_dim_ordering('th')


#X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
#                               n_classes=3, random_state=1)
def synethetic_data (n_features=10,     n_classes=3):
    #generat randon synethetic data
    from sklearn.datasets import make_gaussian_quantiles
    #########################plot the hist and pie
    import matplotlib as mpl
    def plot_hist(y_test, oName0):
        
        mpl.rc('font', family = 'Times New Roman')
    #        (n, bins, patches)=plt.hist(y_train, bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25])
        (n, bins, patches)=plt.hist(y_test+1, bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25])#, 3.75, 4.25, 4.75, 5.25])
            
        #    (n, bins, patches)=plt.hist(y_train+1, bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25])
        plt.xlabel('Class')
        plt.ylabel('# Samples')
        oName = oName0+ '_hist.png'          
        plt.savefig(oName,dpi=200)
        plt.show()
        print_t = 'The Histogram of the data is saved as: ' + oName
        print(print_t)
        print (n)
        print(n/len(y_test))
            
       ########################Basci pie chart:
        labels = 'C1', 'C2', 'C3'#, 'C4', 'C5'
        sizes = [v for i, v in enumerate(n) if (i%2)==0]  
    #      sizes = [15, 30, 45, 10]
        explode = (0, 0, 0.1)#, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
            
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        oName = oName0 +'_pie.png'
        plt.savefig(oName,dpi=200)

        plt.show()
        print_t = 'The Histogram of the data is saved as: ' + oName
        print(print_t)
    
    #############################################
    #epochs=6
    
    X, y = make_gaussian_quantiles(n_samples=13000, n_features=n_features,
                                   n_classes=n_classes, random_state=1)
    
    n_split = 3000
    
    X_train, X_test = X[:n_split], X[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]
    
    #    df=pd.DataFrame({'a':y_train})
    N_re=[200, 500]
#    N_re=[200, 0]

    
    a=[index for index, v in enumerate(y_train) if v == 0]
    y_train=numpy.delete(y_train, a[0:N_re[0]]) 
    X_train=numpy.delete(X_train, a[0:N_re[0]], axis=0)     
    a=[index for index, v in enumerate(y_train) if v == 1]
    y_train=numpy.delete(y_train, a[0:N_re[1]])  
    X_train=numpy.delete(X_train, a[0:N_re[1]], axis=0)  
    ########plot hist and pie
    plot_hist(y_train, oName0 = 'synethetic_train')

    plot_hist(y_test, oName0 = 'synethetic_test')
    ###################
    return X_train, y_train, X_test, y_test

def reshape_for_CNN(X):
       ###########reshape input mak it to be compatibel to CNN
       newshape=X.shape
       newshape = list(newshape)
       newshape.append(1)
       newshape = tuple(newshape)
       X_r = numpy.reshape(X, newshape)#reshat the trainig data to (2300, 10, 1) for CNN

       return X_r
 
    
n_features=10   
n_classes=3

X_train, y_train, X_test, y_test = synethetic_data (n_features=n_features, n_classes = n_classes)
batch_size=10
#X_train_r, X_test_r = reshape_for_CNN()
X_train_r = reshape_for_CNN(X_train)
X_test_r = reshape_for_CNN(X_test)


###########################################Adaboost+CNN:

from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
n_estimators =10
epochs =1
bdt_real_test_CNN = Ada_CNN(
    base_estimator=baseline_model(n_features=n_features),
    n_estimators=n_estimators,
    learning_rate=1,
    epochs=epochs)
#######discreat:

bdt_real_test_CNN.fit(X_train_r, y_train, batch_size)
test_real_errors_CNN=bdt_real_test_CNN.estimator_errors_[:]



y_pred_CNN = bdt_real_test_CNN.predict(X_train_r)
print('\n Training accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(accuracy_score(bdt_real_test_CNN.predict(X_train_r),y_train)))

y_pred_CNN = bdt_real_test_CNN.predict(X_test_r)
print('\n Testing accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(accuracy_score(bdt_real_test_CNN.predict(X_test_r),y_test)))

##########################################single CNN:

train_CNN(X_train = X_train, y_train = y_train, epochs=10, 
          batch_size=batch_size ,X_test = X_test, y_test = y_test, 
          n_features=n_features, seed=seed)

