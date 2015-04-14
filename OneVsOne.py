'''
this file use the one vs one svm to train the model,
each model is named by the form of 'mode_orginallabel1_orginallabel2_model',
and label1 is orginallabeled as 1 while orginallabel2 is label as -1,.
all the model is stored in the OneVsOneModel,you can come to see what is in it
Attention:we show use the cross validation method to find the best parameters
'''
from numpy import *
import os
import sys
import time
from svmutil import *
from FileOperation import LoadData
trainfiledir = './trainfile/'
OneVsOneModel= './OneVsOneModel/'

def OneVsOneSvmTrain(traindir):
    print 'trainging'
    filename = [trainfiledir + str(f) for f in os.listdir(traindir)]
    filename.sort(key=lambda ff:int(ff[16:-4]))#sort the train in ascending order
    #print filename
    length  = len(filename)
    #print filename
    flagcmp = zeros((length,length))#if the two class has been cmp
    for i in range(length):
        iylabel,iymat,ixmat = LoadData(filename[i],1)
        for j in range(length):
            if( i==j or flagcmp[i][j] == 1 or flagcmp[j][i] == 1):
                continue
            flagcmp[i][j]=1
            flagcmp[j][i]=1
            featuremat = []
            labelmat = []
            jylabel,jymat,jxmat = LoadData(filename[j],-1)
            featuremat.extend(ixmat)
            featuremat.extend(jxmat)
            labelmat.extend(iymat)
            labelmat.extend(jymat)
            prob = svm_problem(labelmat,featuremat,isKernel=True)
            param = svm_parameter('-t 2 -g 128 -c 4')#use the cross validation(grip.py) to find the best c ang gamma
            #param = svm_parameter('-t 0')
            #param = svm_parameter('-t 1 -d 10 -r 1 -g 10')  
            m = svm_train(prob,param)
            modelname = OneVsOneModel+'model_'+str(i)+'_'+str(j)+'_model'
            svm_save_model(modelname,m)
    print 'trainging end'
def GetModelDigit(modelstr):#return which two class does the modelstr belong to 
    tmp = modelstr.strip().split('_')
    return int(tmp[1]),int(tmp[2])

def FindTheMaxLabelIndex(labelcount):
    m,n = shape(labelcount)
    mmin = -1
    index = -1
    for i in range(m):
        if(labelcount[i][0] > mmin):
            mmin = labelcount[i][0]
            index = i
    return int(index)

def OneVsOneTest(modeldir,testfile):
    try:
        frtest = open(testfile)
    except Exception,e:
        print ' error '
    
    labelmat = []
    datamat = []
    for line in frtest.readlines():
        linearr = line.strip().split(' ')
        labelmat.append(int(linearr[0]))
        featdict = {}
        for i in range(1,len(linearr)):
            feature = linearr[i].strip().split(':')
            featdict[int(feature[0])] = float(feature[1])
        datamat.append(featdict)
    labelset = set(labelmat)
    labelnum = max(labelset) + 1
    m = shape(datamat)[0]
    modelname = [OneVsOneModel + str(md) for md in os.listdir(modeldir)]
    errorcount = 0
    modellength = len(modelname)
    for i in range(m):
        labelcount = mat(zeros((labelnum,1)))
        for j in range(modellength):
            label1,label2 = GetModelDigit(modelname[j])
            model  = svm_load_model(modelname[j])
            tmpdata = []
            tmpdata.append(datamat[i])
            tmplabel = []
            tmplabel.append(labelmat[i])
            p_label, p_acc, p_val = svm_predict(tmplabel, tmpdata, model)
            if(p_val[0][0]>=0):
                labelcount[label1][0] = labelcount[label1][0] + 1
            else:
                labelcount[label2][0] = labelcount[label2][0] + 1
        
        maxlabelindex = FindTheMaxLabelIndex(labelcount)
        if(int(labelmat[i]) != maxlabelindex):
            print 'the ' ,i ,' case is classified wrong,it belongs to',labelmat[i]
            errorcount = errorcount + 1
    err =  (errorcount*1.0)/m
    print 'the total error rate is  ',err
    return err
if __name__ =='__main__':
    t1 = time.clock()
    OneVsOneSvmTrain('./trainfile')
    t2 = time.clock()
    print 'training cost %.4f ' % (t2-t1)
    OneVsOneTest('./OneVsOneModel','./test.txt')
