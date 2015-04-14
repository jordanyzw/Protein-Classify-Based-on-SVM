'''this file use 0neVsRest method to classify the data'''
from numpy import *
import os
from svmutil import *
import time
from FileOperation import LoadData
OneVsRestModel = './OneVsRestModel/'
trainfiledir = './trainfile/'

def OneVsRestSvmTrain(traindir):
    filename = [trainfiledir + str(f) for f in os.listdir(traindir)]
    filename.sort(key=lambda ff:int(ff[16:-4]))#sort the train in ascending order
    length = len(filename)
    for i in range(length):
        iylabel,iymat,ixmat = LoadData(filename[i],1)
        labelmat = []
        datamat = []
        labelmat.extend(iymat)
        datamat.extend(ixmat)
        for j in range(length):
            if(i==j):
                continue
            jylabel,jymat,jxmat = LoadData(filename[j],-1)
            labelmat.extend(jymat)
            datamat.extend(jxmat)
            prob = svm_problem(labelmat,datamat,isKernel=True)
            param = svm_parameter('-t 2 -g 128 -c 4')
            #param = svm_parameter('-t 0')
            #param = svm_parameter('-t 1 -g 10 -r 1 -d 10')
            m = svm_train(prob,param)
            modelname = OneVsRestModel + 'model_'+str(i)+'_model'
            svm_save_model(modelname,m)
def GetModelDigit(modelstr):
    tmp = modelstr.strip().split('_')
    return int(tmp[1])

def FindTheMaxLabelIndex(labelcount):
    m,n = shape(labelcount)
    mmin = -1
    index = -1
    for i in range(m):
        if(labelcount[i][0] > mmin):
            mmin = labelcount[i][0]
            index = i
    return int(index)
def OneVsRestTest(modeldir,testfile):
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
    m = shape(datamat)[0]
    labelset = set(labelmat)
    labelnum = max(labelset) + 1
    modelname = [OneVsRestModel + str(md) for md in os.listdir(modeldir)]
    errorcount = 0
    modellength = len(modelname)
    for i in range(m):
        labelmax = mat(zeros((labelnum,1)))
        for  j in range(modellength):
            label = GetModelDigit(modelname[j])
            model  = svm_load_model(modelname[j])
            tmpdata = []
            tmpdata.append(datamat[i])
            tmplabel = []
            tmplabel.append(labelmat[i])
            p_label, p_acc, p_val = svm_predict(tmplabel, tmpdata, model)
            labelmax[label] = p_val[0][0]
        maxlabelindex = FindTheMaxLabelIndex(labelmax)
        if(int(labelmat[i]) != maxlabelindex):
            print 'the ' ,i ,' case is classified wrong'
            errorcount = errorcount + 1
    err =  (errorcount*1.0)/m
    print 'the total error rate is  ',err
    return err
if __name__ =='__main__':
    t1 = time.clock()
    OneVsRestSvmTrain('./trainfile')
    t2 = time.clock()
    print 'training cost %.4f ' % (t2-t1)
    OneVsRestTest('./OneVsRestModel','./test.txt')
            
        
