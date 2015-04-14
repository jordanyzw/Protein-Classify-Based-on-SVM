'''
this file use partvspart svm to classify the problem,First it constructs k(k-1)/2 two class problem,
And then it further divides each two class probelminto a small one , here I just divide each two problem
into 9 subproblem,then use the min_max modularto find the correct one,Finally use the max vote to tell
where the test example belongs to
'''
from svmutil import *
from numpy import *
import random
import os
import time
from operator import itemgetter,attrgetter
from FileOperation import LoadData
__all__=['PartVsPartSvmTrain','PartVsPartSvmTest']
PVPSVM = './PartVsPartSvmModel/'
trainfiledir='./trainfile/'
def PartVsPartSvmTrain(traindir,rate):
    filename = [trainfiledir + str(f) for f in os.listdir(traindir)]
    filename.sort(key=lambda ff:int(ff[16:-4]))#sort the train in ascending order
    length  = len(filename) 
    flagcmp = zeros((length,length))#if the two class has been cmp
    for i in range(length):
        iylabel,iymat,ixmat = LoadData(filename[i],1)
        
        for j in range(length):
            if( i==j or flagcmp[i][j] == 1 or flagcmp[j][i] == 1):
                continue
            flagcmp[i][j]=1
            flagcmp[j][i]=1
            jylabel,jymat,jxmat = LoadData(filename[j],-1)
           
            randomchoose(iylabel,iymat,ixmat,jylabel,jymat,jxmat,rate)

def randomchoose(iylabel,iymat,ixmat,jylabel,jymat,jxmat,rate):
    length1 = len(ixmat)
    length_1 = len(jxmat)

    len1 = int(rate * length1)
    len_1 = int(rate * length_1)
    tmplabelmat1 = []
    tmplabelmat_1 = []
    tmpdatamat1 = []
    tmpdatamat_1 = []
    for i in range(3):#random choose the data from the two class and combine them later
        array1 = [k for k in range(length1)]
        array_1 = [k for k in range(length_1)]
        random.shuffle(array1)
        random.shuffle(array_1)
        tmp1 = []
        tmpd1 = []
        for j in range(len1):
            tmp1.extend([iymat[array1[j]]])
            tmpd1.extend([ixmat[array1[j]]])
        tmplabelmat1.append(tmp1)
        tmpdatamat1.append(tmpd1)
        tmp_1 = []
        tmpd_1 = []
        for j in range(len_1):
            tmp_1.extend([jymat[array_1[j]]])
            tmpd_1.extend([jxmat[array_1[j]]])
        tmplabelmat_1.append(tmp_1)
        tmpdatamat_1.append(tmpd_1)
    
    for i in range(3):
        for j in range(3):
            labelmat= []
            datamat = []
            labelmat.extend(tmplabelmat1[i])
            labelmat.extend(tmplabelmat_1[j])
            datamat.extend(tmpdatamat1[i])
            datamat.extend(tmpdatamat_1[j])
            prob = svm_problem(labelmat,datamat,isKernel=True)
            param = svm_parameter('-t 1 -d 10 -r 1 -g 10')
            #param = svm_parameter('-t 0')
            #param = svm_parameter('-t 2 -g 128 -c 4')
            m = svm_train(prob,param)
            modelname = PVPSVM + 'model_'+str(iylabel)+'_'+str(jylabel)+'_'+str(i)+'_'+str(j)+'_model'
            svm_save_model(modelname,m)
def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_keys(s):
    return[tryint(c) for c in GetModelDigit(s)]
def PartVsPartSvmTest(modeldir,testfile):
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
    modelname = [PVPSVM + str(md) for md in os.listdir(modeldir)]
    modelname.sort(key = alphanum_keys)
    errorcount = 0
    modellength = len(modelname)
    errorcount = 0
    modellength = len(modelname)
    for i in range(m):
        labelcount = mat(zeros((labelnum,1)))
        submodeldict ={}
        for j in range(modellength):
            label1,label2,sublabel1,sublabel2 = GetModelDigit(modelname[j])
            model = svm_load_model(modelname[j])
            tmpdata = []
            tmpdata.append(datamat[i])
            tmplabel = []
            tmplabel.append(labelmat[i])
            p_label, p_acc, p_val = svm_predict(tmplabel, tmpdata, model)
            string = str(label1)+'_'+str(label2)
            if( string not in submodeldict.keys()):
                submodeldict[string] = []
            submodeldict[string].extend([p_val[0][0]])
        
        for k in submodeldict.keys():
            labelstr = str(k).split('_')
            tmpL1 = int(labelstr[0])
            tmpL_1 = int(labelstr[1])
            value = max(min(submodeldict[k][0:3]),min(submodeldict[k][3:6]),min(submodeldict[k][6:9]))
            if(value>0):
                labelcount[tmpL1][0] = labelcount[tmpL1][0] + 1
            else:
                labelcount[tmpL_1][0] = labelcount[tmpL_1][0] + 1                
        maxlabelindex = FindTheMaxLabelIndex(labelcount)
        
        if(int(labelmat[i]) != maxlabelindex):
            print 'the ' ,i ,' case is classified wrong,it belongs to',labelmat[i]
            errorcount = errorcount + 1
    err =  (errorcount*1.0)/m
    print 'the total error rate is  ',err
    return err
def GetModelDigit(modelname):
    tmp = modelname.strip().split('_')
    return int(tmp[1]),int(tmp[2]),int(tmp[3]),int(tmp[4])
def FindTheMaxLabelIndex(labelcount):
    m,n = shape(labelcount)
    mmin = -1
    index = -1
    for i in range(m):
        if(labelcount[i][0] > mmin):
            mmin = labelcount[i][0]
            index = i
    return int(index)
if __name__ =='__main__':
    t1 = time.clock()
    PartVsPartSvmTrain('./trainfile',0.8)
    t2 = time.clock()
    print 'training cost %.4f ' % (t2-t1)
    PartVsPartSvmTest('./PartVsPartSvmModel','./test.txt')














    
