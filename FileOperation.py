'''
the FilePretreatment() function just divide a big file with multiple class
label into different file whose name is defined by the
class label from 0 to k,
LoadData() load the data set from given filepath
'''
from numpy import *
import sys

trainfile = './trainfile/'
def LoadData(filepath,label):
    try:
        fr = open(filepath)
    except Exception,e:
        print 'error'
    classlabel = int(filepath[16:-4])#return the orginal class label
    xmat = []
    ymat = [] 
    for line in fr.readlines():
        ymat.extend([label])
        linearr = line.strip().split(' ')
        featdict = {}
        for i in range(1,len(linearr)):
            feature = linearr[i].strip().split(':')
            featdict[int(feature[0])] = float(feature[1])
        xmat.append(featdict)
    return classlabel,ymat,xmat
def FilePretreatment(filename):
    try:
        fr = open(filename)
    except Exception ,e:
        print "error"
    labelmat = []
    datamat = []
    for line in fr.readlines():
        linearr = line.strip().split(' ')
        labelmat.append(str(linearr[0]))
        string=''
        for i in range(len(linearr)):
            string = string + str(linearr[i]) +' '
        string = string.strip()
        datamat.append(string)
    fr.close()
    labelset = set(labelmat)
    filename = []
    for y in labelmat:
        if(y  in labelset and len(labelset) !=0):
            filename.append(trainfile+'file'+str(y)+'.txt')#create file for different class label
            labelset.discard(y)
    length = len(datamat)
    fpointer = []
    for fn in filename:
        ff = open(fn,'wd')#open with append mode
        fpointer.append(ff)
    for i in range(length):
        digit = str(labelmat[i])
        for j in range(len(filename)):#find which file the data belongs to
            filedigit = str(filename[j][16:-4])
            if(digit == filedigit):
                string = str(datamat[i])+'\n'
                fpointer[j].write(string)
                break#already find one,jump out the circulation
    for fp in fpointer:
        fp.close()
if __name__ =='__main__':
    FilePretreatment(sys.argv[1])
        
        
    
            
            
            
