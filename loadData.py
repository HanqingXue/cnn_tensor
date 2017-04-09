import numpy as np

def prasedata():
  train = open('zip.train')
  LabelList = []
  trainList = []
  linecount = 1
  for t in train:
    label = [0]*10
    t = t.split()
    m = map(float, t)
    m =  (list(m))
    zero = [0]*(28*28 - 16*16)

    trainList.append(m[1:] + zero)
    label[int(m[0])] = 1
    LabelList.append(label)
    if linecount == 7000:
        break
    linecount += 1

  trainData  = np.array(trainList)
  trainLabel = np.array(LabelList)
  return trainData, trainLabel

def praseTestdata():
  train = open('zip.test')
  LabelList = []
  trainList = []
  linecount = 1
  for t in train:
    label = [0]*10
    t = t.split()
    m = map(float, t)
    m =  (list(m))
    zero = [0]*(28*28 - 16*16)
    trainList.append(m[1:]+zero)
    label[int(m[0])] = 1
    LabelList.append(label)

  trainData  = np.array(trainList)
  trainLabel = np.array(LabelList)
  return trainData, trainLabel

trainData, trainLabel = prasedata()
print (trainData[0:100].shape)