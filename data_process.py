import cv2
import numpy as np
import toFindtime
import os
import test_update
import test_temp
from skimage import io, color, data
np.set_printoptions(threshold=np.inf)

def synchronization(path, startvalue, endvalue):
    """
    read the picture that have been saved and convert to 0-1 image array
    :return:
    """
    temp_paths = os.listdir(path,)
    paths=[]
    for i in range(len(temp_paths)):
        if i>=startvalue and i<=endvalue:
            paths.append(str(i)+'.jpg')
        if i >endvalue:
            break
    pathC = []
    image_array = None
    for picname in paths:
        pathC.append(os.path.join(path, picname))
    print pathC[0],pathC[-1]

    index=0
    for eachpic in pathC:
        im = io.imread(eachpic, as_gray=False)
        im = color.rgb2gray(im)
        rows, cols = im.shape
        im = im.astype(np.float32)
        max = np.max(im)
        min = np.min(im)
        im=(im - min)/(max - min)
        image_array = np.array([im]) if image_array is None else np.append(image_array, [im], axis=0)

    return image_array

def picture_select(path, startvalue, endvalue):
    origin_path = '../data_523/%s/img/' % path
    save_path='../data_523/%s/picture_no_background/' %path
    paths = []
    temp_paths = os.listdir(origin_path,)
    for i in range(len(temp_paths)):
        paths.append(str(i) + '.jpg')
    #print paths
    pathC = []
    image_array = None
    for picname in paths:
        pathC.append(os.path.join(origin_path, picname))
    print pathC[-1]
    index=0
    for eachpic in pathC:
        im = io.imread(eachpic, as_gray=False)
        im = color.rgb2gray(im)
        # if index ==246:
        #     print im
        if index>=startvalue and index<=endvalue:
            cv2.imwrite(save_path+str(index)+'.jpg',im)
        index+=1


def index_find(path, startvalue, endvalue):
    """
    1) linear operation for csi timestamp
    2) find the same timestamp S between CSI packet and vedio
    3) save the picture from S to the end of CSI packet
    :return: S and length of csi stamp
    """
    """
    *****linear operation for csi timestamp*****
    """
    sysStamp = np.loadtxt('../data_523/%s/time.txt'%(path))
    sysStamp = [x / 1000.0 for x in sysStamp]
    start = sysStamp[0]
    time = []
    for times in range( len(sysStamp)):
        time.append((sysStamp[times] - start))
    timestamp = interp(start, time)
    file = open('../data_523/%s/ctime_new.txt'  % (path), 'w')
    for item in timestamp:
        file.write(str(item) + '\n')
    file.close()

    """
    ***** find the same timestamp S between CSI packet and vedio*****
    """
    Ctimelocation  = '../data_523/%s/ctime_new.txt' % (path)
    Vtimelocation  = '../data_523/%s/vtime.txt' % (path)

    Ctimestamp = np.loadtxt(Ctimelocation)
    Vtimestamp = np.loadtxt(Vtimelocation)
    Ctimestamp = [x  for x in Ctimestamp]
    time_start_Index = Vtimestamp[startvalue]
    time_end_Index = Vtimestamp[endvalue]
    error_1, index_start = toFindtime.getTimeIndex(Ctimestamp, time_start_Index)
    error_2, index_end = toFindtime.getTimeIndex(Ctimestamp, time_end_Index)


    return index_start, index_end

def interp(start, timestamp):

    blockedT = []
    flag, count = 0, 0
    for tIndex in range(1, len(timestamp)):
        if timestamp[tIndex] - timestamp[tIndex - 1] > test_update.TIMEINYERVAL * 1:
            numOfInterp = int((timestamp[tIndex] - timestamp[tIndex - 1]) / test_update.TIMEBIASE)
            for num in range(0, numOfInterp-1):
                blockedT.append(tIndex )
        else:
            pass

    for timeIndex in blockedT:
        timestamp.insert(timeIndex + count, 0)
        count += 1
    for times in range(1, len(timestamp)):
        if timestamp[times] == 0:
            timestamp[times] = timestamp[times-1] + test_update.TIMEINYERVAL

    return [x + start  for x in timestamp ]
