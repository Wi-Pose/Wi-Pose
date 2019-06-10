import threading
from read_file import read_bf_file
import numpy as np
import pylab
import pickle
from scipy.fftpack import fft, ifft

from DWTfliter import dwtfilter

TIMEINYERVAL = 0.01
TIMEBIASE = 0.095
TIMELEN = 1000
IMAGETOCSIRATIO = 2


def radReverse(subcarrier):
    return map(lambda x: float("%.2f" % np.arctan(x.imag / (x.real))), subcarrier)


def complexToLatitude(subcarrier):
    return map(lambda x: float("%.2f" % abs(x)), subcarrier)


def reviseInterp(timestamp, eachsubcarrier):
    blockedTime = []
    flag, count = 0, 0
    for tIndex in range(1, len(timestamp)):
        if timestamp[tIndex] - timestamp[tIndex - 1] > TIMEINYERVAL:
            numOfInterp = int(
                (timestamp[tIndex] - timestamp[tIndex - 1]) / TIMEBIASE)  
            for num in range(0, numOfInterp - 1):
                blockedTime.append(tIndex)

    ca = eachsubcarrier.tolist()
    for csiIndex in blockedTime:
        ca.insert(csiIndex + count, 0)
        count += 1

    for csiApt in range(0, len(ca) - 1):
        if ca[csiApt] == 0 and ca[csiApt + 1] != 0:
            ca[csiApt] = "%.2f" % ((ca[csiApt - 1] + ca[csiApt + 1]) / 2.0)
        elif ca[csiApt] == 0 and ca[csiApt + 1] == 0:
            for zeros in range(csiApt, len(ca)):
                if ca[zeros] != 0:
                    flag = zeros
                    break
            numOfZeros = flag - csiApt
            for num in range(0, numOfZeros):
                ca[csiApt + num] = ca[csiApt + num - 1] + float('%.2f' % ((ca[flag] - ca[csiApt - 1]) / numOfZeros))

    caNew = [float(x) for x in ca]
    blockedTime.extend([x for x in range(len(caNew), TIMELEN)])
    if len(caNew) < TIMELEN:
        caNew.extend([caNew[-1] for _ in range(0, TIMELEN - len(caNew))])
    else:
        caNew = caNew[:TIMELEN]

    return caNew, blockedTime


def compelx_reviseInterp(timestamp, eachsubcarrier):
    blockedTime = []
    flag, count = 0, 0
    for tIndex in range(1, len(timestamp)):
        if timestamp[tIndex] - timestamp[tIndex - 1] > TIMEINYERVAL:
            numOfInterp = int(
                (timestamp[tIndex] - timestamp[tIndex - 1]) / TIMEBIASE)  # todo:   Timebiase needed to be fixed
            for num in range(0, numOfInterp - 1):
                blockedTime.append(tIndex)

    ca_real = eachsubcarrier.real.tolist()
    ca_imag = eachsubcarrier.imag.tolist()
    for csiIndex in blockedTime:
        ca_real.insert(csiIndex + count, 0)
        ca_imag.insert(csiIndex + count, 0)
        count += 1

    for csiApt in range(0, len(ca_real) - 1):
        if ca_real[csiApt] == 0 and ca_real[csiApt + 1] != 0:
            ca_real[csiApt] = "%.2f" % ((ca_real[csiApt - 1] + ca_real[csiApt + 1]) / 2.0)
        elif ca_real[csiApt] == 0 and ca_real[csiApt + 1] == 0:
            for zeros in range(csiApt, len(ca_real)):
                if ca_real[zeros] != 0:
                    flag = zeros
                    break
            numOfZeros = flag - csiApt
            for num in range(0, numOfZeros):
                ca_real[csiApt + num] = ca_real[csiApt + num - 1] + float(
                    '%.2f' % ((ca_real[flag] - ca_real[csiApt - 1]) / numOfZeros))

        if ca_imag[csiApt] == 0 and ca_imag[csiApt + 1] != 0:
            ca_imag[csiApt] = "%.2f" % ((ca_imag[csiApt - 1] + ca_imag[csiApt + 1]) / 2.0)
        elif ca_imag[csiApt] == 0 and ca_imag[csiApt + 1] == 0:
            for zeros in range(csiApt, len(ca_imag)):
                if ca_imag[zeros] != 0:
                    flag = zeros
                    break
            numOfZeros = flag - csiApt
            for num in range(0, numOfZeros):
                ca_imag[csiApt + num] = ca_imag[csiApt + num - 1] + float(
                    '%.2f' % ((ca_imag[flag] - ca_imag[csiApt - 1]) / numOfZeros))

    caNew_real = [float(x) for x in ca_real]
    blockedTime.extend([x for x in range(len(caNew_real), TIMELEN)])
    if len(caNew_real) < TIMELEN:
        caNew_real.extend([caNew_real[-1] for _ in range(0, TIMELEN - len(caNew_real))])
    else:
        caNew_real = caNew_real[:TIMELEN]

    caNew_imag = [float(x) for x in ca_imag]
    blockedTime.extend([x for x in range(len(caNew_imag), TIMELEN)])
    if len(caNew_imag) < TIMELEN:
        caNew_imag.extend([caNew_imag[-1] for _ in range(0, TIMELEN - len(caNew_imag))])
    else:
        caNew_imag = caNew_imag[:TIMELEN]

    complex_list = []
    for i in range(len(caNew_real)):
        complex_list.append(caNew_real[i] + 1j * caNew_imag[i])
    return complex_list, blockedTime


def linearInterpolation(matrix, timestamp):
    raw, blockedTime = None, None
    for eachsubcarrier in matrix:
        eachsubcarrier, blockedTime = reviseInterp(timestamp, eachsubcarrier)
        raw = np.array([eachsubcarrier]) if raw is None else np.append(raw, [eachsubcarrier], axis=0)
    return raw, blockedTime

def complex_linearInterpolation(matrix, timestamp):
    raw, blockedTime = None, None
    for eachsubcarrier in matrix:
        eachsubcarrier, blockedTime = compelx_reviseInterp(timestamp, eachsubcarrier)
        raw = np.array([eachsubcarrier]) if raw is None else np.append(raw, [eachsubcarrier], axis=0)
    return raw, blockedTime

def varianceOperation(*args):
    var_list = [np.var(args[0]), np.var(args[1]), np.var(args[2])]
    list = [args[0], args[1], args[2]]
    print var_list[0],var_list[1],var_list[2]
    mini, maxi = var_list.index(min(var_list)), var_list.index(max(var_list))
    print maxi+1,mini+1
    secend_index=0
    for i in range(3):
        if i!=maxi and i!=mini:
           secend_index=i
    print secend_index+1

    return args[maxi], args[secend_index]


def relativePhaseOperation(pair_max, pairs):
    amp, relativePhase_one, relativePhase_two, relativePhase_three, relativePhase_four = None, None, None, None, None
    alpha, Phase_one, Phase_two = None, None, None
    amp_one, amp_two = None, None
    for subcarrier in pair_max:
        temp_amp = complexToLatitude(subcarrier)
        alpha = np.array([min(temp_amp)]) if alpha is None else np.append(alpha,
                                                                          [min(temp_amp)], axis=0)
    
    belta = alpha * 1000
   
    for subcarrier in pair_max:
        Phase_one = np.array([radReverse(subcarrier)]) if Phase_one is None else np.append(
            Phase_one, [radReverse(subcarrier)], axis=0)
        amp_one = np.array([complexToLatitude(subcarrier)]) if amp_one is None else np.append(amp_one,
                                                                                              [complexToLatitude(
                                                                                                  subcarrier)],
                                                                                              axis=0)
   
    for subcarrier in pairs:
        Phase_two = np.array([radReverse(subcarrier)]) if Phase_two is None else np.append(
            Phase_two, [radReverse(subcarrier)], axis=0)
        amp_two = np.array([complexToLatitude(subcarrier)]) if amp_two is None else np.append(amp_two,

                                                                                              [complexToLatitude(
                                                                                                  subcarrier)],
                                                                                              axis=0)

    for i in range(len(pair_max)):
        amp_one[i] = amp_one[i] + belta[i]
        amp_two[i] = amp_two[i] - alpha[i]

    sub_pair_max = amp_one * np.cos(Phase_one) + 1j * amp_one * np.sin(Phase_one)

    add_piar_two = amp_two * np.cos(Phase_two) + 1j * amp_two * np.sin(Phase_two)
    
    con_mul_one = sub_pair_max * (add_piar_two.conjugate())
    
    con_mul_one = np.nan_to_num(con_mul_one)  
    antenna_two = pair_max * (pairs.conjugate())
    flag = 0
    for subcarrier in con_mul_one:
        temp_mean = np.mean(subcarrier.real)
       
        subcarrier.real = subcarrier.real - temp_mean
      

    for subcarrier in pair_max:
        amp = np.array([complexToLatitude(subcarrier)]) if amp is None else np.append(amp,
                                                                                      [complexToLatitude(subcarrier)],
                                                                                      axis=0)
    for subcarrier in con_mul_one:
        relativePhase_one = np.array([radReverse(subcarrier)]) if relativePhase_one is None else np.append(
            relativePhase_one, [radReverse(subcarrier)], axis=0)

  
    for subcarrier in relativePhase_one:
        phase_temp = subcarrier[0]
        for i in range(len(subcarrier)):
       
            subcarrier[i]=round(subcarrier[i]-phase_temp,2)
            if subcarrier[i] >np.pi/2.0:
                subcarrier[i]=round(subcarrier[i]-np.pi,2)
            elif subcarrier[i] <-np.pi/2.0:
                subcarrier[i] = round(subcarrier[i] + np.pi,2)
           


    for subcarrier in antenna_two:
        relativePhase_two = np.array([radReverse(subcarrier)]) if relativePhase_two is None else np.append(
            relativePhase_two, [radReverse(subcarrier)], axis=0)

    for subcarrier in relativePhase_two:
        phase_temp = subcarrier[0]
        for i in range(len(subcarrier)):
        
            subcarrier[i]=round(subcarrier[i]-phase_temp,2)
            if subcarrier[i] >np.pi/2.0:
                subcarrier[i]=round(subcarrier[i]-np.pi,2)
            elif subcarrier[i] <-np.pi/2.0:
                subcarrier[i] = round(subcarrier[i] + np.pi,2)

    return amp, relativePhase_one


def readFile(filepath):
    file = read_bf_file.read_file(filepath)
    print "Length of packets: ", len(file)

    timestamp = np.array([])
    startTime = file[0].timestamp_low
    print "Start timestamp:" + str(startTime)
    antennaPair_raw, antennaPair_One, antennaPair_Two, antennaPair_Three = [], [], [], []
    for item in file:
        timestamp = np.append(timestamp, (item.timestamp_low - startTime) / 1000000.0)
        for eachcsi in range(0, 30):
            ''''
            acquire csi complex value for each antenna pair with shape ( len(file) * 30), i.e., packet number * subcarrier number
            '''
            antennaPair_One.append(item.csi[eachcsi][0][0])
            antennaPair_Two.append(item.csi[eachcsi][0][1])
            antennaPair_Three.append(item.csi[eachcsi][0][2])

    antennaPair_One = np.reshape(antennaPair_One, (len(file), 30)).transpose()
    antennaPair_Two = np.reshape(antennaPair_Two, (len(file), 30)).transpose()
    antennaPair_Three = np.reshape(antennaPair_Three, (len(file), 30)).transpose()
    raw1, blocked1 = complex_linearInterpolation(antennaPair_One, timestamp)
    raw2, blocked2 = complex_linearInterpolation(antennaPair_Two, timestamp)
    raw3, blocked3 = complex_linearInterpolation(antennaPair_Three, timestamp)
    """
    To get the relative phase between each antenna pair.
    Linear inteplotation operation.
    """
    pair_max, pair_secend = varianceOperation(raw1, raw2, raw3)


    amplitude, relativePhase1= relativePhaseOperation(pair_max, pair_secend)


    dwt_amp,dwt_relativePhase1=None,None
 
    for subcarrier in range(len(amplitude)):
        dwt_amp = np.array([dwtfilter(amplitude[subcarrier]).filterOperation()]) if dwt_amp is None else np.append(
            dwt_amp,[dwtfilter(amplitude[subcarrier]).filterOperation()] , axis=0)
    print dwt_amp.shape


    csi_matrix = np.array([dwt_amp])
    csi_matrix = np.append(csi_matrix, [relativePhase1], axis=0)

    return csi_matrix


if __name__ == '__main__':
    csi = readFile("/home/...")
    pylab.figure()
    pylab.subplot(3, 1, 1)
    pylab.plot(csi[0][0], 'g-', label='butterworth')
    pylab.legend(loc='best')
    pylab.ylim(0, 50)

    pylab.subplot(3, 1, 2)
    pylab.plot(csi[1][0], 'g-', label='butterworth')
    pylab.legend(loc='best')
    pylab.ylim(-2, 2)

    pylab.show()



