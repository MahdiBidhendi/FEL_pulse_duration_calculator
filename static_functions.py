import h5py
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Iterable, Optional, Dict, Union
from pandas import Series, DataFrame
import pandas as pd
from scipy.optimize import curve_fit
import glob
from scipy.signal import savgol_filter
from scipy import integrate
import matplotlib.gridspec as gridspec
from skimage.measure import block_reduce
from os import path as path_func
import os
from matplotlib.ticker import MaxNLocator
import statistics
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy import stats

def dset_data_frame(group: h5py.Group, alias: Optional[str] = None) -> DataFrame:
    """Returns a pandas DataFrame for the given HDF group"""
    if not alias:
        alias = "/".join(group.name.split("/")[-3:])
    train_id = Series(group["index"], name="Train ID")
    return Series((group["value"][i] for i in train_id.index), name=alias, index=train_id).to_frame()

def time2energy_conversion(parameters_calibration, sample_rate_value, sample_no_min, sample_no_max):
    """time axis in nanoseconds"""
    tof_axis = np.linspace(sample_no_min, sample_no_max, sample_no_max-sample_no_min)*sample_rate_value
    l_eff_2 = parameters_calibration[0]
    time_zero = parameters_calibration[1]
    constant = parameters_calibration[2]

    def time2energy_function(time_array, *q):
        l_eff_squared, t_0, constant = q
        return (l_eff_squared / (time_array + t_0) ** 2) #+ constant

    energy_calibration = time2energy_function(tof_axis, *[l_eff_2, time_zero, constant])

    return energy_calibration

def remove_bckg(data_matrix):
    background = 30
    data_matrix_no_background = np.zeros((len(data_matrix[:, 0]), len(data_matrix[0, :])))
    for i in range(len(data_matrix[0, :])):
        background_value = np.nanmean(np.absolute(data_matrix[0:background, i]), axis=0)
        data_matrix_no_background[:, i] = data_matrix[:, i] + background_value

    return data_matrix_no_background

def get_pulses_peak_energy_value(energy_array, spectogram_matrix):
    idx_max_value_energy = np.zeros(len(spectogram_matrix[0, :]))
    max_values_energy = np.zeros(len(spectogram_matrix[0, :]))
    for i in range(len(spectogram_matrix[0, :])):
        idx_max_value_energy[i] = spectogram_matrix[:, i].tolist().index(np.nanmax(spectogram_matrix[:, i], axis=0))
        max_values_energy[i] = energy_array[int(idx_max_value_energy[i])]
    return max_values_energy, idx_max_value_energy


def shift_index_2peaks(array_matrix, idx_peak_no, shift_values_array):
    shifted_streaked_matrix = np.zeros((len(array_matrix[:, 0]), len(array_matrix[0, :])))
    for i in range(len(idx_peak_no)):
        shifted_streaked_matrix[:, i] = np.roll(array_matrix[:, i], int(shift_values_array[i]))

    return shifted_streaked_matrix, np.nanmean(shifted_streaked_matrix, axis=1)

def data2matrixarray(data_frame_pds, sample_no_min, sample_no_max):
    new_matrix = np.zeros((len(data_frame_pds), sample_no_max-sample_no_min))
    for i in range(len(data_frame_pds)):
        new_matrix[i, :] = data_frame_pds[i, 0][sample_no_min:sample_no_max]
    return new_matrix

def normalize_matrix(matrix):
    normalized_matrix = np.zeros((len(matrix[:, 0]), len(matrix[0, :])))
    for i in range(len(matrix[0, :])):
        normalized_matrix[:, i] = matrix[:, i]/np.nanmax(matrix[:, i])

    return normalized_matrix

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y, amp, mean, fwhm):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[0, amp, mean, fwhm]) #[min,max,mean,sigma]  
    return popt,pcov  

def get_path(run_number,path,file_num):
    filename = glob.glob(path+'FLASH2_USER2_stream_2_'+run_number+f'_file{file_num}_*.h5')[0]
    file_path = Path(filename)
    return file_path

def stat_str(adc_data,laser_delay_data,retardation_voltage_data,sample_frequency_data,tofnumber,sample_str_min,sample_ref_min,width,energy_calibration_parameters): 
    
    sample_min = 0     # min sample number of the adc
    sample_max = 20000   # max sample number of the adc
    adc_matrix_data = adc_data.to_numpy()
    adc_spectrum = -np.transpose(data2matrixarray(adc_matrix_data, sample_min, sample_max)) # [shot#, sample#]
    adc_no_bck = remove_bckg(adc_spectrum)
    laser_delay = laser_delay_data.to_numpy()[:, 0]
    retardation_voltage = float(retardation_voltage_data.mean())
    sample_frequency = float(sample_frequency_data.mean())
    sample_rate = 1E3/sample_frequency   # Frequency in GIGAHERTZ ----> nanoseconds
       
    energy_axis = time2energy_conversion(energy_calibration_parameters, sample_rate, sample_min, sample_max) + retardation_voltage

    reference_signal_matrix = adc_no_bck[sample_ref_min:sample_ref_min+width, :]
    reference_signal_mean = np.nanmean(reference_signal_matrix, axis=1)
    streaked_signal_matrix = adc_no_bck[sample_str_min:sample_str_min+width, :]
    streaked_signal_mean = np.nanmean(streaked_signal_matrix, axis=1)
    extra = tofnumber
    
    return reference_signal_matrix, streaked_signal_matrix, energy_axis, adc_no_bck, sample_rate

def trace_plotter(axis,energy_ax,run_num,data_type):
    
    X,Y = np.meshgrid(np.linspace(0, len(axis[0, :]),  len(axis[0, :])), energy_ax)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='w')
    plt.xlabel('Shot Number')
    plt.ylabel(' $ Energy [eV] $')
    plt.title(' %s %s spectrum' %(run_num,data_type))
    #plt.xlim(50,150)
    CS = plt.contourf(X, Y, axis, 100, cmap=plt.cm.jet) #vmax=700, vmin=-1800)
    plt.colorbar()
    plt.show()
    
def temporal_sorter(sample_str_min , sample_str_max , width , no_bck , no_bck2 , threshold , energy_axis , str_spd1 , str_spd2 , thickness = 30):

    cg1,cg2=[],[]
    difference=[]
    temporal_inliers=[]    
    #thickness = 30 -- > width of the data we take to calculate the center of gravity, the more the thickness, the higher the precision
    
    #len(no_bck[1])
    for data_number in range(0,len(no_bck[0, :])):

        tof1_spec = savgol_filter(no_bck[sample_str_min:sample_str_max , data_number], 15,3)
        tof2_spec = savgol_filter(no_bck2[sample_str_min:sample_str_max , data_number], 15,3)

        index_max_tof1 = np.argmax(tof1_spec)
        index_max_tof2 = np.argmax(tof2_spec)

        cg_tof1 = np.sum(energy_axis[index_max_tof1-thickness:index_max_tof1+thickness]*tof1_spec[index_max_tof1-thickness:index_max_tof1+thickness])/np.sum(tof1_spec[index_max_tof1-thickness:index_max_tof1+thickness])
        cg_tof2 = np.sum(energy_axis[index_max_tof2-thickness:index_max_tof2+thickness]*tof2_spec[index_max_tof2-thickness:index_max_tof2+thickness])/np.sum(tof2_spec[index_max_tof2-thickness:index_max_tof2+thickness])    
        cg1.append(cg_tof1/str_spd1)
        cg2.append(cg_tof2/str_spd2)
        
    cg1_fluc = np.subtract(cg1,np.nanmean(cg1))
    cg2_fluc = np.subtract(cg2,np.nanmean(cg2))
    reversed_cg2 = np.subtract(0,cg2_fluc)
    difference = reversed_cg2-cg1_fluc
    
    for i in range(0,len(difference)):
        if np.abs(difference[i]) < threshold:
            temporal_inliers.append(1) #list of the index of inliers, 
        else:
            temporal_inliers.append(0)
        
    print("total number:              ",len(no_bck[0, :]))
    print("number of temporal inliers:",np.sum(temporal_inliers))  
    
    return cg1,cg2,difference,temporal_inliers,cg1_fluc,cg2_fluc,reversed_cg2,difference

#this function is used for the initial values of the gaussian fit:
def mean_calculator(energy_axis,reference,streaked,TOF_num):
    
    averaged_spectrum_ref = block_reduce(reference, block_size=(1,len(reference[0])), func=np.mean, cval=np.mean(reference))
    averaged_spectrum_str = block_reduce(streaked, block_size=(1,len(streaked[0])), func=np.mean, cval=np.mean(streaked))
    mean_ref = energy_axis[np.argmax(averaged_spectrum_ref)]
    mean_str = energy_axis[np.argmax(averaged_spectrum_str)]
    amp_ref = np.max(averaged_spectrum_ref)
    amp_str = np.max(averaged_spectrum_str)
    
    print("ref mean - TOF%s:" %TOF_num , mean_ref)
    print("str mean - TOF%s:" %TOF_num ,mean_str)
    
    plt.plot(energy_axis,averaged_spectrum_ref ,label="ref TOF%s" %TOF_num)
    plt.plot(energy_axis,averaged_spectrum_str ,label="str TOF%s" %TOF_num)
    plt.xlabel('energy (eV)')
    plt.legend()
    #plt.show()
    
    return mean_ref, mean_str, amp_ref, amp_str

def fitting_sorter(energy_axis,reference,streaked,amp_ref,amp_str,mean_ref,mean_str,amp_threshold=50,base_threshold=50,sigma_ref_threshold=1): #threshold values are to sort out the fitting outliers 
    
    sigma_ref_list,amplitude_ref_list,base_ref_list,err_ref_list,fit_integ_ref_list,x0_ref_list = [],[],[],[],[],[]
    sigma_str_list,amplitude_str_list,base_str_list,err_str_list,fit_integ_str_list,x0_str_list = [],[],[],[],[],[]

    #sigma_ref_list_2,sigma_str_list_2,amplitude_ref2,amplitude_str2,base_ref2,base_str2,err_ref2,err_str2,fit_integ_ref2,fit_integ_str2 = [],[],[],[],[],[],[],[],[],[]

    fitting_inliers = []

    for i in range(0,len(reference[1,:])):
    #for i in inliers_index:

        try:
            [H_ref, A_ref, x0_ref, sigma_ref], pcov_ref = gauss_fit(energy_axis, reference[:,i], np.max(reference[:,i]), energy_axis[np.argmax(reference[:,i])], 2) #for reference spectrum TOF1
        except RuntimeError:
            [H_ref, A_ref, x0_ref, sigma_ref], pcov_ref = [0,0,0,0],[0,0]
        try:
            [H_str, A_str, x0_str, sigma_str], pcov_str = gauss_fit(energy_axis, streaked[:,i], np.max(streaked[:,i]), energy_axis[np.argmax(streaked[:,i])], 5) #for streaked spectrum TOF1
        except RuntimeError:
            [H_str, A_str, x0_str, sigma_str], pcov_str = [0,0,0,0],[0,0]
        

        # here I filter outliers (based on the baseline and amplitude of the gaussian peak)
        if  100*amp_threshold>A_ref>amp_threshold and 100*amp_threshold>A_str>amp_threshold and 0<H_str<base_threshold and 0<H_ref<base_threshold and sigma_str>sigma_ref and sigma_ref*2.35>sigma_ref_threshold:
            fitting_inliers.append(1)
        else:
            fitting_inliers.append(0)

        amplitude_ref_list.append(A_ref)
        base_ref_list.append(H_ref)
        sigma_ref_list.append(np.abs(sigma_ref*2.35))
        x0_ref_list.append(x0_ref)
        err_ref_list.append(np.sqrt(np.diag(pcov_ref)))
        fit_integ_ref_list.append(integrate.cumulative_trapezoid(gauss(energy_axis, H_ref, A_ref, x0_ref, sigma_ref))[-1]) #integrating the gaussian fit

        amplitude_str_list.append(A_str)
        base_str_list.append(H_str)
        sigma_str_list.append(np.abs(sigma_str*2.35))
        x0_str_list.append(x0_str)
        err_str_list.append(np.sqrt(np.diag(pcov_str)))
        fit_integ_str_list.append(integrate.cumulative_trapezoid(gauss(energy_axis, H_str, A_str, x0_str, sigma_str))[-1])
        
    return amplitude_ref_list,amplitude_str_list,base_ref_list,base_str_list,sigma_ref_list,sigma_str_list,x0_ref_list,x0_str_list,err_ref_list,err_str_list,fit_integ_ref_list,fit_integ_str_list,fitting_inliers

def error_plotter(error_value):

    error_matrix = np.asmatrix(error_value)

    plt.subplot(1, 4, 1)
    plt.plot(error_matrix[:,0],"o", alpha=0.6)
    plt.title('error of the fit offset')
    plt.ylabel('error value')
    plt.xlabel('shot number')

    plt.subplot(1, 4, 2)
    plt.plot(error_matrix[:,1],"o", alpha=0.6)
    plt.title('error of the fit amplitude')
    plt.xlabel('shot number')

    plt.subplot(1, 4, 3)
    plt.plot(error_matrix[:,2],"o", alpha=0.6)
    plt.title('error of the fit mean value')
    plt.xlabel('shot number')

    plt.subplot(1, 4, 4)
    plt.plot(error_matrix[:,3],"o", alpha=0.6)
    plt.title('error of the fit sigma')
    plt.xlabel('shot number')
    plt.show()
    
def fwhm_plotter(sigma_ref1,sigma_ref2,sigma_str1,sigma_str2):
    
    plt.subplot(1, 2, 1)
    plt.plot(sigma_ref1,'o',label="ref", alpha=0.6)
    plt.plot(sigma_str1,'o',label="str", alpha=0.6)
    plt.ylabel('FWHM (eV)')
    plt.xlabel('shot number')
    plt.title('TOF 1 FWHM reference spectrum')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sigma_ref2,'o',label="ref", alpha=0.6)
    plt.plot(sigma_str2,'o',label="str", alpha=0.6)
    plt.ylabel('FWHM (eV)')
    plt.xlabel('shot number')
    plt.title('TOF 2 FWHM reference spectrum')
    plt.legend()
    plt.show()
    
def gaussian_amp_base_plotter(base_ref1,base_ref2,base_str1,base_str2,amp_ref1,amp_ref2,amp_str1,amp_str2):

    plt.subplot(2, 4, 1)
    plt.plot(base_ref1,'o',label="ref", alpha=0.6)
    plt.plot(base_str1,'o',label="str", alpha=0.6)
    plt.ylabel('Amplitude')
    plt.title('TOF 1 Gaussian fitting offset')
    plt.xlabel('shot number')
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.plot(base_ref2,'o',label="ref", alpha=0.6)
    plt.plot(base_str2,'o',label="str", alpha=0.6)
    plt.title('TOF 2 Gaussian fitting offset')
    plt.xlabel('shot number')
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.plot(amp_ref1,'o',label="ref", alpha=0.6)
    plt.plot(amp_str1,'o',label="str", alpha=0.6)
    plt.title('TOF 1 Gaussian fitting amplitude')
    plt.xlabel('shot number')
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.plot(amp_ref2,'o',label="ref", alpha=0.6)
    plt.plot(amp_str2,'o',label="str", alpha=0.6)
    plt.title('TOF 2 Gaussian fitting amplitude')
    plt.xlabel('shot number')
    plt.legend()
    plt.show()
    
def fitting_plotter(signal,energy,amp,mean,index,plt_title):

    [H, A, x0, sigma], __ = gauss_fit(energy, signal[:,index], amp, mean, 2)
    plt.title(plt_title)
    plt.plot(energy,signal[:,index],label=('FWHM = %s eV' % round(sigma*2.35,3)))
    plt.plot(energy, gauss(energy, H, A, x0, sigma), '--r')
    plt.legend()
    
def duration_calculator(run_number,inliers_index,sigma_str1,sigma_str2,sigma_ref1,sigma_ref2,str_speed1,str_speed2,relative_arrival_time_rms):

    t_tof1,t_tof2=[],[]
    t_tof1_inlier,t_tof2_inlier=[],[]

    for j in range(0,len(sigma_str1)):
        t_tof1.append(np.abs(np.sqrt(sigma_str1[j]**2-sigma_ref1[j]**2)/str_speed1))
        t_tof2.append(np.abs(np.sqrt(sigma_str2[j]**2-sigma_ref2[j]**2)/str_speed2))

        if inliers_index[j]>0:
            t_tof1_inlier.append(t_tof1[j])
            t_tof2_inlier.append(t_tof2[j])

    print('Run number:',run_number)
    print('Number of all shots:           ',len(sigma_str1))
    print('Average pulse duration - TOF 1: %d \u00B1 %d'%(round(np.nanmean(t_tof1)),round(np.nanstd(t_tof1))))
    print('Average pulse duration - TOF 2: %d \u00B1 %d'%(round(np.nanmean(t_tof2)),round(np.nanstd(t_tof2))))
    print('Uncertainty - TOF 1:            %d %% '%(round(np.nanstd(t_tof1))/round(np.nanmean(t_tof1))*100))
    print('Uncertainty - TOF 2 :           %d %% '%(round(np.nanstd(t_tof2))/round(np.nanmean(t_tof2))*100))
    print('-------------------------------------')
    print('Number of inlier shots:        ',len(t_tof1_inlier))
    print('Average pulse duration - TOF 1: %d \u00B1 %d'%(np.nanmean(t_tof1_inlier),np.nanstd(t_tof1_inlier)))
    print('Average pulse duration - TOF 2: %d \u00B1 %d'%(np.nanmean(t_tof2_inlier),np.nanstd(t_tof2_inlier)))
    print('Uncertainty - TOF 1:            %d %% '%((np.nanstd(t_tof1_inlier))/(np.nanmean(t_tof1_inlier))*100))
    print('Uncertainty - TOF 2:            %d %% '%((np.nanstd(t_tof2_inlier))/(np.nanmean(t_tof2_inlier))*100))
    print('-------------------------------------')
    print('Arrival time resolution (rms):  %d fs' %relative_arrival_time_rms)

    f = open("Run_summary.txt","w+")
    f.write("Run number:%s \n"%run_number)
    f.write('Number of all shots:            %d\n'%len(sigma_str1))
    f.write('Average pulse duration - TOF 1: %d \u00B1 %d\n'%(round(np.nanmean(t_tof1)),round(np.nanstd(t_tof1))))
    f.write('Average pulse duration - TOF 2: %d \u00B1 %d\n'%(round(np.nanmean(t_tof2)),round(np.nanstd(t_tof2))))
    f.write('Uncertainty - TOF 1:            %d %% \n'%(round(np.nanstd(t_tof1))/round(np.nanmean(t_tof1))*100))
    f.write('Uncertainty - TOF 2 :           %d %% \n'%(round(np.nanstd(t_tof2))/round(np.nanmean(t_tof2))*100))
    f.write('-------------------------------------\n')
    f.write('Number of inlier shots:         %d\n' %len(t_tof1_inlier))
    f.write('Average pulse duration - TOF 1: %d \u00B1 %d\n'%(round(np.nanmean(t_tof1_inlier)),round(np.nanstd(t_tof1_inlier))))
    f.write('Average pulse duration - TOF 2: %d \u00B1 %d\n'%(round(np.nanmean(t_tof2_inlier)),round(np.nanstd(t_tof2_inlier))))
    f.write('Uncertainty - TOF 1:            %d %% \n'%(round(np.nanstd(t_tof1_inlier))/round(np.nanmean(t_tof1_inlier))*100))
    f.write('Uncertainty - TOF 2:            %d %% \n'%(round(np.nanstd(t_tof2_inlier))/round(np.nanmean(t_tof2_inlier))*100))
    f.write('-------------------------------------\n')
    f.write('Arrival time resolution (rms):  %d fs'%relative_arrival_time_rms)
    f.close()
    

    return t_tof1,t_tof2,t_tof1_inlier,t_tof2_inlier

def correlation_plotter(t_tof1_inlier,t_tof2_inlier,sigma_ref_list1,sigma_str_list1,sigma_ref_list2,sigma_str_list2,amplitude_ref1,amplitude_ref2,amplitude_str1,amplitude_str2,fit_integ_ref1,fit_integ_ref2,fit_integ_str1,fit_integ_str2,cg1,cg2,str_slope_tof1,str_slope_tof2,run_number):


    #colors = np.linspace(0,len(tof1_duration),len(tof1_duration))
    plt.subplot(3, 4, 1, aspect='auto')
    plt.hist(t_tof1_inlier,bins=40, edgecolor='k',label="TOF 1 - mean = %s fs" % round(np.nanmean(t_tof1_inlier),1))
    plt.hist(t_tof2_inlier,bins=40, edgecolor='k', alpha=0.8,label="TOF 2 - mean = %s fs" % round(np.nanmean(t_tof2_inlier),1))
    plt.xlabel('Pulse duration (fs)', fontweight='bold')
    plt.ylabel('Counts', fontweight='bold')
    plt.title(run_number, fontweight='bold')
    plt.legend()

    plt.subplot(3, 4, 2, aspect='auto')
    plt.hist(sigma_ref_list1, bins=40,edgecolor='k', label='reference - mean = %s eV' % round(np.nanmean(sigma_ref_list1),1))
    plt.hist(sigma_str_list1, bins=40,edgecolor='k', label='streaked - mean = %s eV' % round(np.nanmean(sigma_str_list1),1))
    plt.legend(loc='upper right')
    plt.ylabel('Counts', fontweight='bold')
    plt.xlabel('FWHM (eV)', fontweight='bold')
    plt.title('TOF 1', fontweight='bold')

    plt.subplot(3, 4, 3, aspect='auto')
    plt.hist(sigma_ref_list2, bins=40, edgecolor='k', label='reference - mean = %s eV' % round(np.nanmean(sigma_ref_list2),1))
    plt.hist(sigma_str_list2, bins=40, edgecolor='k', label='streaked - mean = %s eV' % round(np.nanmean(sigma_str_list2),1))
    plt.legend(loc='upper right')
    plt.ylabel('Counts', fontweight='bold')
    plt.xlabel('FWHM (eV)', fontweight='bold')
    plt.title('TOF 2', fontweight='bold')

    plt.subplot(3, 4, 4, aspect='auto')
    xy = np.vstack([t_tof1_inlier,t_tof2_inlier])
    colors = gaussian_kde(xy)(xy)
    plt.plot(t_tof1_inlier,'o',label='TOF 1', alpha=0.5)
    plt.plot(t_tof2_inlier,'o',label='TOF 2', alpha=0.5)
    plt.xlabel('shot number', fontweight='bold')
    plt.ylabel('pulse duration (fs)', fontweight='bold')
    plt.legend()

    plt.subplot(3, 4, 5, aspect='auto')
    r,p = stats.pearsonr(t_tof1_inlier,t_tof2_inlier)
    plt.scatter(t_tof1_inlier,t_tof2_inlier, s = None, c = colors, alpha=0.5, label=f"Pearson correlation: {r:.2f}")
    plt.title('Pulse duration correlation', fontweight='bold')
    plt.xlabel('pulse duration TOF 1 (fs)', fontweight='bold')
    plt.ylabel('pulse duration TOF 2 (fs)', fontweight='bold')
    plt.legend()

    plt.subplot(3, 4, 6, aspect='auto')
    xy = np.vstack([sigma_ref_list1,sigma_ref_list2])
    colors = gaussian_kde(xy)(xy)
    r,p = stats.pearsonr(sigma_ref_list1,sigma_ref_list2)
    plt.scatter(sigma_ref_list1,sigma_ref_list2, s = None, c = colors, alpha=0.5, label=f"Pearson correlation: {r:.2f}")
    plt.title('Reference - FWHM correlation', fontweight='bold')
    plt.xlabel('FWHM of reference - TOF 1 (fs)', fontweight='bold')
    plt.ylabel('FWHM of reference - TOF 2 (fs)', fontweight='bold')
    plt.legend()

    plt.subplot(3, 4, 7, aspect='auto')
    xy = np.vstack([sigma_str_list1,sigma_str_list2])
    colors = gaussian_kde(xy)(xy)
    r,p = stats.pearsonr(sigma_str_list1,sigma_str_list2)
    plt.scatter(sigma_str_list1,sigma_str_list2, s = None, c = colors, alpha=0.5, label=f"Pearson correlation: {r:.2f}")
    plt.title('Streaked - FWHM correlation', fontweight='bold')
    plt.xlabel('FWHM of streaked - TOF 1 (fs)', fontweight='bold')
    plt.ylabel('FWHM of streaked - TOF 2 (fs)', fontweight='bold')
    plt.legend()
    
    plt.subplot(3, 4, 8, aspect='auto')
    xy = np.vstack([amplitude_ref1,amplitude_ref2])
    colors = gaussian_kde(xy)(xy)
    r,p = stats.pearsonr(amplitude_ref1,amplitude_ref2)
    plt.scatter(amplitude_ref1,amplitude_ref2, s = None, c = colors, alpha=0.5, label=f"Pearson correlation: {r:.2f}")
    plt.title('Reference - amplitude correlation', fontweight='bold')
    plt.xlabel('reference amplitude TOF 1', fontweight='bold')
    plt.ylabel('reference amplitude TOF 2', fontweight='bold')
    plt.legend()

    plt.subplot(3, 4, 9, aspect='auto')
    xy = np.vstack([amplitude_str1,amplitude_str2])
    colors = gaussian_kde(xy)(xy)
    r,p = stats.pearsonr(amplitude_str1,amplitude_str2)
    plt.scatter(amplitude_str1,amplitude_str2, s = None, c = colors, alpha=0.5, label=f"Pearson correlation: {r:.2f}")
    plt.title('Streaked - amplitude correlation', fontweight='bold')
    plt.xlabel('streaked amplitude TOF 1', fontweight='bold')
    plt.ylabel('streaked amplitude TOF 2', fontweight='bold')
    plt.legend()

    plt.subplot(3,4,10)
    xy = np.vstack([fit_integ_ref1,fit_integ_ref2])
    colors = gaussian_kde(xy)(xy)
    r,p = stats.pearsonr(fit_integ_ref1,fit_integ_ref2)
    plt.scatter(fit_integ_ref1,fit_integ_ref2, s = None, c = colors, alpha=0.5, label=f"Pearson correlation: {r:.2f}")
    plt.title('Reference - integration correlation', fontweight='bold')
    plt.xlabel('reference pulse TOF 1', fontweight='bold')
    plt.ylabel('reference pulse TOF 2', fontweight='bold')
    plt.legend()

    plt.subplot(3,4,11)
    xy = np.vstack([fit_integ_str1,fit_integ_str2])
    colors = gaussian_kde(xy)(xy)
    r,p = stats.pearsonr(fit_integ_str1,fit_integ_str2)
    plt.scatter(fit_integ_str1,fit_integ_str2, s = None, c = colors, alpha=0.5, label=f"Pearson correlation: {r:.2f}")
    plt.title('Streaked - integration correlation', fontweight='bold')
    plt.xlabel('streaked pulse TOF 1', fontweight='bold')
    plt.ylabel('streaked pulse TOF 2', fontweight='bold')
    plt.legend()

    plt.subplot(3,4,12)
    sns.set(style='white', font_scale=1.2)
    slope, intercept, r, p, stderr = scipy.stats.linregress(cg1, cg2)
    sns.regplot(cg1, cg2, color="g", ci=68, truncate=False, label=f"regression slope: {slope:.2f}")
    plt.legend()
    plt.title('Arrival time correlation', fontweight='bold')
    plt.xlabel('arrival time TOF 1 (fs)', fontweight='bold')
    plt.ylabel('arrival time TOF 2 (fs)', fontweight='bold')
    #plt.show()
    plt.style.use('default')

def rel_arrival_time(min_rel_arrival,max_rel_arrival,energy_axis,energy_axis2,no_bck,no_bck2,str_slope_tof1,str_slope_tof2,reference_signal,reference_signal2,streaked_signal,streaked_signal2):
    
    rel_arrival = np.arange(min_rel_arrival,max_rel_arrival,10) #third input defines the step size
    x_axis_start = 3800
    x_axis_width = 400
    thickness = 20 #width of the data we take to calculate the center of gravity, the more the thickness, the higher the precision

    xaxis = np.linspace(x_axis_start , x_axis_start + x_axis_width , x_axis_width) #raw time axis from TOF

    cg1,cg2=[],[]
    difference=[]
    t_tof1_list,t_tof2_list=[],[]
    t_tof1_err,t_tof2_err=[],[]

    for i in rel_arrival:

        t_tof1,t_tof2=[],[]

        for data_number in range(0,len(no_bck[0])):

            tof1_spec = savgol_filter(no_bck[x_axis_start:x_axis_start + x_axis_width , data_number], 15,3)
            tof2_spec = savgol_filter(no_bck2[x_axis_start:x_axis_start + x_axis_width , data_number], 15,3)

            index_max_tof1 = np.argmax(tof1_spec)
            index_max_tof2 = np.argmax(tof2_spec)

            cg_tof1 = np.sum(xaxis[index_max_tof1-thickness:index_max_tof1+thickness]*tof1_spec[index_max_tof1-thickness:index_max_tof1+thickness])/np.sum(tof1_spec[index_max_tof1-thickness:index_max_tof1+thickness])
            cg_tof2 = np.sum(xaxis[index_max_tof2-thickness:index_max_tof2+thickness]*tof2_spec[index_max_tof2-thickness:index_max_tof2+thickness])/np.sum(tof2_spec[index_max_tof2-thickness:index_max_tof2+thickness])    
            difference.append(cg_tof2-cg_tof1)
            cg1.append(cg_tof1)
            cg2.append(cg_tof2)

            if np.abs(cg_tof2-cg_tof1) < i:

                [H_ref1, A_ref1, x0_ref1, sigma_ref1], __ = gauss_fit(energy_axis, reference_signal[:,data_number], np.max(reference_signal[:,data_number]), energy_axis[np.argmax(reference_signal[:,data_number])], 2)
                [H_str1, A_str1, x0_str1, sigma_str1], __ = gauss_fit(energy_axis, streaked_signal[:,data_number], np.max(streaked_signal[:,data_number]), energy_axis[np.argmax(streaked_signal[:,data_number])], 5)
                [H_ref2, A_ref2, x0_ref2, sigma_ref2], __ = gauss_fit(energy_axis2, reference_signal2[:,data_number], np.max(reference_signal2[:,data_number]), energy_axis2[np.argmax(reference_signal2[:,data_number])], 2)
                [H_str2, A_str2, x0_str2, sigma_str2], __ = gauss_fit(energy_axis2, streaked_signal2[:,data_number], np.max(streaked_signal2[:,data_number]), energy_axis2[np.argmax(streaked_signal2[:,data_number])], 5)
                FWHM_ref1 = sigma_ref1*2.35
                FWHM_str1 = sigma_str1*2.35
                FWHM_ref2 = sigma_ref2*2.35
                FWHM_str2 = sigma_str2*2.35

                t_tof1.append(np.abs(np.sqrt(FWHM_str1**2-FWHM_ref1**2)/str_slope_tof1))
                t_tof2.append(np.abs(np.sqrt(FWHM_str2**2-FWHM_ref2**2)/str_slope_tof2))

        t_tof1_list.append(np.nanmean(t_tof1))
        t_tof2_list.append(np.nanmean(t_tof2))
        t_tof1_err.append(np.nanstd(t_tof1))
        t_tof2_err.append(np.nanstd(t_tof2))
        
    return rel_arrival,t_tof1_list,t_tof2_list,t_tof1_err,t_tof2_err