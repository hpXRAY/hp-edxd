from hpm.models.mcaModel import MCA
from os import listdir
from os.path import isfile, join
import numpy as np

from PyQt6 import QtWidgets
##  WARNING do not run this unless you are sure

files = [
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min001.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min002.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min003.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min004.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min005.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min006.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min007.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min008.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min009.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min010.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min011.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min012.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min013.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min014.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min015.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min016.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min017.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min018.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min019.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/2023_Ce_W_Cd109_Co57_30min020.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_001.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_002.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_003.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_004.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_005.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_006.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_007.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_008.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_009.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_010.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_011.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_012.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_013.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_014.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_015.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_016.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_017.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_018.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_019.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_020.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_001.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_002.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_003.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_004.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_005.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_006.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_007.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_008.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_009.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_010.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_011.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_012.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_013.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_014.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_015.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_016.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_017.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_018.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_019.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_020.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_021.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_022.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_023.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_024.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_025.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_026.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_027.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_028.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_029.dat.mca',
    '/Volumes/data/16bmb/PEC/Data/2023-1/HPCAT/Ross/20230414_detectors/GSD/Cd_Co_Ce_W/20230418_Ce_W_Cd109_Co57_30min_2_030.dat.mca'
]

file_out = '/Users/hrubiak/Desktop/2023_Ce_W_Cd109_Co57_30min001-070.dat.hpmca'

def sum_files(files, file_out):



    mca = MCA()

    
    for i, file in enumerate(files):
        
        [file, ok] = mca.read_file(file)
        data = mca.get_data()
        if i == 0:
            summed_data = np.zeros(data.shape)
        summed_data = summed_data + data
        mca.data = summed_data

    mca.write_file(file_out)
   
app = QtWidgets.QApplication([])

sum_files(files, file_out)