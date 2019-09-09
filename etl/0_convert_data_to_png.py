import pydicom
import cv2
import os
import re

import pandas as pd
import numpy as np

from tqdm import tqdm

def extract_meta(dicom):
    return {'view': dicom.ViewPosition,
            'sex': dicom.PatientSex,
            'age': dicom.PatientAge,
            'monochrome': dicom.PhotometricInterpretation,
            'sop': dicom.SOPInstanceUID,
            'series': dicom.SeriesInstanceUID,
            'study': dicom.StudyInstanceUID}

def listify(dct): 
    for key in dct.keys():
        dct[key] = [dct[key]]
    return dct

def convert_and_extract(dicoms, image_save_dir, df_savefile):
    list_of_dicom_df = []
    for dcmfile in tqdm(dicoms, total=len(dicoms)):
        tmp_dcm = pydicom.read_file(dcmfile, force=True) 
        tmp_meta = extract_meta(tmp_dcm)
        tmp_meta['filename'] = dcmfile
        tmp_meta['height'] = tmp_dcm.pixel_array.shape[0]
        tmp_meta['width']  = tmp_dcm.pixel_array.shape[1]
        tmp_meta_df = pd.DataFrame(listify(tmp_meta))
        list_of_dicom_df.append(tmp_meta_df)
        tmp_array = tmp_dcm.pixel_array
        assert tmp_array.dtype == 'uint8'
        if tmp_meta['monochrome'] == 'MONOCHROME1':
            print('Inverting image ...')
            tmp_array = np.invert(tmp_array)
        status = cv2.imwrite(os.path.join(image_save_dir, tmp_meta['sop'][0] + '.png'), tmp_array)
    #
    dicom_df = pd.concat(list_of_dicom_df)
    dicom_df.to_csv(df_savefile, index=False)

# Convert train
TRAIN_DICOM_DIR = '../data/dicom-images-train/'
TRAIN_IMAGE_DIR = '../data/pngs/train/'
TRAIN_DF_SAVEFILE = '../data/train_meta.csv'

if not os.path.exists(TRAIN_IMAGE_DIR): os.makedirs(TRAIN_IMAGE_DIR)

train_dicoms = []
for root, dirs, files in os.walk(TRAIN_DICOM_DIR):
    for fi in files:
        if re.search('dcm', fi):
            train_dicoms.append(os.path.join(root, fi))

convert_and_extract(train_dicoms, TRAIN_IMAGE_DIR, TRAIN_DF_SAVEFILE)

# Convert test
TEST_DICOM_DIR = '../data/dicom-images-test/'
TEST_IMAGE_DIR = '../data/pngs/test/'
TEST_DF_SAVEFILE = '../data/test_meta.csv'

if not os.path.exists(TEST_IMAGE_DIR): os.makedirs(TEST_IMAGE_DIR)

test_dicoms = []
for root, dirs, files in os.walk(TEST_DICOM_DIR):
    for fi in files:
        if re.search('dcm', fi):
            test_dicoms.append(os.path.join(root, fi))

convert_and_extract(test_dicoms, TEST_IMAGE_DIR, TEST_DF_SAVEFILE)

# Convert stage 2 test
TEST_DICOM_DIR = '../data/dicom-images-stage2/'
TEST_IMAGE_DIR = '../data/pngs/stage2/'
TEST_DF_SAVEFILE = '../data/stage2_meta.csv'

if not os.path.exists(TEST_IMAGE_DIR): os.makedirs(TEST_IMAGE_DIR)

test_dicoms = []
for root, dirs, files in os.walk(TEST_DICOM_DIR):
    for fi in files:
        if re.search('dcm', fi):
            test_dicoms.append(os.path.join(root, fi))

convert_and_extract(test_dicoms, TEST_IMAGE_DIR, TEST_DF_SAVEFILE)

