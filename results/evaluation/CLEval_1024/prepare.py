import argparse
from pathlib import Path
import glob
import zipfile
import shutil
import os
import numpy as np
import cv2
import re

def gt_prepare(gt_path, process_gt_path):
    process_gt = Path(process_gt_path)
    if process_gt.exists() and process_gt.is_dir():
        shutil.rmtree(process_gt)
    if not os.path.exists(process_gt):
        os.makedirs(process_gt)
    files=glob.glob(os.path.join(gt_path,'*.txt')) 
    print('number of gt files')
    print(len(files))
    for file in files:
        base_name=os.path.basename(file)
        new_path=os.path.join(process_gt,base_name)
        with open(file,'r',encoding='utf-8-sig') as f:
            with open(new_path,'w',encoding='utf8') as fw:
                lines=f.read().splitlines()
                for line in lines:
                    li=line.split('\t')
                    word = li[-1]
#                     if word=='USUI':
#                         print(li)
#                         print(len(li))
    
                    word=word.replace('@@@','').replace('ROT','')
                    match_rot = re.match(r'[ROT[0-9]+]', word,re.IGNORECASE)
                    if match_rot:
                        word=word.replace(match_rot[0],'')
                    match_unk = re.match(r'[UNK[0-9]+]', word,re.IGNORECASE)
                    if match_unk:
                        word=word.replace(match_unk[0],'[UNK]')
                    bbox =[int(float(li[j])) for j in range(len(li)-1)]
                    box=np.array(bbox).flatten()
                    if len(box)>8:
                        box=np.reshape(box,(-1,2))
                        rect = cv2.minAreaRect(box)
                        box = cv2.boxPoints(rect)
                        box=np.array(bbox).flatten()
                    point=','.join([str(int(float(b))) for b in box])
                    fw.write(point)
                    fw.write(','+word)
                    fw.write('\n')

def result_prepare(result_path, process_result_path):
    respath = Path(process_result_path)
    if respath.exists() and respath.is_dir():
        shutil.rmtree(respath)
    if not os.path.exists(respath):
            os.makedirs(respath)
    print(result_path)
    files=glob.glob(os.path.join(result_path,'*.txt'))
    print('number of result files')
    print(len(files))
    
    for file in files:
        base_name=os.path.basename(file)
        base_name=base_name.replace('_stdcombined','').replace('_combined','')
        new_path=os.path.join(respath,base_name)
        with open(file,'r',encoding='utf8') as f:
            with open(new_path,'w',encoding='utf8') as fw:
                lines=f.read().splitlines()
                for line in lines:
                    li=line.split('\t')
                    word=li[-1]
#                     if word=='USUI':
#                         print(li)
#                         print(len(li))
                    bbox =[int(float(li[j])) for j in range(len(li)-1)]
                    box=np.array(bbox).flatten()
                    point=','.join([str(int(float(b))) for b in box])
                    fw.write(point)
                    fw.write(','+word)
                    fw.write('\n')

                    
def file_rename(gt_path,result_path):
    txt_list=glob.glob(os.path.join(result_path, '*.txt'))
    for n,i in enumerate(txt_list):
        base_name=os.path.basename(i)
        #print(base_name)
        result_name='res_'+'img'+'_'+str(n)+'.txt'
        gt_name='gt_'+'img'+'_'+str(n)+'.txt'
        ori_gt_name=base_name[:-4]+'.txt'
        ori_gt_name=ori_gt_name.replace('_stdcombined','').replace('_combined','')
        os.rename(os.path.join(result_path,base_name),os.path.join(result_path,result_name))
        os.rename(os.path.join(gt_path,ori_gt_name),os.path.join(gt_path,gt_name))
        
        
        
def zip_creation(gt_path,result_path):
    respath = Path('Evaluation_data/result.zip')
    if respath.exists() and respath.is_dir():
        shutil.rmtree(respath)
    
    gtzip=Path('Evaluation_data/gt.zip')
    if gtzip.exists() and gtzip.is_dir():
        shutil.rmtree(gtzip) 
    submitfile = 'Evaluation_data/result.zip'
    if not os.path.exists('Evaluation_data'):
            os.makedirs('Evaluation_data')
    filenames = [file for file in os.listdir(result_path) if file.endswith('.txt')]
    zip = zipfile.ZipFile(submitfile, "w", zipfile.ZIP_DEFLATED)
    for filename in filenames:
        filepath = os.path.join(result_path, filename)
        zip.write(filepath, filename)
    zip.close()
    
    gtfile = 'Evaluation_data/gt.zip'
    gt_filenames = [file for file in os.listdir(gt_path) if file.endswith('.txt')]
    zip = zipfile.ZipFile(gtfile, "w", zipfile.ZIP_DEFLATED)
    for filename in gt_filenames:
        filepath = os.path.join(gt_path, filename)
        zip.write(filepath, filename)
    zip.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path')
    parser.add_argument('--path', default='Ver_1')
    args = parser.parse_args()
    gt_path = '../../../data/TestData/txt/'
    result_path = '../../../outputs/' + args.path + '/'
    process_gt = '../../../results/evaluation/CLEval_1024/gt'
    process_result = '../../../results/evaluation/CLEval_1024/result'
    gt_prepare(gt_path,process_gt)
    result_prepare(result_path, process_result)
    file_rename(process_gt, process_result)
    zip_creation(process_gt, process_result)
