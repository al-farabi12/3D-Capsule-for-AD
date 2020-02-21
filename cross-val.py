import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from torch.optim import Adam,SGD
from torchvision import datasets, transforms
import datetime
root = os.getcwd()+'/'
print(f"Current folder:{root}")
currentFolder = "3D-Capsule-for-AD/"
data_path=root[:-len(currentFolder)]
print(f"Data folder:{data_path}")
data_15t_1mm = data_path+'15t/1mm-reg-nii-gz/'
data_15t_2mm = data_path+'15t/2mm-reg-nii-gz/'
data_3t_1mm = data_path+'3t/1mm-reg-nii-gz/'
data_3t_2mm = data_path+'3t/2mm-reg-nii-gz/'
data_type = [data_15t_1mm, data_15t_2mm, data_3t_1mm, data_3t_2mm]

all_diagnos ={'AD': 0,'MCI': 1,'CN': 2}
# diagnos = {'AD': 0,'MCI': 1,'CN': 2}
# val_dia = { 0:'AD', 1:'MCI',2:'CN'}
diagnos = {'AD': 0,'CN': 1}
# val_dia = { 0:'AD', 1:'MCI'}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
use_cross = True
cross_validation_folds = 4

trials = np.zeros((len(diagnos),cross_validation_folds))
for i in range(len(diagnos)):
    x = list(range(cross_validation_folds))
#     print(x)
    random.shuffle(x)
#     print(x)
    trials[i]=x
    
    

SUBJECT_INDEPENDENT = True
print("SUBJECT_INDEPENDENT:",SUBJECT_INDEPENDENT)
choice = 1 # 0==( 15t 1mm ) 1==(15t 2mm ) 2==( 3t 1mm ) 3==( 3t 2mm ) 
# USE_CUDA = True
# random_seed= 42
# random.seed( random_seed )
batch_size = 32#SHOULD BE devisable by 3
use_whole_data  = False
print("use_whole_data",use_whole_data)
split_portion = 0.2

use_all_copies_of_same_subject = False


print('use ALL copies of same subject',use_all_copies_of_same_subject)
s = 1
test_img_per_class_si = s
train_img_per_class_si = s
if not use_all_copies_of_same_subject:
    print(f'test/train imgs per subject {test_img_per_class_si}/{train_img_per_class_si}')

limit_testing_number = not use_whole_data
test_sub_per_class = 19
test_img_per_class_sd = 19
limit_training_number = not use_whole_data
train_sub_per_class = 70
train_img_per_class_sd = 70



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# n_classes_by_val= len(set(diagnos.values()))
# n_classes_by_key= len(set(diagnos.keys()))
list_of_all_imgs = os.listdir(data_type[choice])
total_imgs = {cls:[pat for pat in list_of_all_imgs if cls in pat] for cls in all_diagnos.keys()}
n_imgs_total = {k:len(v) for k,v in total_imgs.items()}
    
by_diag = {}
for i in all_diagnos.keys():
    by_diag[i]={}
for file in list_of_all_imgs:
    name, image, diagnose =file.split("__")
    if diagnose in diagnos:
        if name not in by_diag[diagnose]:
#             by_diag[diagnose][name]={}
            by_diag[diagnose][name]=[]
        by_diag[diagnose][name].append(file)
# print(by_diag)      
if SUBJECT_INDEPENDENT:
    list_of_all_subs = set([i[:10] for i in list_of_all_imgs])
    total_subs = {k:list(set([img[:10] for img in v])) for k,v in total_imgs.items()}
    n_subj_total ={k:len(v) for k,v in total_subs.items()}
    
    if limit_testing_number:
        if use_cross:
            test_subs = {cross:{k:random.sample(v,test_sub_per_class) for k,v in total_subs.items() if k in diagnos.keys()} for cross in range(cross_validation_folds)}
            n_subs_test = {cross:{k:len(v) for k,v in test_subs[cross].items()}for cross in range(cross_validation_folds)}

        else:
            test_subs = {k:random.sample(v,test_sub_per_class) for k,v in total_subs.items() if k in diagnos.keys()}
            n_subs_test = {k:len(v) for k,v in test_subs.items()}

    else:
        validation_split = split_portion
        if use_cross:
            test_subs = {cross:{k:random.sample(v,int(validation_split*len(v))) for k,v in total_subs.items()  if k in diagnos.keys()} for cross in range(cross_validation_folds)}
            n_subs_test = {cross:{k:len(v) for k,v in test_subs[cross].items()}for cross in range(cross_validation_folds)}

        else:
            test_subs = {k:random.sample(v,int(validation_split*len(v))) for k,v in total_subs.items()  if k in diagnos.keys()}
            n_subs_test = {k:len(v) for k,v in test_subs.items()}

    if use_all_copies_of_same_subject:
        if use_cross:
            test_imgs = {}
            n_imgs_test= {}
            for cross in range(cross_validation_folds):
                test_imgs[cross] = {k:[img for img in v if img[:10] in test_subs[cross][k]] 
                            for k,v in total_imgs.items() if k in diagnos.keys()}
                n_imgs_test[cross] = {k:len(v) for k,v in test_imgs[cross].items()}

        else:
            test_imgs = {k:[img for img in v if img[:10] in test_subs[k]] for k,v in total_imgs.items() if k in diagnos.keys()}
            n_imgs_test = {k:len(v) for k,v in test_imgs.items()}

                     
    else:
        if use_cross:
            test_imgs = {}
            n_imgs_test= {}
            for cross in range(cross_validation_folds):
                test_imgs[cross] = {}
                new_test_imgs ={}
                for dia,group in test_subs[cross].items():
                    new_group = []
                    for sub in group:
                        sample = random.sample(by_diag[dia][sub],min(test_img_per_class_si,len(by_diag[dia][sub])))
                        new_group+=sample
                    new_test_imgs[dia] = new_group
                test_imgs[cross] = new_test_imgs
                n_imgs_test= {k:len(v) for k,v in test_imgs[cross].items()}
        else:
            

    train_subs = {}
    for cross in range(cross_validation_folds):
        train_subs[cross] = {}
        for k in diagnos.keys():
            train_subs[cross][k]=[]
    #     for k,v in total_subs.items():
            for sub in total_subs[k]:
                if sub not in test_subs[cross][k]:
                    train_subs[cross][k].append(sub)
                if len(train_subs[cross][k])==train_sub_per_class:
                    if limit_training_number:
                        break

    n_subs_train = {cross:{k:len(v) for k,v in train_subs[cross].items()} for cross in range(cross_validation_folds)}
    train_imgs = {}
    if use_all_copies_of_same_subject:
        train_imgs = {cross:{k:[img for img in v if img[:10] in train_subs[cross][k]] for k,v in total_imgs.items() if k in diagnos.keys()} for cross in range(cross_validation_folds)}
    else:
        new_train_imgs = {}
        for cross in range(cross_validation_folds):
            train_imgs[cross] = {}
            for dia,group in train_subs[cross].items():
                new_group = []
                for sub in group:
                    sample = random.sample(by_diag[dia][sub],min(train_img_per_class_si,len(by_diag[dia][sub])))
                    new_group+=sample
                new_train_imgs[dia] = new_group
            train_imgs[cross] = new_train_imgs
#     print(train_imgs)
    n_imgs_train={}
    n_imgs_used={}
    n_subs_used={}
    for cross in range(cross_validation_folds):
        n_imgs_train[cross] = {k:len(v) for k,v in train_imgs[cross].items()} 
        n_imgs_used[cross] = {k:len(train_imgs[cross][k])+ len(test_imgs[cross][k]) for k in  diagnos.keys() } 
        n_subs_used[cross] = {k:len(train_subs[cross][k])+ len(test_subs[cross][k]) for k in  diagnos.keys() } 
    if use_cross:
                
        print("\n")
    #     print(f"Total Number of SUBJECTS of dataset {len(list_of_all_subs)}")
    #     print(f"Total Number of IMAGES of dataset {len(list_of_all_imgs)}")
        print("Number of SUBJECTS in each group of dataset\t", n_subj_total,"in TOTAL ",len(list_of_all_subs))
        print("Number of IMAGES in each group of dataset\t", n_imgs_total,"in TOTAL ",len(list_of_all_imgs))
        print("\n")
    #     print(f"Total USED Number of SUBJECTS {sum(n_subs_used.values())}")
    #     print(f"Total USED Number of IMAGES {sum(n_imgs_used.values())}")
        print("Number of USED SUBJECTS in each group\t\t", n_subs_used,"in TOTAL in cross-folds",[sum(n_subs_used[cross].values()) for cross in range(cross_validation_folds)])
        print("Number of USED IMAGES in each group \t\t", n_imgs_used,"in TOTAL ",               [sum(n_imgs_used[cross].values()) for cross in range(cross_validation_folds)])
        print("\n")
        print("Number of SUBJECTS in each group of TEST split\t",n_subs_test,"in TOTAL ",[sum(n_subs_test[cross].values())for cross in range(cross_validation_folds)])
        print("Number of SUBJECTS in each group of TRAIN split\t", n_subs_train,"in TOTAL ",[sum(n_subs_train[cross].values())for cross in range(cross_validation_folds)])
        print("\n")

        print("Number of IMAGES in each group of TEST split\t",n_imgs_test,"in TOTAL ",[sum(n_imgs_test[cross].values())for cross in range(cross_validation_folds)])
        print("Number of IMAGES in each group of TRAIN split\t", n_imgs_train,"in TOTAL ",[sum(n_imgs_train[cross].values())for cross in range(cross_validation_folds)])

    else:
        
        print("\n")
    #     print(f"Total Number of SUBJECTS of dataset {len(list_of_all_subs)}")
    #     print(f"Total Number of IMAGES of dataset {len(list_of_all_imgs)}")
        print("Number of SUBJECTS in each group of dataset\t", n_subj_total,"in TOTAL ",len(list_of_all_subs))
        print("Number of IMAGES in each group of dataset\t", n_imgs_total,"in TOTAL ",len(list_of_all_imgs))
        print("\n")
    #     print(f"Total USED Number of SUBJECTS {sum(n_subs_used.values())}")
    #     print(f"Total USED Number of IMAGES {sum(n_imgs_used.values())}")
        print("Number of USED SUBJECTS in each group\t\t", n_subs_used,"in TOTAL ",sum(n_subs_used.values()))
        print("Number of USED IMAGES in each group \t\t", n_imgs_used,"in TOTAL ",sum(n_imgs_used.values()))
        print("\n")
        print("Number of SUBJECTS in each group of TEST split\t",n_subs_test,"in TOTAL ",sum(n_subs_test.values()))
        print("Number of SUBJECTS in each group of TRAIN split\t", n_subs_train,"in TOTAL ",sum(n_subs_train.values()))
        print("\n")

        print("Number of IMAGES in each group of TEST split\t",n_imgs_test,"in TOTAL ",sum(n_imgs_test.values()))
        print("Number of IMAGES in each group of TRAIN split\t", n_imgs_train,"in TOTAL ",sum(n_imgs_train.values()))
else:
#     test_imgs = {k:random.sample(v,test_sub_per_class) for k,v in total_imgs.items() if k in diagnos.keys()}
    if limit_testing_number:
        test_imgs = {k:random.sample(v,test_img_per_class_sd) for k,v in total_imgs.items() if k in diagnos.keys()}
    else:
        validation_split = split_portion
        test_imgs = {k:random.sample(v,int(validation_split*len(v))) for k,v in total_imgs.items() if k in diagnos.keys()}
    n_imgs_test = {k:len(v) for k,v in test_imgs.items()}
    rest_imgs = {k:[img for img in v if img not in test_imgs[k]] for k,v in total_imgs.items() if k in diagnos.keys()}
    if limit_training_number:
        train_imgs = {k:random.sample(v,train_img_per_class_sd) for k,v in rest_imgs.items() }
    else:
        train_imgs = rest_imgs

    n_imgs_train = {k:len(v) for k,v in train_imgs.items()}
    n_imgs_used = {k:len(train_imgs[k])+ len(test_imgs[k]) for k in  diagnos.keys() }

    print("\n")
#     print(f"Total Number of SUBJECTS of dataset {len(list_of_all_subs)}")
#     print(f"Total Number of IMAGES of dataset {len(list_of_all_imgs)}")
#     print("Number of SUBJECTS in each group of dataset\t\t\t", n_subj_total)
    print("Number of IMAGES in each group of dataset\t", n_imgs_total,"in TOTAL \t",len(list_of_all_imgs))
#     print("\n")
#     print(f"Total USED Number of SUBJECTS {sum(n_subs_used.values())}")
#     print(f"Total USED Number of IMAGES {sum(n_imgs_used.values())}")
#     print("Number of USED SUBJECTS in each group\t\t\t", n_subs_used)
    print("Number of USED IMAGES in each group\t\t", n_imgs_used, "in TOTAL \t",sum(n_imgs_used.values()))
#     print("\n")
#     print("Number of SUBJECTS in each group of TEST split\t",n_subs_test,"in TOTAL ",sum(n_subs_test.values()))
#     print("Number of SUBJECTS in each group of TRAIN split\t", n_subs_train,"in TOTAL ",sum(n_subs_train.values()))
    print("\n")

    print("Number of IMAGES in each group of TEST split\t",n_imgs_test,"in TOTAL",sum(n_imgs_test.values()))
    print("Number of IMAGES in each group of TRAIN split\t", n_imgs_train,"in TOTAL",sum(n_imgs_train.values()))

    
all_img = []
for i in test_imgs.values():
    all_img = all_img+i
for i in train_imgs.values():
    all_img = all_img+i
class MyDataFinal(Dataset):
    def __init__(self, path,all_imgs):
        self.folder=path
        self.img = all_imgs
        self.len = len(self.img)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
#         temp_size=100
#         vol = torch.zeros(temp_size,temp_size,temp_size,dtype=torch.float32)
        add_noise = False #random.sample([True,False],1)[0]
        di = self.img[index].split('__')[-1]
        name =  os.path.join(self.folder,self.img[index])+'/mri/registered.nii.gz'
        nii = nb.load(name)
#         volume = torch.from_numpy(nii.get_fdata()).type(torch.FloatTensor)#[7:82,10:100,10:75]
        #slices 
#         volume = torch.from_numpy(nii.get_fdata()).type(torch.FloatTensor)[45,:,:]
#         print(volume.shape)
#         volume = torch.from_numpy(nii.get_fdata()).type(torch.FloatTensor)#[:,54,:]
#         volume = torch.from_numpy(nii.get_fdata()).type(torch.FloatTensor)[:-3,:-1,45]
        volume = torch.from_numpy(nii.get_fdata()).type(torch.FloatTensor)[1:-2,:-1,1:-2]
#         print(volume.shape)
#         x,y,z = volume.shape
#         if add_noise:
#             noise = 10*torch.randn(volume.shape)
#             volume = volume + noise
#         x,y,z= random.randint(0,temp_size-x),random.randint(0,temp_size-y),random.randint(0,temp_size-z)

        volume = volume/volume.max()
#         vol[x:x+75,y:y+90,z:z+65]=volume
        labels = diagnos[di]
#         return vol.unsqueeze(0), labels
        volume = volume.unsqueeze(0)
        
#         print(volume.shape)
#         volume = volume.permute(0,3,1,2)
#         print(volume.shape)
        
#         volume = F.adaptive_avg_pool2d(volume,28)#.squeeze()
#         print(volume.shape)
        
        return volume, labels
# 1: 21-79
# 2: 30-80
# 3: 15-55
dataset=MyDataFinal(data_type[choice],all_img)
test_tot=sum(n_imgs_test.values())
train_tot=sum(n_imgs_train.values())
if SUBJECT_INDEPENDENT:
    test_indices=[i for i in range(test_tot)]
    train_indices=[i for i in range(test_tot,test_tot+train_tot)]
else:
    test_indices=[i for i in range(test_tot)]
    train_indices=[i for i in range(test_tot,test_tot+train_tot)]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset,sampler=train_sampler, num_workers=10, drop_last=False,batch_size=batch_size)
test_loader = DataLoader(dataset,sampler=test_sampler, num_workers=10, drop_last=False,batch_size=batch_size)