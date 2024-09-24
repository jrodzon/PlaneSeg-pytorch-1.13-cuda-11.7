import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from skimage.transform import resize

# path = '/home/agent/Masters/Repos/PlaneSeg-pytorch-1.13-cuda-11.7/PlaneRCNN/test/nyu/'
# path = '/home/agent/Masters/Repos/PlaneSeg-pytorch-1.13-cuda-11.7/PlaneRCNN/test/eval_2nd_5epoch_nyu_2/'
# path = '/home/agent/Masters/Repos/PlaneSeg-pytorch-1.13-cuda-11.7/PlaneRCNN/test/eval_res_concat_BACK_nyu_2/'
path = '/home/agent/Masters/Repos/PlaneSeg-pytorch-1.13-cuda-11.7/PlaneRCNN/test/PlaneRCNNwoPlaneSeg_visualization300/'
# savedir = "/home/agent/Masters/Repos/PlaneSeg-pytorch-1.13-cuda-11.7/PlaneRCNN/test/nyudepth/"
path_list = os.listdir(path)
for filename in path_list:
    ff = filename.split('.')
    suffix = ff[1]
    picname = ff[0]
    if(suffix!="npy"):
        continue
    # print(ff)
    f = open(os.path.join(path, filename), 'rb')
    arr = np.load(f)
    print("picname:::",picname)
    # print(arr.shape)
    # output_directory = os.path.dirname('F:\SCRGAN\data\ceshi/npdata/train')  # 提取文件的路径
    # output_name = os.path.splitext(os.path.basename("name.npy"))[0]  # 提取文件名
    #图片shape
    # if(picname[-3:]=="gt_"):
    #     disp_to_img = resize(arr, output_shape=(190, 250))
    
    print(arr.shape)
    

    if(len(arr.shape)==4):
        arr=arr[0]
        #arr2=480 640 -80 80-560
        arr_2=arr[:,80:560]
        arr_3=arr_2[0]
        print("arr3::",arr_3.shape)
        
        
        disp_to_img = resize(arr_3, output_shape=(480, 640))
        
    else:
        disp_to_img = resize(arr, output_shape=(480, 640))
        # save_name=picname
    save_path=path + picname + ".png"
    print(save_path)
    plt.imsave(save_path, disp_to_img, cmap='plasma')
    # plt.imsave(save_path, disp_to_img)
    # print(f)
    
