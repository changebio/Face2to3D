import sys
sys.path.append("..")
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
from .resnet_xgtu_4chls import resnet50
import mobilenet_v1
from utils.inference import parse_roi_box_from_landmark,crop_img,predict_68pts,predict_dense,get_colors
from deec.write import param2pose
import os.path as osp
from .preinput import transform,obtain_18pts_map

def make_abs_path(d):
    return osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), d)


#morphmodel
STD_SIZE = 120
mode='gpu'

def image2input(img,roi_box,lmkfun):   
    img = crop_img(img, roi_box)
    img = cv2.resize(
        img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR
    )
    input_img = transform(img)
    lmk68 = lmkfun(img) 
    lmk68[lmk68>119] = 119
    ipt = torch.cat((input_img,torch.Tensor(obtain_18pts_map(lmk68)[None,:,:])),0)
    return ipt

def images2batch(imgs,lmks,lmkfun,batch = 128):
    n = min(len(imgs),batch)
    input_img_lst,roi_box_lst,inputs = [],[],[]
    for i in range(n):
        roi_box = parse_roi_box_from_landmark(lmks[i])
        img = crop_img(imgs[i], roi_box)
        img = cv2.resize(
            img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR
        )
        roi_box_lst.append(roi_box)
        input_img = transform(img)
        lmk68 = lmkfun(img) #new img
        lmk68[lmk68>119] = 119
        inputs.append(torch.cat((input_img,torch.Tensor(obtain_18pts_map(lmk68)[None,:,:])),0))
        
    return input_img_lst,roi_box_lst,torch.stack(inputs)

def load2dasl(checkpoint_fp,num_classes=62):
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['res_state_dict']
    model = resnet50(pretrained=False, num_classes=num_classes) # 62 = 12(pose) + 40(shape) +10(expression)
    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    return model

# 1. load pre-tained model
def load2DASL():
    checkpoint_fp = make_abs_path('2DASL/models/2DASL_checkpoint_epoch_allParams_stage2.pth.tar')
    model=load2dasl(checkpoint_fp)
    if mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()
    return model

def load3DDFA():
    # 1. load pre-tained model
    checkpoint_fp = make_abs_path('3DDFA/models/phase1_wpdc_vdc.pth.tar')
    arch = 'mobilenet_1'
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)[
        'state_dict'
    ]
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()
    return model

def video3d(video,model,landmark68,input_process,frameset=200):
    # 0. open video
    vc = cv2.VideoCapture(str(video) if len(video) == 1 else video)
    # read image
    success, frame = vc.read()
    frame_lst,param_lst,roi_box_lst,last_frame_pts = [],[],[],[]
    count=0
    yaw_lst = None
    img_lst = [0,0,0]
    while success:
        vc.set(cv2.CAP_PROP_POS_MSEC,frameset*count)
        frame_lst.append(frame)
        #last_frame_pts = []
        if len(last_frame_pts) == 0:
            pts = landmark68(frame) # dlib,fan...
            last_frame_pts.append(pts)
            
        for lmk in last_frame_pts:
            #try
            ## crop image
            roi_box = parse_roi_box_from_landmark(lmk)
            img = crop_img(frame, roi_box)
            img = cv2.resize(
                img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR
            )
            roi_box_lst.append(roi_box)
            
            #input process
            input_img = input_process(img)
            with torch.no_grad():
                if mode == 'gpu':
                    input_img = input_img.cuda()
                param = model(input_img)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            param_lst.append(param)

            pts68 = predict_68pts(param, roi_box)
            lmk[:] = pts68[:2]
            yaw,_,_ = param2pose(param)
            if yaw_lst == None:
                yaw_lst = [yaw,yaw,yaw]
            else:
                if np.abs(yaw)<yaw_lst[0]:
                    yaw_lst[0]=yaw
                    img_lst[0]=count
                elif np.abs(yaw-0.5)<np.abs(yaw_lst[1]-0.5):
                    yaw_lst[1]=yaw
                    img_lst[1]=count
                elif np.abs(yaw+0.5)<np.abs(yaw_lst[2]+0.5):
                    yaw_lst[2]=yaw
                    img_lst[2]=count
        
        success, frame = vc.read()
        count+=1
    
    par_lst=[]
    col_lst = []
    for i in img_lst:
        par = param_lst[i]
        frame = frame_lst[i]
        #plt.imshow(frame)
        #plt.show()
        roi_box = roi_box_lst[i]
        vertex = predict_dense(par, roi_box)
        col = get_colors(frame, vertex)
        col_lst.append(col)
        par_lst.append(par)
    param_np = np.array(par_lst)
    mean_param = param_np.mean(0)
    
    return mean_param,col_lst

def predict3d(model,input_img):
    with torch.no_grad():
        if mode == 'gpu':
            input_img = input_img.cuda()
        param = model(input_img)
        param = param.squeeze().cpu().numpy()
    return param

def selectimages(params):
    yaw_lst = None
    img_lst = [0,0,0]
    for count,param in enumerate(params):
        yaw,_,_ = param2pose(param)
        if yaw_lst == None:
            yaw_lst = [yaw,yaw,yaw]
        else:
            if np.abs(yaw)<yaw_lst[0]:
                yaw_lst[0]=yaw
                img_lst[0]=count
            elif np.abs(yaw-0.5)<np.abs(yaw_lst[1]-0.5):
                yaw_lst[1]=yaw
                img_lst[1]=count
            elif np.abs(yaw+0.5)<np.abs(yaw_lst[2]+0.5):
                yaw_lst[2]=yaw
                img_lst[2]=count
    return img_lst

def paramandcolors(frame_lst,roi_box_lst,params):
    img_lst = selectimages(params)
    par_lst=[]
    col_lst = []
    for i in img_lst:
        par = params[i]
        frame = frame_lst[i]
        roi_box = roi_box_lst[i]
        vertex = predict_dense(par, roi_box)
        col = get_colors(frame, vertex)
        col_lst.append(col)
        par_lst.append(par)
    param_np = np.array(par_lst)
    mean_param = param_np.mean(0)
    return mean_param,col_lst

def images3d(imgs,model,landmark68,input_process,frameset=200):
    # read image
    frame_lst,param_lst,roi_box_lst,last_frame_pts = [],[],[],[]
    yaw_lst = None
    img_lst = [0,0,0]
    for count, image_path in enumerate(imgs):
        # read image
        frame = cv2.imread(image_path)
        frame_lst.append(frame)
        last_frame_pts = []
        if len(last_frame_pts) == 0:
            pts = landmark68(frame) # dlib,fan...
            last_frame_pts.append(pts)
            
        for lmk in last_frame_pts:
            #try
            ## crop image
            roi_box = parse_roi_box_from_landmark(lmk)
            img = crop_img(frame, roi_box)
            img = cv2.resize(
                img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR
            )
            roi_box_lst.append(roi_box)
            
            #input process
            input_img = input_process(img)
            with torch.no_grad():
                if mode == 'gpu':
                    input_img = input_img.cuda()
                param = model(input_img)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            param_lst.append(param)

            #pts68 = predict_68pts(param, roi_box)
            #lmk[:] = pts68[:2]
            yaw,_,_ = param2pose(param)
            if yaw_lst == None:
                yaw_lst = [yaw,yaw,yaw]
            else:
                if np.abs(yaw)<yaw_lst[0]:
                    yaw_lst[0]=yaw
                    img_lst[0]=count
                elif np.abs(yaw-0.5)<np.abs(yaw_lst[1]-0.5):
                    yaw_lst[1]=yaw
                    img_lst[1]=count
                elif np.abs(yaw+0.5)<np.abs(yaw_lst[2]+0.5):
                    yaw_lst[2]=yaw
                    img_lst[2]=count
        
        #success, frame = vc.read()
        #count+=1
    
    par_lst=[]
    col_lst = []
    for i in img_lst:
        par = param_lst[i]
        frame = frame_lst[i]
        #plt.imshow(frame)
        #plt.show()
        roi_box = roi_box_lst[i]
        vertex = predict_dense(par, roi_box)
        col = get_colors(frame, vertex)
        col_lst.append(col)
        par_lst.append(par)
    param_np = np.array(par_lst)
    mean_param = param_np.mean(0)
    
    return mean_param,col_lst



