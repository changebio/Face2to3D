import os.path as osp
import torch
import numpy as np
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from utils.ddfa import ToTensorGjz, NormalizeGjz
import dlib
import face_alignment


def make_abs_path(d):
    return osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), d)


m = make_abs_path('models')

transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

# 2. load fan model for face detection and landmark used for face cropping
fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')
# 2. load dlib model for face detection and landmark used for face cropping
dlib_landmark_model = osp.join(m,'shape_predictor_68_face_landmarks.dat')
face_regressor = dlib.shape_predictor(dlib_landmark_model)
face_detector = dlib.get_frontal_face_detector()


def fan68(frame,fan=fan):
    preds = fan.get_landmarks(frame)[-1]
    pts = preds.T
    return pts

def dlib68(frame):
    rects = face_detector(frame, 1)
    for rect in rects:
        pts = face_regressor(frame, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
    return pts

def map_2d_18pts_2d(lms2d_68):
    _18_indx_3d22d = [17, 19, 21, 22, 24, 26, 36, 40, 39, 42, 46, 45, 31, 30, 35, 48, 66, 54]
    lms2d = lms2d_68[:,_18_indx_3d22d]
    lms2d[:,7] = (lms2d_68[:,37] + lms2d_68[:,40])/2
    lms2d[:,10] = (lms2d_68[:,43] + lms2d_68[:,46])/2
    lms2d[:,16] = (lms2d_68[:,62] + lms2d_68[:,66])/2
    return lms2d


def obtain_18pts_map(pts):
    pts = map_2d_18pts_2d(pts)
    ptsMap = np.zeros([120, 120]) - 1
    indx = np.int32(np.floor(pts))
#    print(pts)
    ptsMap[indx[1], indx[0]] = 1

    '''
    aa = ptsMap
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(aa)

    for ind in range(18):
        ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w', markeredgecolor='black', alpha=0.8)
    ax.axis('off')

    cv2.imwrite(('./imgs/lms_18pts/' + lms.split(',')[0]), ptsMap*255)
    '''
    return ptsMap

def comb_inputs(imgs, lmsMaps, permu=False):
    lmsMaps = np.array(lmsMaps).astype(np.float32)
    if permu == True:
        imgs = imgs.permute(0, 2, 3, 1)
    else:
        imgs = imgs
    outputs = [np.dstack((imgs[idx].cpu().numpy(),lmsMaps[idx])) for idx in range(imgs.shape[0])]
    outputs = np.array(outputs).astype(np.float32)
    return outputs

def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x.cuda()
_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))


def input_process_for_2dasl(img,lmkfun=dlib68):
    input_img = transform(img)#.unsqueeze(0)
    lmk68 = lmkfun(img) #new img
    lmk68 = np.array([lmk68]).astype(np.float32)
    lmk68 = lmk68[:,:2,:]
    lmk68[lmk68>119] = 119
    lmsMap = [obtain_18pts_map(aa) for aa in lmk68]
    comInput1 = comb_inputs(input_img[None,:,:,:], lmsMap, permu=True)
    comInput1 = _numpy_to_cuda(comInput1)
    input_img = comInput1.permute(0, 3, 1, 2)
    return input_img

def input_process_for_3ddfa(img):
    input_img = transform(img).unsqueeze(0)
    return input_img