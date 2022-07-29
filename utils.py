import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import numpy as np
import cv2
import math

# https://github.com/jorge-pessoa/pytorch-colors
import pytorch_colors



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def RGB2YUV(rgb):
  yuv = torch.FloatTensor(rgb.shape)

  r = rgb[:,0:1]
  g = rgb[:,1:2]
  b = rgb[:,2:3]

  y = 0.299*r + 0.587*g + 0.114*b
  u = 0.492*(b-y)
  v = 0.877*(r-y)

  yuv[:,0:1] = y
  yuv[:,1:2] = u
  yuv[:,2:3] = v

  return yuv

def YUV2RGB(yuv):
  rgb = torch.FloatTensor(yuv.shape)
  y = yuv[:,0:1]
  u = yuv[:,1:2]
  v = yuv[:,2:3]

  r = y + 1.14*v
  g = y - 0.395*u - 0.581*v
  b = y + 2.032*u

  rgb[:,0:1] = r
  rgb[:,1:2] = g
  rgb[:,2:3] = b

  return rgb




def torch2cv2(img):
    return np.array(transforms.ToPILImage()(torch.clip(img,0,1)))[:,:,::-1]


class imagePatchesDataset(Dataset):
    def __init__(self,datasetnoised,transform,yuv=True):
        self.noise=datasetnoised
        self.transform=transform
        self.yuv = yuv
  
    def __len__(self):
        return len(self.noise)
  
    def __getitem__(self,idx):
        xNoise=self.noise[idx]
        if self.transform != None:
            xNoise=self.transform(xNoise)
            if self.yuv:
                xNoise = pytorch_colors.rgb_to_yuv(xNoise)      

        return xNoise


def denoise_spatial_patches(img,model,patch_size=256,pad=20,BATCH_SIZE=8,device=device):
    padded = cv2.cvtColor(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB)
    imgs = []
    patch_loactions = {}
    counter = 0
    for y in range(0+pad,img.shape[0]+pad,patch_size):
        for x in range(0+pad,img.shape[1]+pad,patch_size):
            y2 = y+patch_size
            extend_y = 0
            extend_x = 0
            if y2 > img.shape[0]:
                y2 = img.shape[0]
                extend_y = patch_size-y2+y
            x2 = x+patch_size
            if x2 > img.shape[1]:
                x2 = img.shape[1]
                extend_x = patch_size-x2+x
        
            if extend_y or extend_x:
                imgs.append(cv2.copyMakeBorder(padded[y-pad:y2+pad,x-pad:x2+pad], 0, extend_y, 0, extend_x, cv2.BORDER_REPLICATE))
                if not extend_y: # only x
                    patch_loactions[counter] = (y-pad,y2-pad,x-pad,x2)
                elif not extend_x: # only y
                    patch_loactions[counter] = (y-pad,y2,x-pad,x2-pad)
                else: # Both x and y
                    patch_loactions[counter] = (y-pad,y2,x-pad,x2)
            else:
                imgs.append(padded[y-pad:min(y+patch_size+pad,img.shape[0]+pad),x-pad:min(x+patch_size+pad,img.shape[1]+pad)])
                patch_loactions[counter] = (y-pad,y2-pad,x-pad,x2-pad)
            counter += 1
  


    tsfms=transforms.Compose([transforms.ToTensor()])

    trainset=imagePatchesDataset(imgs,tsfms,yuv=True)
    trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False)


    img_recon = torch.zeros([3,img.shape[0],img.shape[1]], dtype=torch.float32)
    
    counter = 0
    for noisy in trainloader:
        noisy = noisy.to(device)
        with torch.no_grad():
            output=model(noisy)
    
        output = YUV2RGB(output)

        for i in range(len(output)):
            y,y2,x,x2 = patch_loactions[counter]
            img_recon[:,y:y2,x:x2] = output[i][:,pad:pad+y2-y,pad:pad+x2-x]
            counter += 1
    
    return img_recon



def denoise_spatial(img,model,pad=20,device=device):
    padded = cv2.cvtColor(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB)

    BATCH_SIZE = 1

    tsfms=transforms.Compose([transforms.ToTensor()])

    trainset=imagePatchesDataset([padded],tsfms,yuv=True)
    trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False)

    yuv = True
    second_model = False
    for noisy in trainloader:
        noisy = noisy.to(device)
        with torch.no_grad():
            output=model(noisy)

    return YUV2RGB(output)[0].detach().cpu()[:,pad:-pad,pad:-pad]





class videoPatchesDataset(Dataset):
    def __init__(self,noisedconcats,transform,yuv=True):
        self.noisedconcats=noisedconcats
        self.transform=transform
        self.yuv = yuv

    def __len__(self):
        return len(self.noisedconcats)
  
    def __getitem__(self,idx):
        xNoise=self.noisedconcats[idx]
    
        if self.transform != None:
            xNoise=self.transform(xNoise)
            xNoise2 = torch.zeros(xNoise.shape, dtype=torch.float32)

            if self.yuv:
                xNoise2[:3] = pytorch_colors.rgb_to_yuv(xNoise[:3])
                xNoise2[3:6] = pytorch_colors.rgb_to_yuv(xNoise[3:6])   
                xNoise2[6:] = pytorch_colors.rgb_to_yuv(xNoise[6:]) 

        return xNoise2


def denoise_video_frame_patches(img0,img1,img2,model_spatial,model_temporal,patch_size=256,pad=20,BATCH_SIZE=8,device=device):
    model_temporal.eval()
    model_spatial.eval()
    
    padded0 = cv2.cvtColor(cv2.copyMakeBorder(img0, pad, pad, pad, pad, cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB)
    padded1 = cv2.cvtColor(cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB)
    padded2 = cv2.cvtColor(cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB)
    
    imgs0 = np.ndarray((math.ceil(img0.shape[0]/patch_size)*math.ceil(img0.shape[1]/patch_size),patch_size+2*pad,patch_size+2*pad,3),dtype=img0.dtype)
    imgs1 = np.ndarray((math.ceil(img0.shape[0]/patch_size)*math.ceil(img0.shape[1]/patch_size),patch_size+2*pad,patch_size+2*pad,3),dtype=img0.dtype)
    imgs2 = np.ndarray((math.ceil(img0.shape[0]/patch_size)*math.ceil(img0.shape[1]/patch_size),patch_size+2*pad,patch_size+2*pad,3),dtype=img0.dtype)
    
    patch_loactions = {}
    counter = 0
    for y in range(0+pad,img0.shape[0]+pad,patch_size):
        for x in range(0+pad,img0.shape[1]+pad,patch_size):
            y2 = y+patch_size
            extend_y = 0
            extend_x = 0
            if y2 > img0.shape[0]:
                y2 = img0.shape[0]
                extend_y = patch_size-y2+y
            x2 = x+patch_size
            if x2 > img0.shape[1]:
                x2 = img0.shape[1]
                extend_x = patch_size-x2+x
        
            if extend_y or extend_x:
                imgs0[counter] = cv2.copyMakeBorder(padded0[y-pad:y2+pad,x-pad:x2+pad], 0, extend_y, 0, extend_x, cv2.BORDER_REPLICATE)
                imgs1[counter] = cv2.copyMakeBorder(padded1[y-pad:y2+pad,x-pad:x2+pad], 0, extend_y, 0, extend_x, cv2.BORDER_REPLICATE)
                imgs2[counter] = cv2.copyMakeBorder(padded2[y-pad:y2+pad,x-pad:x2+pad], 0, extend_y, 0, extend_x, cv2.BORDER_REPLICATE)
                if not extend_y: # only x
                    patch_loactions[counter] = (y-pad,y2-pad,x-pad,x2)
                elif not extend_x: # only y
                    patch_loactions[counter] = (y-pad,y2,x-pad,x2-pad)
                else: # Both x and y
                    patch_loactions[counter] = (y-pad,y2,x-pad,x2)
            else:
                imgs0[counter] = padded0[y-pad:min(y+patch_size+pad,img0.shape[0]+pad),x-pad:min(x+patch_size+pad,img0.shape[1]+pad)]
                imgs1[counter] = padded1[y-pad:min(y+patch_size+pad,img1.shape[0]+pad),x-pad:min(x+patch_size+pad,img1.shape[1]+pad)]
                imgs2[counter] = padded2[y-pad:min(y+patch_size+pad,img2.shape[0]+pad),x-pad:min(x+patch_size+pad,img2.shape[1]+pad)]
                patch_loactions[counter] = (y-pad,y2-pad,x-pad,x2-pad)
            counter += 1
    
    
    
    imgs = np.concatenate((imgs0,imgs1,imgs2),3)
    
    tsfms=transforms.Compose([transforms.ToTensor()])
    trainset = videoPatchesDataset(imgs,tsfms,yuv=True)
    trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False)  
        
    
    img_recon = torch.zeros([3,img0.shape[0],img0.shape[1]], dtype=torch.float32)
    
    counter = 0
    for noisy in trainloader:
        noisy_spatials = torch.reshape(noisy, (noisy.shape[0]*3,  3, noisy.shape[2], noisy.shape[3])).to(device)
        with torch.no_grad():
            output=model_spatial(noisy_spatials)
            
        del noisy_spatials
        torch.cuda.empty_cache()
        
        denoised1 = torch.reshape(output, (output.shape[0]//3, 9, output.shape[2], output.shape[3]))
        del output
        torch.cuda.empty_cache()

        with torch.no_grad():
            denoised2 = model_temporal((denoised1,denoised1[:,3:6]))
            
        del denoised1
        torch.cuda.empty_cache()
        

        output = YUV2RGB(denoised2)
        del denoised2
        torch.cuda.empty_cache()
        
        for i in range(len(output)):
            y,y2,x,x2 = patch_loactions[counter]
            img_recon[:,y:y2,x:x2] = output[i][:,pad:pad+y2-y,pad:pad+x2-x]
            counter += 1
        
    
    return img_recon


def denoise_video_frame(imgn0,imgn1,imgn2,model_spatial,model_temporal,device=device):
    model_spatial.eval()
    model_temporal.eval()
    
    tsfms=transforms.Compose([transforms.ToTensor()])

    imgn0 = cv2.cvtColor(imgn0, cv2.COLOR_BGR2RGB)
    imgn1 = cv2.cvtColor(imgn1, cv2.COLOR_BGR2RGB)
    imgn2 = cv2.cvtColor(imgn2, cv2.COLOR_BGR2RGB)

    BATCH_SIZE = 3

    trainset=imagePatchesDataset([imgn0,imgn1,imgn2],tsfms,yuv=True)
    trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False)

    
    for noisy in trainloader:
        noisy = noisy.to(device)
        with torch.no_grad():
            output=model_spatial(noisy)

    denoised1 = torch.cat((output[0],output[1],output[2]),dim=0)[None,]
    
    with torch.no_grad():
        denoised2 = model_temporal((denoised1,output[1]))

    del output
    torch.cuda.empty_cache()

    denoised_frame = pytorch_colors.yuv_to_rgb(denoised2.detach())[0].cpu()
    
    del denoised2
    torch.cuda.empty_cache()
    
    return denoised_frame
