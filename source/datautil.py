"""
Utility for data processing
Author: Harry He @ NCA Lab, CBS, RIKEN
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pickle
from PIL import Image
from pycocotools import mask as coco_mask

def save_data(data,file,large_data=False):
    if not large_data:
        pickle.dump(data, open(file, "wb"))
        print("Data saved to ", file)
    else:
        pickle.dump(data, open(file, "wb"), protocol=4)
        print("Large Protocal 4 Data saved to ", file)

def load_data(file):
    data = pickle.load(open(file, "rb"))
    print("Data load from ", file)
    return data

def load_model(model,file,map_location=None,except_list=[]):
    try:
        if map_location is None:
            state_dict=torch.load(file)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(file,map_location=map_location))
        print("Model load from ", file)
    except Exception as inst:
        print(inst)
        pretrained_dict = torch.load(file)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (k not in except_list)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

def plot_mat(data,start=0,lim=1000,symmetric=False,title=None,tick_step=None,show=True,xlabel=None,ylabel=None, clim=None):
    if show:
        plt.figure()
    data=np.array(data)
    if len(data.shape) != 2:
        data=data.reshape(1,-1)
    img=data[:,start:start+lim]
    if symmetric:
        plt.imshow(img, cmap='seismic',clim=(-np.amax(np.abs(data)), np.amax(np.abs(data))))
        # plt.imshow(img, cmap='seismic', clim=(-2,2))
    else:
        plt.imshow(img, cmap='seismic',clim=clim)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if tick_step is not None:
        plt.xticks(np.arange(0, len(img[0]), tick_step))
        plt.yticks(np.arange(0, len(img), tick_step))
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show:
        plt.show()

class ClevrDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_path, get_mode="maskedimage_posishapematerial"):
        """
        Clevr Dataset Util
        :param path: pictures path
        :param json_path: path for json file
        :param img_seg_path: pretrain CLEVR segmentation model
        """
        self.image_path = image_path

        self.json_clevr = load_data(json_path)
        self.img_size = [320, 480] # [H, W]

        self.masks = []
        self.imgs = []
        print("Parsing dataset ...")
        for iis in range(len(self.json_clevr["scenes"])):
            self.imgs.append(self.json_clevr["scenes"][iis]["image_filename"])

        self.get_mode = get_mode

        self.color_map={
            "gray":0, "blue":1, "brown":2, "yellow":3, "red":4, "green":5, "purple":6, "cyan":7
        }
        self.shape_map = {
            "cube": 0, "cylinder": 1, "sphere": 2
        }
        self.material_map = {
            "metal":0, "rubber":1
        }

    def __getitem__(self, idx):
        if self.get_mode == "full":
            return self.getitem_full(idx)
        elif self.get_mode in ["maskedimage_colorshape","maskedimage_posicolorshape","maskedimage_posicolormaterial","auto_encode","maskedimage_posishapematerial"
                               ,"maskedimage_shape","maskedimage_color", "maskedimage_posi","auto_encode_focused"]:
            return self.getitem_mask_posicolorshape(idx)
        elif self.get_mode == "whole_pic_auto":
            return self.getitem_mask_wholepicauto(idx)
        elif self.get_mode == "hidden_only_no_pic_posi":
            return self.hidden_only_no_pic_posi(idx)

    def getitem_full(self, idx):
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        json_scene = self.json_clevr["scenes"][idx]

        mask_t = np.zeros(self.img_size)
        for iio in range(len(json_scene["objects"])):
            rle = json_scene["objects"][iio]["mask"]
            compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            mask = coco_mask.decode(compressed_rle)
            mask = mask*(iio + 1)
            mask_t = mask_t+mask

        return img, mask_t, json_scene

    def getitem_mask_posicolorshape(self,idx):
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)/255.0

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        objp = int(np.random.rand()*Nobj)
        rle = json_scene["objects"][objp]["mask"]
        compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
        mask = coco_mask.decode(compressed_rle)
        self.img = img
        self.mask = mask
        masked_img = img * mask.reshape(self.img_size + [1])
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img).type(torch.FloatTensor)
        posix = json_scene["objects"][objp]["pixel_coords"][0] / self.img_size[1]
        posiy = json_scene["objects"][objp]["pixel_coords"][1] / self.img_size[0]
        color = self.color_map[json_scene["objects"][objp]["color"]]
        shape = self.shape_map[json_scene["objects"][objp]["shape"]]
        material = self.material_map[json_scene["objects"][objp]["material"]]

        if self.get_mode =="maskedimage_posicolorshape":
            return masked_img, np.array([float(posix),float(posiy),color,shape])

        elif self.get_mode =="maskedimage_colorshape":
            return masked_img, np.array([color,shape])

        elif self.get_mode == "maskedimage_posicolormaterial":
            return masked_img, np.array([float(posix),float(posiy),color, material])

        elif self.get_mode == "maskedimage_posishapematerial":
            return masked_img, np.array([float(posix), float(posiy), shape, material])

        elif self.get_mode == "maskedimage_shape":
            return masked_img, shape

        elif self.get_mode == "maskedimage_color":
            return masked_img, color

        elif self.get_mode == "maskedimage_posi":
            if posix>2/3:
                posi_right=1
            else:
                posi_right=0
            return masked_img, posi_right

        elif self.get_mode == "auto_encode":
            return masked_img, masked_img

        elif self.get_mode == "auto_encode_focused":
            xrange = 75
            yrange = 75
            pixposix = json_scene["objects"][objp]["pixel_coords"][0]
            pixposiy = json_scene["objects"][objp]["pixel_coords"][1]
            xcropstart = pixposix-xrange
            xcropend = pixposix + xrange
            ycropstart = pixposiy - yrange
            ycropend = pixposiy + yrange
            # Boundary handling
            xlowshift=0
            if xcropstart<0:
                xlowshift = 0-xcropstart
                xcropstart=0
            xhighshift=0
            if xcropend>self.img_size[1]:
                xhighshift=xcropend-self.img_size[1]
                xcropend = self.img_size[1]
            ylowshift=0
            if ycropstart<0:
                ylowshift = 0-ycropstart
                ycropstart = 0
            yhighshift=0
            if ycropend>self.img_size[0]:
                yhighshift = ycropend-self.img_size[0]
                ycropend = self.img_size[0]
            focused_masked_img = torch.zeros((3, 2*xrange, 2*yrange))
            focused_masked_img[:,ylowshift:2*yrange-yhighshift,xlowshift:2*xrange-xhighshift] = masked_img[:, ycropstart:ycropend, xcropstart:xcropend]
            return masked_img, focused_masked_img

    def getitem_mask_wholepicauto(self,idx):
        assert self.get_mode == "whole_pic_auto"
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)/255.0

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        masked_imgl = torch.zeros([10, 3] + self.img_size).type(torch.FloatTensor)
        for iio in range(Nobj):

            rle = json_scene["objects"][iio]["mask"]
            compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            mask = coco_mask.decode(compressed_rle)
            masked_img = img * mask.reshape(self.img_size + [1])
            masked_img = np.transpose(masked_img, (2, 0, 1))
            masked_img = torch.from_numpy(masked_img).type(torch.FloatTensor)
            masked_imgl[iio,:,:,:] = masked_img

        return masked_imgl, masked_imgl

    def hidden_only_no_pic_posi(self,idx):

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        objp = int(np.random.rand() * Nobj)
        pAutocode = torch.from_numpy(json_scene["objects"][objp]["auto_code"]).type(torch.FloatTensor)
        posix = json_scene["objects"][objp]["pixel_coords"][0] / self.img_size[1]
        if posix > 2 / 3:
            posi_right = 1
        else:
            posi_right = 0
        return pAutocode, posi_right

    def __len__(self):
        return len(self.imgs)

    # def get_instance_segmentation_model(self, num_classes=2):
    #     """
    #     https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=mTgWtixZTs3X
    #     :param num_classes: number of classes
    #     :return:
    #     """
    #     import torchvision
    #     from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    #     from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    #
    #     # load an instance segmentation model pre-trained on COCO
    #     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    #
    #     # get the number of input features for the classifier
    #     in_features = model.roi_heads.box_predictor.cls_score.in_features
    #     # replace the pre-trained head with a new one
    #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #
    #     # now get the number of input features for the mask classifier
    #     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #     hidden_layer = 256
    #     # and replace the mask predictor with a new one
    #     model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                        hidden_layer,
    #                                                        num_classes)
    #
    #     return model


class MultipleChoiceClevrDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_path, epoch_len = 10000, question_mode="exist_colorshape"):
        """
        Clevr Dataset Util
        :param path: pictures path
        :param json_path: path for json file
        :param img_seg_path: pretrain CLEVR segmentation model
        """

        self.image_path = image_path
        self.json_clevr = load_data(json_path)
        self.epoch_len = epoch_len
        self.img_size = [320, 480] # [H, W]
        self.question_mode = question_mode
        self.checkmode = False
        self.singlecheckmode = False

        self.imgs = []
        print("Parsing dataset ...")
        for iis in range(len(self.json_clevr["scenes"])):
            self.imgs.append(self.json_clevr["scenes"][iis]["image_filename"])

        self.color_map={"gray":0, "blue":1, "brown":2, "yellow":3, "red":4, "green":5, "purple":6, "cyan":7}
        self.shape_map = {"cube": 0, "cylinder": 1, "sphere": 2}
        self.size_map = {"small": 0, "large": 1}
        self.material_map = {"rubber": 0, "metal": 1}

    def __getitem__(self, idx_temp):
        """
        Get 4 pictures and an answaer in 0,1,2,3
        :param idx:
        :return:
        """
        ## get a random sets of imgs of a question and an answer
        if self.question_mode == "exist_colorshape":
            idxl, answ = self.question_exist_colorshapesize(color="red", shape = "cube", size = None, material = None)
        elif self.question_mode == "exist_shapesize":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "sphere", size = "large", material = None)
        elif self.question_mode == "exist_shapesize2":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "cylinder", size = "large", material = None)
        elif self.question_mode == "exist_sizematerial":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = None, size = "small", material = "rubber")
        elif self.question_mode == "exist_sizecolor":
            idxl, answ = self.question_exist_colorshapesize(color="yellow", shape = None, size = "small", material = None)
        elif self.question_mode == "exist_colormaterial":
            idxl, answ = self.question_exist_colorshapesize(color="green", shape = None, size = None, material = "metal")
        elif self.question_mode == "posimost_property":
            idxl, answ = self.question_posimost_colorshapesize(posimost = "rightmost", color=None, shape = "cylinder", size = None, material = None)
        elif self.question_mode == "posiside_property":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = None, size = None, material = "rubber", posi_side = "left")
        elif self.question_mode == "posiside_property2":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "cylinder", size = None, material = None, posi_side = "right")
        elif self.question_mode == "posiside_property3":
            idxl, answ = self.question_exist_colorshapesize(color=None, shape = "sphere", size = None, material = None, posi_side = "right")
        elif self.question_mode == "posimost_property2":
            idxl, answ = self.question_posimost_colorshapesize(posimost = "rightmost", color=None, shape = None, size = "small", material = None)
        elif self.question_mode == "number_shape":
            idxl, answ = self.question_number_colorshapesize(key_aim = ["shape","cube"], num = 3)
        else:
            raise Exception("Not implemented")
        dataxl = []
        self.imgl=[]
        self.mask_tl=[]
        Nobjl = []
        singecheckl=[]
        for idx in idxl:
            # dataxl.append(img.unsqueeze(0))
            autocodel, Nobj = self.getitem_autocodeperobj(idx)
            dataxl.append(autocodel.unsqueeze(0))
            Nobjv = torch.zeros(10)
            Nobjv[:Nobj]=1
            Nobjl.append(Nobjv)
            if self.checkmode:
                print("Image getting ...")
                img = self.getitem_maskedperobj(idx)
                self.imgl.append(self.img)
                self.mask_tl.append(self.mask_t)
                self.idxl = idxl
            if self.singlecheckmode:
                # print(self.singlecheck.shape) # torch.Size([4, 10])
                singecheckl.append(self.singlecheck)

        datax = torch.cat(dataxl,dim=0)
        Nobjl = torch.stack(Nobjl)
        if self.singlecheckmode:
            singecheckmat = torch.stack(singecheckl)
            return datax, Nobjl, answ, singecheckmat
        else:
            return datax, Nobjl, answ

    def collate_fn(self, batchdata):
        datax = [item[0].view([1]+list(item[0].shape)) for item in batchdata]
        Nobjl = [item[1].view([1]+list(item[1].shape)) for item in batchdata]
        answ = [item[2] for item in batchdata]
        datax = torch.cat(datax,dim=0)
        Nobjl = torch.cat(Nobjl, dim=0)
        answ = torch.LongTensor(answ)
        if self.singlecheckmode:
            batchcheckmat = [item[3].view([1] + list(item[3].shape)) for item in batchdata]
            batchcheckmat = torch.cat(batchcheckmat, dim=0)
            return (datax, Nobjl), answ, batchcheckmat
        else:
            return (datax, Nobjl), answ

    def question_exist_colorshapesize(self, color=None, shape = "cube", size = "large", material = None, posi_side = None):

        checklist=[]
        if color is not None:
            checklist.append(["color",color])
        if shape is not None:
            checklist.append(["shape",shape])
        if size is not None:
            checklist.append(["size",size])
        if material is not None:
            checklist.append(["material",material])

        def checkposi(xy, posi_flag):
            if posi_flag == "left":
                if xy[0]/self.img_size[1]<1/3:
                    return True
                else:
                    return False
            elif posi_flag == "right":
                if xy[0]/self.img_size[1]>2/3:
                    return True
                else:
                    return False
            else:
                raise Exception("Checkposi not implemented")


        json_scene = self.json_clevr["scenes"]
        Nscene = len(json_scene)

        ## get one exist scene
        while True:
            idp_exist = int(np.random.rand()*Nscene)
            exist_flag = False
            for obj in json_scene[idp_exist]["objects"]:
                andFlag = True
                for itemd in checklist:
                    andFlag = (andFlag and obj[itemd[0]]==itemd[1])
                if posi_side is not None:
                    andFlag = (andFlag and checkposi(obj["pixel_coords"],posi_side))
                if andFlag:
                    exist_flag=True
                    break
            if exist_flag:
                break
        ## get 3 non-exist scene
        idp_Nexistl=[]
        while True:
            idp_Nexist = int(np.random.rand()*Nscene)
            exist_flag = False
            for obj in json_scene[idp_Nexist]["objects"]:
                andFlag = True
                for itemd in checklist:
                    andFlag = (andFlag and obj[itemd[0]] == itemd[1])
                if posi_side is not None:
                    andFlag = (andFlag and checkposi(obj["pixel_coords"],posi_side))
                if andFlag:
                    exist_flag = True
                    break
            if not exist_flag:
                idp_Nexistl.append(idp_Nexist)
                if len(idp_Nexistl) >= 3:
                    break
        idxl = [idp_exist] + idp_Nexistl
        np.random.shuffle(idxl)
        return idxl , idxl.index(idp_exist)

    def question_posimost_colorshapesize(self, posimost = "rightmost", color=None, shape = None, size = None, material = "metal"):

        checklist=[]
        if color is not None:
            checklist.append(["color",color])
        if shape is not None:
            checklist.append(["shape",shape])
        if size is not None:
            checklist.append(["size",size])
        if material is not None:
            checklist.append(["material",material])

        posimost_dict = {
            "leftmost":0,
            "rightmost":1,
            "farmost":2,
            "nearmost":3
        }

        json_scene = self.json_clevr["scenes"]
        Nscene = len(json_scene)

        ## get one exist scene
        while True:
            idp_exist = int(np.random.rand()*Nscene)
            exist_flag = False
            posimost_id = [-1, 1, -1, -1] # [leftmost, rightmost, farmost, nearmost] ids
            posimost_record = [1.0, 0.0, 1.0, 0.0]
            for iid, obj in enumerate(json_scene[idp_exist]["objects"]):
                posix = obj["pixel_coords"][0] / self.img_size[1]
                posiy = obj["pixel_coords"][1] / self.img_size[0]
                if posix<posimost_record[0]:
                    posimost_id[0] = iid
                    posimost_record[0] = posix
                if posix>posimost_record[1]:
                    posimost_id[1] = iid
                    posimost_record[1] = posix
                if posiy<posimost_record[2]:
                    posimost_id[2] = iid
                    posimost_record[2] = posix
                if posiy>posimost_record[3]:
                    posimost_id[3] = iid
                    posimost_record[3] = posix
            andFlag = True
            objpick = json_scene[idp_exist]["objects"][posimost_id[posimost_dict[posimost]]]
            for itemd in checklist:
                andFlag = (andFlag and objpick[itemd[0]]==itemd[1])
            if andFlag:
                exist_flag=True
            if exist_flag:
                break
        ## get 3 non-exist scene
        idp_Nexistl = []
        while True:
            idp_Nexist = int(np.random.rand()*Nscene)
            exist_flag = False
            posimost_id = [-1, -1, -1, -1] # [leftmost, rightmost, farmost, nearmost] ids
            posimost_record = [1.0, 0.0, 1.0, 0.0]
            for iid, obj in enumerate(json_scene[idp_Nexist]["objects"]):
                posix = obj["pixel_coords"][0] / self.img_size[1]
                posiy = obj["pixel_coords"][1] / self.img_size[0]
                if posix<posimost_record[0]:
                    posimost_id[0] = iid
                    posimost_record[0] = posix
                if posix>posimost_record[1]:
                    posimost_id[1] = iid
                    posimost_record[1] = posix
                if posiy<posimost_record[2]:
                    posimost_id[2] = iid
                    posimost_record[2] = posiy
                if posiy>posimost_record[3]:
                    posimost_id[3] = iid
                    posimost_record[3] = posiy
            andFlag = True
            objpick = json_scene[idp_Nexist]["objects"][posimost_id[posimost_dict[posimost]]]
            for itemd in checklist:
                andFlag = (andFlag and objpick[itemd[0]]==itemd[1])
            if andFlag:
                exist_flag=True

            if not exist_flag:
                idp_Nexistl.append(idp_Nexist)
                if len(idp_Nexistl) >= 3:
                    break

        idxl = [idp_exist] + idp_Nexistl
        np.random.shuffle(idxl)
        return idxl , idxl.index(idp_exist)

    def question_number_colorshapesize(self, key_aim = ["shape","cube"], num = 3):

        json_scene = self.json_clevr["scenes"]
        Nscene = len(json_scene)

        ## get one exist scene
        while True:
            idp_exist = int(np.random.rand()*Nscene)
            cnt = 0
            for obj in json_scene[idp_exist]["objects"]:
                if obj[key_aim[0]] == key_aim[1]:
                    cnt += 1
            if cnt == num:
                break
        ## get 3 non-exist scene
        idp_Nexistl=[]
        while True:
            idp_Nexist = int(np.random.rand()*Nscene)
            cnt = 0
            for obj in json_scene[idp_Nexist]["objects"]:
                if obj[key_aim[0]] == key_aim[1]:
                    cnt += 1
            if cnt != num:
                idp_Nexistl.append(idp_Nexist)
                if len(idp_Nexistl) >= 3:
                    break
        idxl = [idp_exist] + idp_Nexistl
        np.random.shuffle(idxl)
        return idxl , idxl.index(idp_exist)

    def getitem_maskedperobj(self,idx):
        img_path = os.path.join(self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)/255.0
        self.img = img

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        masked_imgl = torch.zeros([10, 3]+self.img_size).type(torch.FloatTensor)

        mask_t = np.zeros(self.img_size)
        for objp in range(Nobj):
            rle = json_scene["objects"][objp]["mask"]
            compressed_rle = coco_mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            mask = coco_mask.decode(compressed_rle)
            masked_img = img * mask.reshape(self.img_size + [1])
            masked_img = np.transpose(masked_img, (2, 0, 1))
            masked_img = torch.from_numpy(masked_img).type(torch.FloatTensor)
            masked_imgl[objp,:,:,:]=masked_img
            maskid = mask * (objp + 1)
            mask_t = mask_t + maskid
        self.mask_t=mask_t

        return masked_imgl # [10, 3, 320, 480]

    def getitem_autocodeperobj(self, idx):

        json_scene = self.json_clevr["scenes"][idx]

        Nobj = len(json_scene["objects"])
        codelen = len(json_scene["objects"][0]["auto_code"])
        autocodel = torch.zeros([10, codelen]).type(torch.FloatTensor)
        for objp in range(Nobj):
            autocodel[objp,:] = torch.from_numpy(json_scene["objects"][objp]["auto_code"]).type(torch.FloatTensor)

        self.singlecheck = torch.zeros((10,4))

        posixl=[]
        for iio in range(Nobj):
            posixl.append(self.json_clevr["scenes"][idx]['objects'][iio]["pixel_coords"][0])

        iio_xmax = np.argmax(posixl)
        self.singlecheck[iio_xmax, 0] = 1

        for iio in range(Nobj):
            if self.json_clevr["scenes"][idx]['objects'][iio]["color"]=="yellow":
                self.singlecheck[iio, 1]=1
            if self.json_clevr["scenes"][idx]['objects'][iio]["size"]=="small":
                self.singlecheck[iio, 3] = 1
            # if self.json_clevr["scenes"][idx]['objects'][iio]["color"]=="red":
            #     self.singlecheck[iio, 1]=1
            # if self.json_clevr["scenes"][idx]['objects'][iio]["shape"]=="cube":
            #     self.singlecheck[iio, 3] = 1
            # if self.json_clevr["scenes"][idx]['objects'][iio]["shape"]=="cylinder":
            #     self.singlecheck[iio, 3] = 1
            # posix = posixl[iio]
            # if posix/self.img_size[1]>2/3: # right side
            #     self.singlecheck[iio, 0] = 1


        return autocodel, Nobj

    def __len__(self):
        return self.epoch_len