"""
PyTorch Modules of All kinds vol2 starting 2019/10/18
Author: Harry He @ NCA Lab, CBS, RIKEN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

def align_to(shape_from, shape_to):
    """
    Align shape from [128,10,4] to [128,10,7,4,40]
    :param shape_from:
    :param shape_to:
    :return: [128,10,1,4,1]
    """
    shape_re = []
    for id in shape_to:
        if id in shape_from:
            shape_re.append(id)
        else:
            shape_re.append(1)
    return shape_re

def cnn_outszie(HWin, kernel_size, stride=1, padding=0, dilation=1):
    HWout = (HWin+2*padding-dilation*(kernel_size-1)-1)/stride+1
    HWout = np.floor(HWout)
    return HWout

def deconv_outszie(HWin, kernel_size, stride=1, padding=0, dilation=1):
    HWout = (HWin-1)*stride-2*padding+kernel_size
    return HWout

class CAL_LOSS(torch.nn.Module):
    """
    An abstract loss wrapper
    """

    def __init__(self, model, para=None):
        super(self.__class__, self).__init__()
        self.model = model
        if para is None:
            para = dict([])
        self.para(para)

    def forward(self, datax, labels, schedule=1.0):

        output = self.model(datax, schedule=schedule)
        self.output = output
        if self.loss_flag == "cross_entropy":
            loss = self.CrossEntropyLoss(output, labels)
        elif self.loss_flag == "masked_cross_entropy":
            loss = self.Masked_CrossEntropyLoss(output, labels, self.model)
        elif self.loss_flag == "mse_loss":
            loss = self.MSELoss(output, labels[...,:output.shape[-2],:output.shape[-1]])*10000
        elif self.loss_flag == "mse_loss_wv":
            loss = self.MSELoss(output, labels)*100
        elif self.loss_flag == "posicolormaterial_loss":
            loss = self.posiColorMaterialLoss(output, labels)
        elif self.loss_flag == "posishapematerial_loss":
            loss = self.posiShapeMaterialLoss(output, labels)
        elif self.loss_flag == "senti_auto_loss":
            loss = self.SentiAutoLoss(output, labels)
        else:
            raise Exception("Not implemented")

        # print("Loss,",loss)
        if self.loss_mode=="train":
            loss = self.cal_loss_reg_recursize(self.model, loss)

        return loss

    def cal_loss_reg_recursize(self, model, loss):
        if hasattr(model, "loss_reg"):
            loss = loss + model.loss_reg
            # print("A", model.loss_reg)
        if hasattr(model, "submodels"):
            # print("B")
            for submodel in model.submodels:
                loss = self.cal_loss_reg_recursize(submodel, loss)
        return loss

    def find_para_recursive(self, model, para_str):
        if hasattr(model, para_str):
            return getattr(model,para_str)
        elif hasattr(model, "submodels"):
            # print("B")
            for submodel in model.submodels:
                return self.find_para_recursive(submodel, para_str)
        else:
            raise Exception("Attr not found!")


    def para(self,para):
        self.misc_para = para
        self.loss_flag = para.get("loss_flag", "posicolorshape_loss")
        self.loss_mode = para.get("loss_mode", "train")
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)

    def posiColorMaterialLoss(self, output, labels): # Also good for posiColorMaterial

        device=output.device
        loss_mse = torch.nn.MSELoss()
        lossposi = torch.sqrt(loss_mse(output[:,0:2],labels[:,0:2].type(torch.FloatTensor).to(device)))
        ## Loss, color
        lossc = torch.nn.CrossEntropyLoss()
        losscolor = lossc(output[:, 2:10], labels[:, 2].type(torch.LongTensor).to(device))
        ## Loss, shape
        lossshape = lossc(output[:, 10:], labels[:, 3].type(torch.LongTensor).to(device))
        # print(lossposi, losscolor, lossshape)
        loss = losscolor+lossshape+lossposi
        return loss

    def posiShapeMaterialLoss(self, output, labels):

        device=output.device
        loss_mse = torch.nn.MSELoss()
        lossposi = torch.sqrt(loss_mse(output[:,0:2],labels[:,0:2].type(torch.FloatTensor).to(device)))
        ## Loss, shape
        lossc = torch.nn.CrossEntropyLoss()
        lossshape = lossc(output[:, 2:5], labels[:, 2].type(torch.LongTensor).to(device))
        ## Loss, material
        lossmaterial = lossc(output[:, 5:], labels[:, 3].type(torch.LongTensor).to(device))
        # print(lossposi, lossshape, lossmaterial)
        loss = lossposi+lossshape+lossmaterial
        return loss

    def SentiAutoLoss(self, output, labels):
        device = labels.device
        lossc = torch.nn.CrossEntropyLoss()
        output_s = output[...,:5]
        labels_s = labels[...,0]
        output_a = output[..., 5:]
        labels_a = labels[...,1]
        loss_s = lossc(output_s.permute(0,2,1), labels_s.type(torch.LongTensor).to(device))
        loss_a = lossc(output_a.permute(0, 2, 1), labels_a.type(torch.LongTensor).to(device))
        # print(loss_s,loss_a)
        return loss_s+loss_a

    def CrossEntropyLoss(self, output, labels):
        device = labels.device
        lossc = torch.nn.CrossEntropyLoss()
        if not self.sample_mode:
            if len(output.shape)==2:
                loss = lossc(output, labels.type(torch.LongTensor).to(device))
            elif len(output.shape)==3:
                loss = lossc(output.permute(0,2,1), labels.type(torch.LongTensor).to(device))
        else:
            labels = labels.type(torch.LongTensor).to(device)
            labels = labels.view(labels.shape[0],1).expand(labels.shape[0],self.sample_size)
            loss = lossc(output, labels)
        return loss

    def Masked_CrossEntropyLoss(self, output, labels, model):
        """
        transformer style masked loss
        :param output: [batch, w, l_size]
        :param labels:
        :param input_mask:
        :return:
        """
        input_mask = self.find_para_recursive(model, "input_mask")
        device = labels.device
        lossc=torch.nn.CrossEntropyLoss(reduce=False)
        assert len(output.shape)==3
        loss = lossc(output.permute(0,2,1), labels.type(torch.LongTensor).to(device))
        lossm = torch.sum(loss*(1-input_mask))/torch.sum(1-input_mask)
        return lossm

    def MSELoss(self, output, labels):
        loss_mse = torch.nn.MSELoss()
        loss = loss_mse(output, labels)
        return loss

class ABS_CNN_SEQ(torch.nn.Module):
    """
    An abstract cooperation container for seqential model
    """

    def __init__(self, seq1_coop, seq2_train, para=None):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        self.seq1_coop = seq1_coop
        self.seq2_train = seq2_train
        self.submodels = [seq1_coop, seq2_train]

        if para is None:
            para = dict([])
        self.para(para)

        if not self.seq1_coop_train_flag:
            for param in self.seq1_coop.parameters():
                param.requires_grad = False

        self.save_para = {
            "model_para":[seq1_coop.save_para, seq2_train.save_para],
            "type": str(self.__class__),
            "misc_para": para
        }

    def para(self,para):
        self.misc_para = para
        self.seq1_coop_train_flag = para.get("seq1_coop_train_flag", False)
        self.auto_code_context_flag = para.get("auto_code_context_flag",False)

    def forward(self, datax, schedule=1.0):
        if self.seq1_coop_train_flag:
            out_seq1 = self.seq1_coop(datax, schedule=schedule)
        else:
            out_seq1 = self.seq1_coop(datax, schedule=1.0)
        self.out_seq1=out_seq1
        if self.auto_code_context_flag == True:
            self.auto_code_context=torch.cat((self.seq1_coop.cooprer.context,self.seq1_coop.trainer.context),dim=-1)
        # out_seq2 = self.seq2_train(out_seq1, labels, schedule=schedule)

        output = self.seq2_train(out_seq1, schedule=schedule)
        self.output = output

        return output

class ABS_CNN_COOP(torch.nn.Module):
    """
    An abstract cooperation container for Cooperation model
    """

    def __init__(self, cooprer, trainer, para=None):
        """

        :param trainer: trainer can be None
        :param cooprer: cooprer is not trained
        :param para:
        """
        super(self.__class__, self).__init__()

        self.cooprer = cooprer
        self.trainer = trainer
        self.submodels = [cooprer, trainer]

        if para is None:
            para = dict([])
        self.para(para)

        if not self.cooprer_train_flag:
            for param in self.cooprer.parameters():
                param.requires_grad = False

        self.save_para = {
            "model_para": [cooprer.save_para, trainer.save_para],
            "type": str(self.__class__),
            "misc_para": para
        }

    def para(self, para):
        self.misc_para = para
        self.cooprer_train_flag = para.get("cooprer_train_flag", False)
        self.coop_mode = para.get("coop_mode", "concatenate")

    def forward(self, datax, schedule=1.0):
        self.device = datax.device
        if self.cooprer_train_flag:
            out_coop = self.cooprer(datax, schedule=schedule)
        else:
            out_coop = self.cooprer(datax, schedule=1.0)
        out_train = self.trainer(datax, schedule=schedule)
        if self.coop_mode == "concatenate":
            output = torch.cat([out_coop,out_train],dim=-1)
            self.output = output
        elif self.coop_mode == "sum":
            output = out_coop+out_train
            self.output = output
        return output

class FF_MLP(torch.nn.Module):
    """
    Feed forward multi-layer perceotron
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        # self.coop_mode=False
        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        if len(self.mlp_layer_para)>0:
            mlp_layer_para0 = [self.input_size] + self.mlp_layer_para
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(mlp_layer_para0[iil], mlp_layer_para0[iil+1], bias=True),
                                    torch.nn.LayerNorm(mlp_layer_para0[iil+1]),torch.nn.ReLU())
                for iil in range(len(mlp_layer_para0)-1)])
            self.h2o = torch.nn.Linear(mlp_layer_para0[-1], self.output_size)
        else:
            self.i2o = torch.nn.Linear(self.input_size, self.output_size)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para=para
        self.dropout_rate = para.get("dropout_rate", 0.0)

    def set_model_para(self,model_para):
        # model_para_h={
        #     "input_size": 64,
        #     "output_size":1,
        #     "mlp_layer_para": [32,16,8]
        # }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.output_size = model_para["output_size"]
        self.mlp_layer_para = model_para["mlp_layer_para"] # [hidden0, hidden1, ...]

    def forward(self, inputx, hidden1=None, add_logit=None, logit_mode=True, schedule=None, context_set=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        self.cuda_device=inputx.device

        inputx = (inputx - 0.5) / np.sqrt(0.25) # Fixed layernorm

        hidden = inputx
        if len(self.mlp_layer_para) > 0:
            for fmd in self.linear_layer_stack:
                hidden = fmd(hidden)
                # hidden = self.dropout(hidden)
            output=self.h2o(hidden)
        else:
            output = self.i2o(inputx)

        return output

    def initHidden(self,batch):
        return None

    def initHidden_cuda(self, device, batch):
        return None

class ConvNet(torch.nn.Module):
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)
        if para is None:
            para = dict([])
        self.para(para)

        Hsize = self.input_H
        Wsize = self.input_W
        sizel=[]
        sizele = []
        for iic in range(len(self.conv_para)):
            Hsize = cnn_outszie(Hsize, self.conv_para[iic][2], stride=self.conv_para[iic][3])
            Wsize = cnn_outszie(Wsize, self.conv_para[iic][2], stride=self.conv_para[iic][3])
            sizel.append([int(Hsize),int(Wsize)])
            Hsize = cnn_outszie(Hsize, self.maxpool_para, stride=self.maxpool_para)
            Wsize = cnn_outszie(Wsize, self.maxpool_para, stride=self.maxpool_para)
            sizele.append([int(Hsize),int(Wsize)])

        self.conv_layer_stack = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(*self.conv_para[iic]), #[channel_in, channel_out, kernal, stride] * conv_num
                                torch.nn.LayerNorm(sizel[iic]),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(self.maxpool_para))
            for iic in range(len(self.conv_para))])
        if self.res_per_conv>0:
            self.res_conv_stack = torch.nn.ModuleList([
                torch.nn.Sequential(*[ResConvModule(
                    {
                            "input_H": sizele[iic][0],
                            "input_W": sizele[iic][1],
                            "channel": self.conv_para[iic][1],
                            "kernel": 5,
                            "conv_layer": 2
                    }
                ) for iires in range(self.res_per_conv)])
                for iic in range(len(self.conv_para))])

        conv_out = int(Hsize * Wsize * self.conv_para[-1][1])
        print("Calculated Hsize: %s, Wsize: %s, ConvOut:%s."%(Hsize,Wsize,conv_out))

        if len(self.mlp_para)>0:
            self.i2h = torch.nn.Linear(conv_out, self.mlp_para[0])
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.mlp_para[iil], self.mlp_para[iil+1], bias=True), torch.nn.LayerNorm(self.mlp_para[iil+1]),torch.nn.ReLU())
                for iil in range(len(self.mlp_para)-1)])
            self.h2o = torch.nn.Linear(self.mlp_para[-1], self.output_size)
        else:
            self.i2o = torch.nn.Linear(conv_out, self.output_size)

        if self.postprocess_mode == "GSVIB":
            self.infobn=GSVIB_InfoBottleNeck(self.infobn_model,para=self.infobn_para)
        elif self.postprocess_mode == "gaussian_noise":
            self.gauss_noise = GaussNoise(std=self.noise_std)


        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para = para
        self.postprocess_mode = para.get("postprocess_mode", "posicolorshape_VIB")
        self.postprocess_partition = para.get("postprocess_partition", [2,5,7])
        self.noise_std = para.get("noise_std",0.2e-2)
        self.infobn_para = para.get("infobn_para", None)
        self.sfmsample_flag = para.get("sfmsample_flag", False)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)

    def set_model_para(self,model_para):
        # model_para = {
        #     "input_H": 320,
        #     "input_W": 480,
        #     "output_size": 2 + 8 + 3,  # posix, posiy, color, shape
        #     "conv_para": [[3, 16, 7, 2], [16, 16, 7, 2], [16, 16, 7, 1]],
        #     "maxpool_para": 2,
        #     "mlp_para": [128, 64]
        # }
        self.model_para = model_para
        self.input_H = model_para["input_H"]
        self.input_W = model_para["input_W"]
        self.output_size = model_para["output_size"]
        self.conv_para = model_para["conv_para"] #[channel_in, channel_out, kernal, stride] * conv_num
        self.maxpool_para = model_para.get("maxpool_para",2)
        self.mlp_para= model_para["mlp_para"] # [hidden1, hidden2, ...]
        self.res_per_conv = model_para.get("res_per_conv",0)
        self.infobn_model = model_para.get("infobn_model", None)

    def forward(self, datax, schedule=1.0):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """
        batch = datax.shape[0]

        for iil, fwd in enumerate(self.conv_layer_stack):
            datax = fwd(datax)
            if self.res_per_conv > 0:
                datax = self.res_conv_stack[iil](datax)


        if len(self.mlp_para) > 0 :
            hidden = self.i2h(datax.view(batch,-1))
            for fmd in self.linear_layer_stack:
                hidden = fmd(hidden)
            output = self.h2o(hidden)
        else:
            output = self.i2o(datax)

        if self.postprocess_mode == "GSVIB":
            output = self.infobn(output, schedule=schedule)
            self.context = self.infobn.context.view(*self.infobn.context.shape[:-2],-1)
            self.loss_reg = self.infobn.cal_regloss()
            self.output = output
            return output
        elif self.postprocess_mode == "posicolorshape_VIB":
            sep1 = self.postprocess_partition[0]
            sep2 = self.postprocess_partition[1]
            outputp = self.infobn(output[..., :sep1], schedule=schedule)
            self.loss_reg = self.infobn.cal_regloss()
            outputcs = self.softmax(output[:, sep1:sep2])
            outputss = self.softmax(output[:, sep2:])
            self.context = torch.cat([output[..., :sep1], outputcs, outputss], dim=-1)
            if not self.sfmsample_flag:
                self.output = torch.cat([outputp,output[..., sep1:]], dim=-1)
                return self.output
            else:
                outputc = self.gsoftmax(outputcs, temperature=np.exp(-5))
                outputs = self.gsoftmax(outputss, temperature=np.exp(-5))
                output = torch.cat([outputp,outputc,outputs],dim=-1)
                self.output = output
                return output
        elif self.postprocess_mode == "gaussian_noise":
            sep1 = self.postprocess_partition[0]
            sep2 = self.postprocess_partition[1]
            outputp = self.gauss_noise(output[..., :sep1])
            outputcs = self.softmax(output[:, sep1:sep2])
            outputss = self.softmax(output[:, sep2:])
            self.context = torch.cat([outputp, outputcs, outputss], dim=-1)
            if not self.sfmsample_flag:
                self.output = torch.cat([outputp,output[..., sep1:]], dim=-1)
                return self.output
            else:
                outputc = self.gsoftmax(outputcs, temperature=np.exp(-5))
                outputs = self.gsoftmax(outputss, temperature=np.exp(-5))
                output = torch.cat([outputp,outputc,outputs],dim=-1)
                self.output = output
                return output
        else:
            raise Exception("Unknown postprocess_mode")

class DeConvNet(torch.nn.Module):
    """
    Deconvolutional Net for auto-encoding
    """
    def __init__(self, model_para, para=None):
        super(self.__class__, self).__init__()

        if para is None:
            para = dict([])
        self.para(para)
        self.set_model_para(model_para)

        Hsize = self.output_H
        Wsize = self.output_W

        for iic in range(len(self.deconv_para)):
            iip=len(self.deconv_para)-iic-1
            Hsize = cnn_outszie(Hsize, self.deconv_para[iip][2], stride=self.deconv_para[iip][3])
            Hsize = cnn_outszie(Hsize, self.maxunpool_para, stride=self.maxunpool_para)
            Wsize = cnn_outszie(Wsize, self.deconv_para[iip][2], stride=self.deconv_para[iip][3])
            Wsize = cnn_outszie(Wsize, self.maxunpool_para, stride=self.maxunpool_para)

        conv_in = int(Hsize * Wsize * self.deconv_para[0][0])
        self.Hsize_in = int(Hsize)
        self.Wsize_in = int(Wsize)
        print("Calculated Hsize: %s, Wsize: %s, ConvOut:%s." % (Hsize, Wsize, conv_in))

        ## Calculate actual H,W size for layernorm
        sizel = []
        sizele= []
        for iic in range(len(self.deconv_para)):
            sizele.append([int(Hsize), int(Wsize)])
            Hsize = deconv_outszie(Hsize, self.maxunpool_para, stride=self.maxunpool_para)
            Hsize = deconv_outszie(Hsize, self.deconv_para[iic][2], stride=self.deconv_para[iic][3])
            Wsize = deconv_outszie(Wsize, self.maxunpool_para, stride=self.maxunpool_para)
            Wsize = deconv_outszie(Wsize, self.deconv_para[iic][2], stride=self.deconv_para[iic][3])
            sizel.append([int(Hsize), int(Wsize)])
        print("Actual output shape (%s,%s)."%(str(Hsize),str(Wsize)))

        self.layer_norm0 = torch.nn.LayerNorm(self.input_size)
        if len(self.mlp_para)>0:
            self.i2h = torch.nn.Linear(self.input_size, self.mlp_para[0])
            self.linear_layer_stack = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.mlp_para[iil], self.mlp_para[iil+1], bias=True), torch.nn.LayerNorm(self.mlp_para[iil+1]),torch.nn.ReLU())
                for iil in range(len(self.mlp_para)-1)])
            self.h2m = torch.nn.Linear(self.mlp_para[-1], conv_in)
        else:
            self.i2m = torch.nn.Linear(self.input_size, conv_in)

        if self.res_per_conv>0:
            self.res_conv_stack = torch.nn.ModuleList([
                torch.nn.Sequential(*[ResConvModule(
                    {
                            "input_H": sizele[iic][0],
                            "input_W": sizele[iic][1],
                            "channel": self.deconv_para[iic][0],
                            "kernel": 5,
                            "conv_layer": 2
                    }
                ) for iires in range(self.res_per_conv)])
                for iic in range(len(self.deconv_para))])

        self.conv_layer_stack = torch.nn.ModuleList([
            torch.nn.Sequential(
                                #torch.nn.MaxUnpool2d(self.maxunpool_para),
                                # Use deconvolution to replace MaxUnpool2d, no change in channel, same kernel and stride
                                torch.nn.ConvTranspose2d(self.deconv_para[iic][0],self.deconv_para[iic][0],self.maxunpool_para,self.maxunpool_para),
                                torch.nn.ConvTranspose2d(*self.deconv_para[iic]),
                                torch.nn.LayerNorm(sizel[iic]),
                                torch.nn.ReLU())
            for iic in range(len(self.deconv_para)-1)])

        if self.output_allow_negative_flag:
            self.conv_layer_stack.append(
                torch.nn.Sequential(
                    # torch.nn.MaxUnpool2d(self.maxunpool_para),
                    # Use deconvolution to replace MaxUnpool2d, no change in channel, same kernel and stride
                    torch.nn.ConvTranspose2d(self.deconv_para[-1][0], self.deconv_para[-1][0], self.maxunpool_para,
                                             self.maxunpool_para),
                    torch.nn.ConvTranspose2d(*self.deconv_para[-1]),
                )
            )
        else:
            self.conv_layer_stack.append(
                torch.nn.Sequential(
                    # torch.nn.MaxUnpool2d(self.maxunpool_para),
                    # Use deconvolution to replace MaxUnpool2d, no change in channel, same kernel and stride
                    torch.nn.ConvTranspose2d(self.deconv_para[-1][0], self.deconv_para[-1][0], self.maxunpool_para,
                                             self.maxunpool_para),
                    torch.nn.ConvTranspose2d(*self.deconv_para[-1]),
                    torch.nn.ReLU()
                )
            )

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def set_model_para(self,model_para):
        # model_para = {"input_size": 300,
        #               "output_size": 7,
        #               "gs_head_dim": 2,
        #               "gs_head_num": 11,
        #               "mlp_num_layers": [3, 0],
        #               "mlp_hidden": 80,
        #               }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.output_H = model_para["output_H"]
        self.output_W = model_para["output_W"]
        self.deconv_para = model_para["deconv_para"] #[channel_in, channel_out, kernal, stride] * conv_num
        self.maxunpool_para = model_para.get("maxunpool_para",2)
        self.mlp_para= model_para["mlp_para"] # [hidden1, hidden2, ...]
        self.res_per_conv = model_para.get("res_per_conv", 0)

    def para(self,para):
        self.misc_para = para
        self.output_allow_negative_flag = para.get("output_allow_negative_flag", False)

    def forward(self, datax, schedule=1.0):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """
        batch = datax.shape[0]

        datax = self.layer_norm0(datax)
        if len(self.mlp_para) > 0 :
            hidden = self.i2h(datax)
            for fmd in self.linear_layer_stack:
                hidden = fmd(hidden)
            tconvin = self.h2m(hidden)
        else:
            tconvin = self.i2m(datax)

        tconvin = tconvin.view(batch, self.deconv_para[0][0], self.Hsize_in, self.Wsize_in)

        for iil, fwd in enumerate(self.conv_layer_stack):
            if self.res_per_conv > 0:
                tconvin = self.res_conv_stack[iil](tconvin)
            tconvin = fwd(tconvin)

        self.outimage=tconvin

        return tconvin

class ResConvModule(torch.nn.Module):
    """
    ResNet Module, dimension, channels are same
    """
    def __init__(self, model_para):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)
        padding=int((self.kernel-1)/2) # Note kernel should be odd
        self.conv = torch.nn.Conv2d(self.channel,self.channel,self.kernel,1,padding)
        self.lnorm = torch.nn.LayerNorm([self.input_H,self.input_W])
        self.relu = torch.nn.ReLU()

    def set_model_para(self,model_para):
        # model_para = {
        #     "input_H": 320,
        #     "input_W": 480,
        #     "channel": 16,
        #     "kernel": 5,
        #     "conv_layer": 2
        # }
        self.model_para = model_para
        self.input_H = model_para["input_H"]
        self.input_W = model_para["input_W"]
        self.channel = model_para["channel"]
        self.kernel = model_para["kernel"] #[channel_in, channel_out, kernal, stride] * conv_num
        self.conv_layer = model_para.get("conv_layer",2)

    def forward(self, datax):
        """
        Input pictures (batch, Cin, Hin, Win)
        :param xin:
        :return:
        """

        datac=self.conv(datax)
        datac = self.lnorm(datac)
        datac = self.relu(datac)

        for iil in range(self.conv_layer-1):
            datac = self.conv(datac)
            datac = self.lnorm(datac)
            if iil<self.conv_layer-2:
                datac = self.relu(datac)

        datao = datac+datax
        datao = self.relu(datao)

        return datao

class ConvFF_MLP_CLEVR(torch.nn.Module):
    """
    Convolutional FF for CLEVR multiple choice.
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.HiddenConvFF = FF_MLP(model_para["HiddenConvFF"], para=para) # [b, 4, objN, hd]-->[b, 4, objN, 1]
        self.ObjConvFF = FF_MLP(model_para["ObjConvFF"], para=para) # [b, 4, objN] --> [b, 4, 1]
        # self.OutFF = FF_MLP(model_para["OutFF"], para=para) # [b, 4] --> [b, 4_softmax]

        if para is None:
            para = dict([])
        self.para(para)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

        # self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)
        # self.layer_norm = torch.nn.LayerNorm(model_para["HiddenConvFF"]["input_size"])

        # self.layer_norm0 = torch.nn.LayerNorm(model_para["HiddenConvFF"]["input_size"])
        # self.layer_norm1 = torch.nn.LayerNorm(model_para["ObjConvFF"]["input_size"])

    def para(self,para):
        self.misc_para=para
        self.dropout_rate = para.get("dropout_rate", 0.0)
        # self.sfmsample_flag = para.get("sfmsample_flag", True)
        # self.loss_flag = para.get("loss_flag", "posicolorshape")
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)

    def forward(self, datax, schedule=None):
        """
        Forward
        :param datax: [b, 4, objN, hd]
        :param hidden:
        :return:
        """
        ### softmax sample
        # if self.sfmsample_flag:
        #     if self.loss_flag == "posicolorshape":
        # output0 = self.gsoftmax(datax[:,:,:, 2:10], temperature=np.exp(-5))
        # output1 = self.gsoftmax(datax[:,:,:, 10:13], temperature=np.exp(-5))
        # output2 = self.gsoftmax(datax[:,:,:, 13:].view([*datax.shape[0:3],-1,2]), temperature=np.exp(-5))
        # output2 = output2.view([*output2.shape[0:3],-1])
        # datax = torch.cat([datax[:,:,:, :2], output0, output1, output2], dim=-1)
        # datax = self.layer_norm(datax)

        # datax = self.layer_norm0(datax)
        datax = self.HiddenConvFF(datax).squeeze()
        # datax = self.layer_norm1(datax)
        output = self.ObjConvFF(datax).squeeze()
        # output = self.ObjConvFF(datax.view(-1,40))[0]#.squeeze()
        # output = self.OutFF(output.permute(0,2,1)).permute(0,2,1)
        self.output = output

        return output

class Multitube_FF_MLP(torch.nn.Module):
    """
    Feed forward multi-tube perceptron (a multiple information flow tube, no mixing within tubes, each tube is an FF)
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        assert len(self.input_divide) == len(self.output_divide) == len(self.mlp_layer_para)
        assert np.sum(self.input_divide) == self.input_size
        assert np.sum(self.output_divide) == self.output_size

        # self.infobnvib = VIB_InfoBottleNeck(para={
        #     "multi_sample_flag":self.sample_mode,
        #     "sample_size":self.sample_size
        # })

        model_paral = []
        for iim in range(len(self.input_divide)):
            model_paral.append({
                    "input_size": self.input_divide[iim],
                    "output_size":self.output_divide[iim],
                    "mlp_layer_para": self.mlp_layer_para[iim]
            })

        self.ff_tubes = torch.nn.ModuleList([
            FF_MLP(model_paral[iim])
            for iim in range(len(self.input_divide))])

        self.infobnl = torch.nn.ModuleList([
            GSVIB_InfoBottleNeck(self.infobn_model[iim],para=self.infobn_para)
            for iim in range(len(self.input_divide))])

        self.layer_norm0 = torch.nn.LayerNorm(self.input_size)
        self.layer_norm1 = torch.nn.LayerNorm(self.output_size)
        # self.batch_norm0 = torch.nn.BatchNorm2d(4)
        # self.batch_norm1 = torch.nn.BatchNorm2d(4)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)
        self.gauss_noise = GaussNoise(std=self.noise_std)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para = para
        self.infobn_para = para.get("infobn_para", None)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)
        self.noise_std = para.get("noise_std",0.2e-2)

    def set_model_para(self,model_para):
        # model_para_h={
        #     "input_size": 2+8+3+64,
        #     "input_divide":[2,8,3,64]
        #     "output_size":12+12+12+32,
        #     "output_divide":[16,16,16,32]
        #     "mlp_layer_para": [[16,16],[16,16],[16,16],[32,32]]
        #     "infobn_model": [{"gs_head_dim":2,"gs_head_num":8},{"gs_head_dim":2,"gs_head_num":8},
        #                    {"gs_head_dim":2,"gs_head_num":8},{"gs_head_dim":2,"gs_head_num":16}]
        # }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.input_divide = model_para["input_divide"]
        self.output_size = model_para["output_size"]
        self.output_divide = model_para["output_divide"]
        self.mlp_layer_para = model_para["mlp_layer_para"] # [hidden0l, hidden1l, ...]
        self.infobn_model = model_para.get("infobn_model", None)

    def forward(self, datax, schedule=None):

        # print(torch.mean(datax), torch.var(datax))
        datax = (datax-0.38)/np.sqrt(0.25)

        startp=0
        dataml=[]
        for iim, fmd in enumerate(self.ff_tubes):
            datam = fmd(datax[...,startp:startp+self.input_divide[iim]])
            dataml.append(datam)
            startp+=self.input_divide[iim]

        gsamplel = []
        contextl = []
        for iim, infobn in enumerate(self.infobnl):
            gsample = infobn(dataml[iim], schedule=schedule)
            gsamplel.append(gsample)
            contextl.append(infobn.contextl)
        self.loss_reg = 0
        for iim in range(len(self.input_divide)):
            self.loss_reg = self.loss_reg + self.infobnl[iim].cal_regloss()

        self.context = torch.cat(contextl, dim=-1)
        output = torch.cat(gsamplel, dim=-1)
        self.output = output
        # print(torch.mean(output), torch.var(output))
        output = (output-0.5)/np.sqrt(0.1)

        return output

class Multitube_BiGRU_MLP(torch.nn.Module):
    """
    Feed forward multi-tube perceptron (a multiple information flow tube, no mixing within tubes, each tube is an FF)
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        assert len(self.input_divide) == len(self.output_divide)
        assert np.sum(self.input_divide) == self.input_size
        assert np.sum(self.output_divide) == self.output_size

        # self.infobnvib = VIB_InfoBottleNeck(para={
        #     "multi_sample_flag":self.sample_mode,
        #     "sample_size":self.sample_size
        # })

        model_paral = []
        for iim in range(len(self.input_divide)):
            model_paral.append({
                    "input_size": self.input_divide[iim],
                    "output_size":self.output_divide[iim],
                    "hidden_size":self.hidden_size,
                    "num_layers":self.num_layers,
                    "mlp_hidden": self.mlp_hidden,
                    "mlp_num_layers":self.mlp_num_layers,
            })

        self.ff_tubes = torch.nn.ModuleList([
            BiGRU_NLP(model_paral[iim])
            for iim in range(len(self.input_divide))])

        self.infobnl = torch.nn.ModuleList([
            GSVIB_InfoBottleNeck(self.infobn_model[iim],para=self.infobn_para)
            for iim in range(len(self.input_divide))])

        self.layer_norm0 = torch.nn.LayerNorm(self.input_size)
        self.layer_norm1 = torch.nn.LayerNorm(self.output_size)
        # self.batch_norm0 = torch.nn.BatchNorm2d(4)
        # self.batch_norm1 = torch.nn.BatchNorm2d(4)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)
        self.gauss_noise = GaussNoise(std=self.noise_std)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para = para
        self.infobn_para = para.get("infobn_para", None)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)
        self.noise_std = para.get("noise_std",0.2e-2)

    def set_model_para(self,model_para):
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.input_divide = model_para["input_divide"]
        self.output_size = model_para["output_size"]
        self.output_divide = model_para["output_divide"]
        self.hidden_size = model_para["hidden_size"]
        self.num_layers = model_para["num_layers"]
        self.mlp_hidden = model_para["mlp_hidden"]
        self.mlp_num_layers = model_para["mlp_num_layers"]
        self.infobn_model = model_para.get("infobn_model", None)

    def forward(self, datax, schedule=None):

        # print(torch.mean(datax), torch.var(datax))
        datax = (datax-0.38)/np.sqrt(0.25)

        startp=0
        dataml=[]
        for iim, fmd in enumerate(self.ff_tubes):
            datam = fmd(datax[...,startp:startp+self.input_divide[iim]])
            dataml.append(datam)
            startp+=self.input_divide[iim]

        gsamplel = []
        contextl = []
        for iim, infobn in enumerate(self.infobnl):
            gsample = infobn(dataml[iim], schedule=schedule)
            gsamplel.append(gsample)
            contextl.append(infobn.contextl)
        self.loss_reg = 0
        for iim in range(len(self.input_divide)):
            self.loss_reg = self.loss_reg + self.infobnl[iim].cal_regloss()

        self.context = torch.cat(contextl, dim=-1)
        output = torch.cat(gsamplel, dim=-1)
        self.output = output
        # print(torch.mean(output), torch.var(output))
        output = (output-0.5)/np.sqrt(0.1)

        return output

class Multitube_FF_MLP_CLEVR(torch.nn.Module):
    """
    Feed forward multi-tube perceptron (a multiple information flow tube, no mixing within tubes, each tube is an FF)
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        assert len(self.input_divide) == len(self.output_divide) == len(self.mlp_layer_para)
        assert np.sum(self.input_divide) == self.input_size
        assert np.sum(self.output_divide) == self.output_size

        # self.infobnvib = VIB_InfoBottleNeck(para={
        #     "multi_sample_flag":self.sample_mode,
        #     "sample_size":self.sample_size
        # })

        model_paral = []
        for iim in range(len(self.input_divide)):
            model_paral.append({
                    "input_size": self.input_divide[iim],
                    "output_size":self.output_divide[iim],
                    "mlp_layer_para": self.mlp_layer_para[iim]
            })

        self.ff_tubes = torch.nn.ModuleList([
            FF_MLP(model_paral[iim])
            for iim in range(len(self.input_divide))])

        self.infobnl = torch.nn.ModuleList([
            GSVIB_InfoBottleNeck(self.infobn_model[iim],para=self.infobn_para)
            for iim in range(len(self.input_divide))])

        self.layer_norm0 = torch.nn.LayerNorm(self.input_size)
        self.layer_norm1 = torch.nn.LayerNorm(self.output_size)
        # self.batch_norm0 = torch.nn.BatchNorm2d(4)
        # self.batch_norm1 = torch.nn.BatchNorm2d(4)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)
        self.gauss_noise = GaussNoise(std=self.noise_std)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para = para
        self.infobn_para = para.get("infobn_para", None)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)
        self.noise_std = para.get("noise_std",0.2e-2)

    def set_model_para(self,model_para):
        # model_para_h={
        #     "input_size": 2+8+3+64,
        #     "input_divide":[2,8,3,64]
        #     "output_size":12+12+12+32,
        #     "output_divide":[16,16,16,32]
        #     "mlp_layer_para": [[16,16],[16,16],[16,16],[32,32]]
        #     "infobn_model": [{"gs_head_dim":2,"gs_head_num":8},{"gs_head_dim":2,"gs_head_num":8},
        #                    {"gs_head_dim":2,"gs_head_num":8},{"gs_head_dim":2,"gs_head_num":16}]
        # }
        self.model_para = model_para
        self.input_size = model_para["input_size"]
        self.input_divide = model_para["input_divide"]
        self.output_size = model_para["output_size"]
        self.output_divide = model_para["output_divide"]
        self.mlp_layer_para = model_para["mlp_layer_para"] # [hidden0l, hidden1l, ...]
        self.infobn_model = model_para.get("infobn_model", None)

    def forward(self, dataxt, schedule=None):
        """
        Forward
        :param input: [window batch l_size]
        :param hidden:
        :return:
        """
        ### softmax sample
        # if self.sfmsample_flag:
        #     if self.loss_flag == "posicolorshape":
        ## Preprocessing
        datax = dataxt[0]
        # outputp = self.infobnvib(datax[..., :4], schedule=1.0)
        outputp = self.gauss_noise(datax[..., :2])
        if self.sample_mode:
            outputp = outputp.view([*outputp.shape[:2],1,*outputp.shape[2:]]).expand([*outputp.shape[:2],self.sample_size,*outputp.shape[2:]])
        output0 = self.gsoftmax(datax[... , 2:10], temperature=np.exp(-5))
        output1 = self.gsoftmax(datax[... , 10:12], temperature=np.exp(-5))
        output2 = self.gsoftmax(datax[... , 12:].view([*datax.shape[0:-1], -1, 2]), temperature=np.exp(-5))
        output2 = output2.view([*output2.shape[0:-2], -1])
        datax = torch.cat([outputp, output0, output1, output2], dim=-1)
        # datax = self.layer_norm0(datax)
        # print(torch.mean(datax), torch.var(datax))
        datax = (datax-0.45)/np.sqrt(0.25)

        startp=0
        dataml=[]
        for iim, fmd in enumerate(self.ff_tubes):
            datam = fmd(datax[...,startp:startp+self.input_divide[iim]])
            dataml.append(datam)
            startp+=self.input_divide[iim]

        gsamplel = []
        contextl = []
        for iim, infobn in enumerate(self.infobnl):
            gsample = infobn(dataml[iim], schedule=schedule)
            gsamplel.append(gsample)
            contextl.append(infobn.contextl)
        self.loss_reg = 0
        for iim in range(len(self.input_divide)):
            self.loss_reg = self.loss_reg + self.infobnl[iim].cal_regloss()

        self.contextl = torch.cat(contextl, dim=-1)
        output = torch.cat(gsamplel, dim=-1)
        objNl = dataxt[1]
        reshape = align_to(objNl.shape, output.shape)
        output = output*objNl.view(reshape)
        self.output = output
        # output = self.layer_norm1(output)
        # print(torch.mean(output), torch.var(output))
        output = (output-0.3)/np.sqrt(0.1)

        return output

class Gumbel_Softmax(torch.nn.Module):
    """
    PyTorch Gumbel softmax function
    Categorical Reprarameterization with Gumbel-Softmax
    """
    def __init__(self,sample=1, sample_mode=False):
        super(self.__class__, self).__init__()
        self.sample=sample
        self.sample_mode=sample_mode

    def forward(self, inmat, temperature=1.0):
        """
        Forward
        :param input:
        :param hidden:
        :return:
        """
        # input must be log probability
        device = inmat.device
        # torch.cuda.set_device(device)
        if self.sample_mode:
            ishape=list(inmat.shape)
            vshape=list(inmat.shape)
            ishape.insert(2, self.sample) # dw, batch, sample, ...
            vshape.insert(2, self.sample)
            inmat=torch.unsqueeze(inmat, 2).expand(ishape)

        # ui = torch.rand(inmat.shape)
        # ui = ui.to(device)
        if inmat.is_cuda:
            ui = torch.cuda.FloatTensor(inmat.shape).uniform_()
        else:
            ui = torch.FloatTensor(inmat.shape).uniform_()
        gi = -torch.log(-torch.log(ui))
        betax1 = (inmat + gi) / temperature
        self.betax1 = betax1
        betam,_=torch.max(betax1,-1,keepdim=True)
        betax = betax1 - betam
        self.betax=betax
        yi=torch.exp(betax)/torch.sum(torch.exp(betax),dim=-1,keepdim=True)
        return yi

class GSVIB_InfoBottleNeck(torch.nn.Module):
    """
    A GSVIB information bottleneck module
    """
    def __init__(self, model_para ,para=None):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        if para is None:
            para = dict([])
        self.para(para)

        self.prior = torch.nn.Parameter(torch.zeros((self.gs_head_num, self.gs_head_dim)), requires_grad=True)

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=self.sample_size, sample_mode=self.sample_mode)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para=para
        self.freeze_mode = para.get("freeze_mode", False)
        self.temp_scan_num = para.get("temp_scan_num", 1)
        self.sample_mode = para.get("sample_mode", False)
        self.sample_size = para.get("sample_size", 1)
        self.reg_lamda = para.get("reg_lamda", 0.1) # weight on KL
        self.scale_factor = para.get("scale_factor", 1.0) # weight on entropy

    def set_model_para(self,model_para):
        # model_para = {
        #     "gs_head_dim": 2,
        #     "gs_head_num": 64,
        #     "output_size": None
        # }
        self.model_para = model_para
        self.gs_head_dim = model_para["gs_head_dim"]
        self.gs_head_num = model_para["gs_head_num"]
        # self.output_size = model_para.get("output_size",None)

    def forward(self, datax, schedule=1.0):

        assert datax.shape[-1] == self.gs_head_num * self.gs_head_dim

        if schedule < 1.0:  # Multi-scanning trial
            schedule = schedule * self.temp_scan_num
            schedule = schedule - np.floor(schedule)

        if not self.freeze_mode:
            temperature = np.exp(-schedule * 5)
        else:
            temperature = np.exp(-5)

        context = datax.view([*datax.shape[:-1], self.gs_head_num, self.gs_head_dim])
        context = self.softmax(context)
        self.context = context
        self.contextl = context.view([*datax.shape[:-1], self.gs_head_num * self.gs_head_dim])
        # if not self.test_mode:
        gssample = self.gsoftmax(context, temperature=temperature)
        self.gssample = gssample
        # else:
        # maxind = torch.argmax(context,dim=-1)
        # gssample = one_hot(maxind, 2)

        self.loss_reg = self.cal_regloss()

        # if not hasattr(self, "output_size"):
        #     self.output_size=None

        # if self.sample_mode:
        #     if self.output_size is not None:
        #         output = self.i2o(gssample.view([*datax.shape[:-1], self.sample_size,self.gs_head_num* self.gs_head_dim]))
        #         return output
        #     else:
        #         return gssample.view([*datax.shape[:-1], self.sample_size,self.gs_head_num* self.gs_head_dim])
        # else:
        #     if self.output_size is not None:
        #         output = self.i2o(gssample.view([*datax.shape[:-1], self.gs_head_num * self.gs_head_dim]))
        #         return output
        #     else:
        #         return gssample.view([*datax.shape[:-1], self.gs_head_num * self.gs_head_dim])

        if self.sample_mode:
            return gssample.view([*datax.shape[:-1], self.sample_size,self.gs_head_num* self.gs_head_dim])
        else:
            return gssample.view([*datax.shape[:-1], self.gs_head_num * self.gs_head_dim])

    def cal_regloss(self):
        # context,prior should be log probability
        prior = self.softmax(self.prior)
        ent_prior = -torch.mean(torch.sum(torch.exp(prior) * prior, dim=-1))
        prior = prior.view(1, self.gs_head_num, self.gs_head_dim).expand_as(self.context)
        flatkl = torch.mean(torch.sum(torch.exp(self.context) * (self.context - prior), dim=-1))
        loss =  self.reg_lamda * (flatkl + self.scale_factor * ent_prior)
        # print("Loss GSVIB", flatkl, ent_prior)
        return loss

class Softmax_Sample(torch.nn.Module):
    """
    A post-hoc module to do softmax sample
    """
    def __init__(self,model_para={},para={}):
        super(self.__class__, self).__init__()

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=1, sample_mode=False)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para=para

    def set_model_para(self,model_para):
        self.model_para = model_para

    def forward(self, datax, schedule=1.0):
        datax = self.softmax(datax)
        output = self.gsoftmax(datax, temperature=np.exp(-5))
        self.context = datax
        self.gssample = output
        return output

class Gsoftmax_Sample(torch.nn.Module):
    """
    A post-hoc module to do softmax sample
    """
    def __init__(self,model_para,para={}):
        super(self.__class__, self).__init__()

        self.set_model_para(model_para)

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.gsoftmax = Gumbel_Softmax(sample=1, sample_mode=False)

        self.save_para = {
            "model_para": model_para,
            "type": str(self.__class__),
            "misc_para": para,
            "id": id(self)  # to get information about real module sharing
        }

    def para(self,para):
        self.misc_para=para

    def set_model_para(self,model_para):
        self.model_para = model_para
        self.gs_head_dim = model_para["gs_head_dim"]
        self.gs_head_num = model_para["gs_head_num"]

    def forward(self, datax, schedule=1.0):

        context = datax.view([*datax.shape[:-1], self.gs_head_num, self.gs_head_dim])
        context = self.softmax(context)
        self.context = context

        gssample = self.gsoftmax(context, temperature=np.exp(-5))
        self.gssample = gssample

        output = gssample.view([*datax.shape[:-1], self.gs_head_num * self.gs_head_dim])

        return output

class GaussNoise(torch.nn.Module):
    """
    A gaussian noise module
    """
    def __init__(self, std=0.1):

        super(self.__class__, self).__init__()
        self.std = std

    def forward(self, x):
        noise = torch.zeros(x.shape).to(x.device)
        noise.data.normal_(0, std=self.std)
        return x + noise

