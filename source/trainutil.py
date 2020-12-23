"""
Python package for NCA learning algorithm
Author: Harry He @ NCA Lab, CBS, RIKEN
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import datetime, time

class PyTrain_Main(object):
    """
    A main class to run training
    """
    def __init__(self, model, data_dict, device="cuda:0", para=None):
        self.model = model.to(device)
        self.data_dict = data_dict
        self.device = device

        if para is None:
            para = dict([])
        self.para(para)

        currentDT = datetime.datetime.now()
        self.log = "Time of creation: " + str(currentDT) + "\n"
        self.train_hist = []
        self.evalmem = None

    def para(self, para):
        self.loss_exp_flag = para.get("loss_exp_flag", False)
        self.figure_plot = para.get("figure_plot", True)
        self.loss_clip = para.get("loss_clip", 50.0)

    def run_training(self, epoch=2, lr=1e-3, optimizer_label="adam", print_step=200):

        self.model.train()

        if optimizer_label == "adam":
            print("Using adam optimizer")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        elif optimizer_label == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        else:
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        startt = time.time()
        pt_model = self.model

        for ii_epoch in range(epoch):
            print("Starting epoch %s." % str(ii_epoch))
            self.model.loss_mode = "train"
            self.model.train()

            for iis,(datax, labels) in enumerate(self.data_dict["train"]):

                step_per_epoch=len(self.data_dict["train"])
                ii_tot = iis + ii_epoch * step_per_epoch
                cstep = ii_tot / (epoch * step_per_epoch)
                try:
                    datax = datax.to(self.device)
                except:
                    datax = [item.to(self.device) for item in datax]
                loss = pt_model(datax, labels.to(self.device), schedule=cstep)
                self._profiler(ii_tot, loss, print_step=print_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            midt = time.time()
            print("Time used till now:", midt - startt)
            print("Validation of epoch ", ii_epoch, ":")
            self.do_eval()

        endt = time.time()
        print("Time used in training:", endt - startt)
        self.log = self.log + "Time used in training: " + str(endt - startt) + "\n"
        self._postscript()

    def do_eval(self, data_pt = "val", eval_mem_flag=False):
        self.model.eval()
        self.model.loss_mode="eval"
        # Validation
        lossl = []
        print("Start evaluation ...", len(self.data_dict[data_pt]))
        startt = time.time()
        for iis, (datax, labels) in tqdm(enumerate(self.data_dict[data_pt])):
            with torch.no_grad():
                try:
                    datax = datax.to(self.device)
                except:
                    datax = [item.to(self.device) for item in datax]
                loss = self.model(datax, labels.to(self.device), schedule=1.0)
                lossl.append(loss.item())
                if eval_mem_flag:
                    self.eval_mem(datax, labels, self.model)
        if self.loss_exp_flag:
            print("Evaluation Perplexity: ", np.exp(np.mean(np.array(lossl))))
        else:
            print("Evaluation Perplexity: ", np.mean(np.array(lossl)))
        endt = time.time()
        print("Time used in evaluation: ", endt - startt)

    def do_test(self, data_pt = "val"):
        """
        Calculate correct rate
        :param step_test:
        :return:
        """
        print("Start testing ...")
        self.model.eval()
        total = 0
        correct = 0
        for iis, (datax, labels) in tqdm(enumerate(self.data_dict[data_pt])):
            with torch.no_grad():
                try:
                    datax = datax.to(self.device)
                except:
                    datax = [item.to(self.device) for item in datax]
                loss = self.model(datax, labels.to(self.device), schedule=1.0)
                output = self.model.output
                _, predicted = torch.max(output, -1)
                predicted = predicted.cpu()
                if len(predicted.shape)==len(labels.shape):
                    total += np.prod(labels.shape)
                    correct += (predicted == labels).sum().item()
                elif len(labels)==1:
                    labels = labels.view(labels.shape[0],1).expand(labels.shape[0],predicted.shape[1])
                    total += labels.shape[0]*labels.shape[1]
                    correct += (predicted == labels).sum().item()
                else:
                    raise Exception("Labels unknown shape")
        crate = correct / total
        print("Correct rate: ", correct / total)

        return crate

    def eval_mem(self, datax, labels, model):
        if self.evalmem is None:
            # self.evalmem = []
            self.evalmem = [[] for ii in range(12)]  # x,label,context
        if self.mem_eval_mode == "task2":
            self.evalmem[0].append(model.model.seq1_coop.output.detach().cpu().numpy())
            self.evalmem[1].append(model.model.seq1_coop.contextl.detach().cpu().numpy())
            self.evalmem[2].append(datax[1].cpu().numpy())

    def _profiler(self, iis, loss, print_step=200):

        if iis % print_step == 0:
            if self.loss_exp_flag:
                print("Perlexity: ", iis, np.exp(loss.item()))
                self.log = self.log + "Perlexity: " + str(iis)+" "+ str(np.exp(loss.item())) + "\n"
            else:
                print("Loss: ", iis, loss.item())
                self.log = self.log + "Loss: " + str(iis) + " " + str(loss.item()) + "\n"

        if self.loss_exp_flag:
            self.train_hist.append(np.exp(loss.item()))
        else:
            self.train_hist.append(loss.item())

    def _postscript(self):
        x = []
        for ii in range(len(self.train_hist)):
            x.append([ii, self.train_hist[ii]])
        x = np.array(x)
        if self.figure_plot:
            try:
                plt.plot(x[:, 0], x[:, 1])
                if self.loss_clip > 0:
                    # plt.ylim((-self.loss_clip, self.loss_clip))
                    if self.loss_exp_flag:
                        low_b=1.0
                    else:
                        low_b=0.0
                    plt.ylim((low_b, self.loss_clip))
                if self.save_fig is not None:
                    filename=self.save_fig+str(self.cuda_device)+".png"
                    print(filename)
                    plt.savefig(filename)
                    self.log = self.log + "Figure saved: " + filename + "\n"
                    plt.gcf().clear()
                else:
                    plt.show()
            except:
                pass
