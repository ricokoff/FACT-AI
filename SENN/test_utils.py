import os 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import numpy as np
import copy
from api.generator import Generator


def show_model_specs(reg_lambda, h_type, nconcepts):
    reg_format = "{:0.0e}" if reg_lambda != 0 and reg_lambda != 1 else "{}"
    print(f"  conceptizer type        = {h_type}")
    print(f"  number of concepts      = {nconcepts}") if not h_type=="input" else ""
    print( "  sparsity parameter      = 2e-05") if not h_type=="input" else ""
    print(("  regularization strength = "+reg_format).format(reg_lambda))   
    return

def check_input(reg_lambda, h_type, nconcepts):
    allowed = {"reg_lambda":[[0,1e-4,1e-3,1e-2,1e-1,1],"-regularization strength"], 
               "h_type":[["cnn","input"],"-conceptizer type"],
               "nconcepts":[[5,20],"-number of concepts"]}
    if reg_lambda not in allowed["reg_lambda"][0] or h_type not in allowed["h_type"][0]\
            or (nconcepts not in allowed["nconcepts"][0] and h_type != "input"):
        err = "Model not recognized...\nTry any combination of the following:"
        for inarg in allowed.keys():
            strlist = "["+", ".join([str(i) for i in allowed[inarg][0]])+"]"
            err+="\n  {:<10} = {:<33} {}".format(inarg,strlist,allowed[inarg][1])
        raise Exception(err)
    return
    
def load_model(reg_lambda=1e-2, h_type="cnn", nconcepts=5, show_specs=True):
    check_input(reg_lambda, h_type, nconcepts)
    path = "./models/mnist"
    if show_specs:
        print("Loading MNIST model:")
        show_model_specs(reg_lambda, h_type, nconcepts)
    if h_type == "cnn":
        #learning rate, sparsity and grad penalty method fixed for each model
        model_path = "grad3_Hcnn_Thsimple_Cpts{}_Reg{:0.0e}_Sp2e-05_LR0.0002".format(nconcepts, reg_lambda)
    elif h_type == "input":
        model_path = "grad3_Hinput_Thsimple_Reg{:0.0e}_LR0.0002".format(reg_lambda)
    #TO DO: adjust model_path so it works from your working directory
    model_path = os.path.join(path, model_path)
    checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'),map_location=lambda storage,loc:storage)     
    return checkpoint["model"]

def show_digit(digit, title, imsize=1.5):
    digit = [digit] if not isinstance(digit,list) else digit
    title = [title] if not isinstance(title,list) else title     
    plt.figure(figsize=(imsize*len(digit),imsize))
    for i, dgt in enumerate(digit):
        plt.subplot(1,len(digit),i+1)
        if len(title[i]) != 0:
            plt.title(title[i])
        plt.imshow(dgt) if dgt.sum()!=0 else ""
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return

def get_digit(dataset, indx):
    d,t = dataset.__getitem__(indx) 
    return d.view(28,28), t

def plot_activations(activations):
    plt.figure(figsize=(0.5*len(activations),3))
    plt.title("ENCODED (CONCEPT ACTIVATIONS)")
    plt.bar(np.arange(len(activations)),activations)
    plt.xticks(np.arange(len(activations)), [f"C{i+1}" for i in range(len(activations))])
    plt.show()
    return

def visualize_reconstruction(model, dataset, indx):
    digit, _ = get_digit(dataset, indx)
    encoded = model.conceptizer.encode(digit.view(1,1,28,28))
    decoded = model.conceptizer.decode(encoded).squeeze()
    show_digit([digit,decoded],["INPUT","RECONSTRUCTED"]*2,imsize=2,)
    plot_activations(encoded.squeeze())
    return


#find lowest (find="low") or highest (find="high") value in dict
def dic_find(dic, find):
    best = 1e9 if find == "low" else -1e9
    for key in dic:
        if (dic[key] < best and find=="low") or (dic[key] > best and find=="high"):
            bestkey = key
            best = dic[key]
    return bestkey

def find_prototypes(model, dataset, num_samples, prototypes, print_freq):
    for i in range(len(dataset)):
        if print_freq!=0 and i%print_freq==0:
            print(f"{i}/{len(dataset)}")
        digit, _ = get_digit(dataset, i)
        encoded = model.conceptizer.encode(digit.view(1,1,28,28)).squeeze()

        for cpt in prototypes.keys():
            lowkey_of_high = dic_find(prototypes[cpt]["high"],"low")
            highkey_of_low = dic_find(prototypes[cpt]["low"],"high")

            if encoded[cpt] > prototypes[cpt]["high"][lowkey_of_high]:
                if len(prototypes[cpt]["high"]) == num_samples:
                     prototypes[cpt]["high"].pop(lowkey_of_high)
                prototypes[cpt]["high"][i] = encoded[cpt]

            if encoded[cpt] < prototypes[cpt]["low"][highkey_of_low]:
                if len(prototypes[cpt]["low"]) == num_samples:
                     prototypes[cpt]["low"].pop(highkey_of_low)
                prototypes[cpt]["low"][i] = encoded[cpt]
    print(f"{len(dataset)}/{len(dataset)}") if print_freq!=0 else ""
 
    for cpt in prototypes.keys():
        for extreme in prototypes[cpt].keys():
            for indx in dict(prototypes[cpt][extreme]).keys():
                prototypes[cpt][extreme][get_digit(dataset,indx)[0]] = prototypes[cpt][extreme][indx]
                prototypes[cpt][extreme].pop(indx)
    return prototypes

def generate_prototypes(model, num_samples, prototypes, print_freq, Nsteps, lr=0.1, p1=1, p2=1):
    for param in model.parameters():
        param.requires_grad = False
        
    losses = copy.deepcopy(prototypes)
        
    generator = Generator(model.conceptizer)
    
    for cpt in prototypes.keys():
        print(f"{cpt+1}/{len(prototypes.keys())}") if print_freq!=0 else ""
        for extreme in prototypes[cpt].keys():
            sign = -1. if extreme=="high" else 1.
            prototypes[cpt][extreme].pop(-1)
            losses[cpt][extreme].pop(-1)
            for sample in range(num_samples):
                sample_loss = []
                generator.initialize() #reset generator
                optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
                for step in range(Nsteps):
                    optimizer.zero_grad()
                    activations = generator.forward()
                    loss = criterion(activations, cpt, generator.generated, sign, p1=p1, p2=p2)
                    loss.backward()
                    optimizer.step()
                    sample_loss.append(loss)
                generated = torch.tensor(generator.generated.detach())
                prototypes[cpt][extreme][generated] = activations[cpt]
                losses[cpt][extreme][sample] = sample_loss
    return prototypes, losses

#prototypes: {0:{"high":{i1:_, i2:_, ...}, "low":{i1:_, i2:_, ...}}, 1:{...}}
def empty_prototypes(model, dummies=False):  
    nchannels = model.conceptizer.conv2.out_channels
    if dummies:
        return {i:{"high":{-1:-1e9},"low":{-1:1e9}} for i in range(nchannels)}
    else:
        return {i:{"high":{},"low":{}} for i in range(nchannels)}
    

def get_prototypes(model, num_samples, dataset=None, print_freq=2500, Nsteps=150, lr=0.1, p1=1, p2=1):
    prototypes = empty_prototypes(model, dummies=True)
    if dataset != None:
        print("Finding prototypes according to the method used in the paper:") if print_freq!=0 else ""
        prototypes = find_prototypes(model, dataset, num_samples, prototypes, print_freq) 
        losses = "" #not applicable for this method
    else:
        print("Generating synthetic images that maximize the activations:") if print_freq!=0 else ""
        prototypes, losses = generate_prototypes(model, num_samples, prototypes, print_freq, Nsteps, lr=lr, p1=p1, p2=p2)
    return prototypes, losses

def visualize_cpts(prototypes, imsize=2):
    for cpt in prototypes.keys():       
        to_show = []
        titles = []
        for extreme in prototypes[cpt].keys(): 
            titles.append(f"{extreme} activation C{cpt+1}")
            for dgt in prototypes[cpt][extreme].keys():
                to_show.append(dgt)
                titles.append("")
            to_show.append(np.zeros((28,28)))
        show_digit(to_show[:-1], titles[:-1],imsize=imsize)
    return

#for p(x)~N(0,1), see:
#https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
def KL(q):
    return -torch.log(torch.std(q)) + (torch.std(q)**2 + torch.mean(q)**2)/2 - 0.5
  
def criterion(activations, cpt, generated, sign, p1=1, p2=1):
    ai = activations[cpt]
    sum_aj = torch.sum(activations) - ai
    return sign*ai - p1*sign*sum_aj + p2*KL(generated)

#given two sets of prototypes, find best ones of each
def get_best_prototypes(model, prototypes_A, prototypes_B):
    prototypes_best = empty_prototypes(model)
    for cpt in prototypes_A.keys():
        for extreme in prototypes_A[cpt].keys():
            best_A = dic_find(prototypes_A[cpt][extreme], extreme)
            best_B = dic_find(prototypes_B[cpt][extreme], extreme)
            prototypes_best[cpt][extreme][best_A] = prototypes_A[cpt][extreme][best_A]
            prototypes_best[cpt][extreme][best_B] = prototypes_B[cpt][extreme][best_B]
    return prototypes_best
  
#prototypes_best needs two have exaclty two elements of each extreme and cpt
def visualize_best(prototypes_best, show_activations=True):
    for cpt in prototypes_best.keys():
        to_show = []
        titles = []
        acts = []
        for extreme in prototypes_best[cpt].keys():
            titles.append(f"{extreme} activation C{cpt+1}")
            for dgt in prototypes_best[cpt][extreme].keys():
                to_show.append(dgt)
                titles.append("")
                acts.append(prototypes_best[cpt][extreme][dgt])
            to_show.append(np.zeros((28,28)))
        show_digit(to_show[:-1], titles[:-1])
        if show_activations:
            print("{:^15.3}{:^11.3}{:>23.3}{:^20.3}".format(acts[0],acts[1],acts[2],acts[3]))
            
        
        
        
        