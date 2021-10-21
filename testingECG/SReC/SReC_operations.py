
import os
def CommandLineCompressionWithWithSReC(pathToInput="/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",
                                       PathToInputNamesTxt="/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt",
                                       PathToOutputFolder="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression",
                                       PathToModelpth="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"):

    import subprocess
    originalCwd=os.getcwd()   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    command ="python -um src.encode --path "+pathToInput+" --file "+PathToInputNamesTxt +" --save-path "+ PathToOutputFolder+" --load "+PathToModelpth
    
    command ="python3 -um src.encode \--path \"/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r\" \--file \"/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt\" \--save-path \"/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression\" \--load \"/home/yonathan/Documents/GitHub/SReC/models/openimages.pth\""
    command ="python3 -um src.encode \--path \"/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r\" \--file \"/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val_Comp_4_images.txt\" \--save-path \"/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression\" \--load \"/home/yonathan/Documents/GitHub/SReC/models/openimages.pth\""

    subprocess.Popen(command,shell=True)
    
    os.chdir(os.path.dirname(originalCwd))

def CommandLineDeCompressionWithWithSReC(pathToInput="/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",
                                       PathToInputNamesTxt="/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt",
                                       PathToOutputFolder="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression",
                                       PathToModelpth="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"):

    import subprocess
    originalCwd=os.getcwd()   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #
    
    command ="python3 -um src.decode \--path \"/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression\" \--file \"/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val (copy).txt\" \--save-path \"/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/decompression\" \--load \"/home/yonathan/Documents/GitHub/SReC/models/openimages.pth\""
    command ="python3 -um src.decode \--path \"/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression\" \--file \"/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val_DeComp_4_images.txt\" \--save-path \"/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/decompression\" \--load \"/home/yonathan/Documents/GitHub/SReC/models/openimages.pth\""

    subprocess.Popen(command,shell=True)
    
    os.chdir(os.path.dirname(originalCwd))
 
 
def CommandLineEvalWithWithSReC():

    import subprocess
    originalCwd=os.getcwd()   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
     
    command ="python3 -um src.eval \--path \"/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r\" \--file \"/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt\" \--load \"/home/yonathan/Documents/GitHub/SReC/models/openimages.pth\""

    subprocess.Popen(command,shell=True)
    
    os.chdir(os.path.dirname(originalCwd))   
def comThenDecompCommandLine():
    from time import sleep
    CommandLineCompressionWithWithSReC()
    sleep(20)
    CommandLineDeCompressionWithWithSReC()
     
     
#-------SRec ops-----


def compressWithSRecMain(connectionToFather, path, save_path ,load,
                         resblocks=3,n_feats = 64, scale = 3,k=10,crop =0,log_likelihood =False,decode =False,
                         suffix=".srec"):
    import os
    import sys

    import click
    import numpy as np
    import torch
    import torchvision.transforms as T
    from torch.utils import data

    from src import configs
    from src import data as lc_data
    from src import network
    from src.l3c import bitcoding, timer

    configs.n_feats = n_feats
    configs.resblocks = resblocks
    configs.K = k
    configs.scale = scale
    configs.log_likelihood = log_likelihood
    configs.collect_probs = True

    print(sys.argv)

    checkpoint = torch.load(load)
    print(f"Loaded model from {load}.")
    print("Epoch:", checkpoint["epoch"])

    compressor = network.Compressor()
    compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()

    print(compressor.nets)

    transforms = []  # type: ignore
    if crop > 0:
        transforms.insert(0, T.CenterCrop(crop))

        
    dataset = lc_data.MyImageFolder(
        filenamesPaths=path,
        scale=scale,transforms= T.Compose(transforms)
    )
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, drop_last=False,
    )
    print(f"Loaded directory with {len(dataset)} images")

    os.makedirs(save_path, exist_ok=True)

    coder = bitcoding.Bitcoding(compressor)
    encoder_time_accumulator = timer.TimeAccumulator()
    decoder_time_accumulator = timer.TimeAccumulator()
    total_file_bytes = 0
    total_num_subpixels = 0
    total_entropy_coding_bytes: np.ndarray = 0  # type: ignore
    total_log_likelihood_bits = network.Bits()

    for i, (filenames, x) in enumerate(loader):
        assert len(filenames) == 1, filenames
        filename = filenames[0]
        file_id = filename.split(".")[0]
        filepath = os.path.join(save_path, f"{file_id}{suffix}")

        with encoder_time_accumulator.execute():
            log_likelihood_bits, entropy_coding_bytes = coder.encode(
                x, filepath)

        total_file_bytes += os.path.getsize(filepath)
        total_entropy_coding_bytes += np.array(entropy_coding_bytes)
        total_num_subpixels += np.prod(x.size())
        if configs.log_likelihood:
            total_log_likelihood_bits.add_bits(log_likelihood_bits)

        if decode:
            with decoder_time_accumulator.execute():
                y = coder.decode(filepath)
                y = y.cpu()
            assert torch.all(x == y), (x[x != y], y[x != y])

        if configs.log_likelihood:
            theoretical_bpsp = total_log_likelihood_bits.get_total_bpsp(
                total_num_subpixels).item()
            print(
                f"Theoretical Bpsp: {theoretical_bpsp:.3f};\t", end="")
        print(
            f"Bpsp: {total_file_bytes*8/total_num_subpixels:.3f};\t"
            f"Images: {i + 1};\t"
            f"Comp: {encoder_time_accumulator.mean_time_spent():.3f};\t",
            end="")
        if decode:
            print(
                "Decomp: "
                f"{decoder_time_accumulator.mean_time_spent():.3f}",
                end="")
        print(end="\r")
        
        if(connectionToFather.poll()):
            try:
                messege=connectionToFather.recv()
                if(messege=="canceled"):
                    connectionToFather.close()
                    return
            except EOFError:
                print("wrong message")
                pass
    print()

    if decode:
        print("Decomp Time By Scale: ", end="")
        print(", ".join(
            f"{scale_time:.3f}"
            for scale_time in coder.decomp_scale_times()))
    else:
        print("Scale Bpsps: ", end="")
        print(", ".join(
            f"{scale_bpsp:.3f}"
            for scale_bpsp in total_entropy_coding_bytes*8/total_num_subpixels))
        
        
    connectionToFather.send("Done")



def deCompressWithSReCMain(connectionToFather, paths, save_path ,load,
                         resblocks=3,n_feats = 64, scale = 3,k=10,suffix=".srec"):
    import os
    import sys
    import ntpath

    import click
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import ImageFile

    from src import configs, network
    from src.l3c import bitcoding, timer
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    configs.n_feats = n_feats
    configs.resblocks = resblocks
    configs.K = k
    configs.scale = scale
    configs.collect_probs = False

    print(sys.argv)

    checkpoint = torch.load(load)
    print(f"Loaded model from {load}.")
    print("Epoch:", checkpoint["epoch"])

    compressor = network.Compressor()
    compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()
    print(compressor.nets)

    print(f"Loaded directory with {len(paths)} images")

    os.makedirs(save_path, exist_ok=True)

    coder = bitcoding.Bitcoding(compressor)
    decoder_time_accumulator = timer.TimeAccumulator()
    total_num_bytes = 0
    total_num_subpixels = 0

    for filepath in paths:
        assert filepath.endswith(suffix), (
            f"{filepath} is not a {suffix} file")
        with decoder_time_accumulator.execute():
            x = coder.decode(filepath)
            x = x.byte().squeeze(0).cpu()
        img = T.functional.to_pil_image(x)
        
        filename=os.path.splitext(ntpath.basename(filepath))[0]
        
        img.save(os.path.join(save_path, f"{filename}.png"))
        print(
            "Decomp: "
            f"{decoder_time_accumulator.mean_time_spent():.3f};\t"
            "Decomp Time By Scale: ",
            end="")
        decomp_scale_times = coder.decomp_scale_times()
        print(
            ", ".join(f"{scale_time:.3f}" for scale_time in decomp_scale_times),
            end="; ")

        total_num_bytes += os.path.getsize(filepath)
        total_num_subpixels += np.prod(x.size())

        print(
            f"Bpsp: {total_num_bytes*8/total_num_subpixels:.3f}", end="\r")
    print()
    connectionToFather.send("Done")

def evalWithSReCMain(connectionToFather,path,load, workers =2, resblocks =3, n_feats=64,
        scale =3, k=10, crop =0,):
    """
    file  <_io.TextIOWrapper name='/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt' mode='r' encoding='UTF-8'>
    workers  2
    resblocks  3
    n_feats  64
    scale  3
    load  /home/yonathan/Documents/GitHub/SReC/models/openimages.pth
    k  10
    crop  0
    """
    from src.eval import run_eval, count_params
    import sys
    from typing import Tuple

    import click
    import numpy as np
    import torch
    import torchvision.transforms as T
    from torch import nn
    from torch.utils import data

    from src import configs
    from src import data as lc_data
    from src import network
    from src.l3c import timer
    
    configs.n_feats = n_feats
    configs.resblocks = resblocks
    configs.K = k
    configs.scale = scale

    print(sys.argv)
    print([item for item in dir(configs) if not item.startswith("__")])

    if load != "/dev/null":
        checkpoint = torch.load(load)
        print(f"Loaded model from {load}.")
        print("Epoch:", checkpoint["epoch"])
    else:
        checkpoint = {}

    compressor = network.Compressor()
    if checkpoint:
        compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()

    print(f"Number of parameters: {count_params(compressor.nets)}")
    print(compressor.nets)

    transforms = []  # type: ignore
    if crop > 0:
        transforms.insert(0, T.CenterCrop(crop))

         
    dataset = lc_data.MyImageFolder(
        filenamesPaths=path,
        scale=scale,transforms= T.Compose(transforms)
    )
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=workers, drop_last=False,
    )
    print(f"Loaded dataset with {len(dataset)} images")

    bits, inp_size = run_eval(loader, compressor)
    for key in bits.get_keys():
        print(f"{key}:", bits.get_scaled_bpsp(key, inp_size))
        
    bpsp = bits.get_total_bpsp(inp_size)
    
    connectionToFather.send(bpsp.item())
  
def timeTest_DeAndCompress(connectionToFather, path ,load,
                         resblocks=3,n_feats = 64, scale = 3,k=10,crop =0,log_likelihood =False):
    import tempfile, time
    save_path = tempfile.TemporaryDirectory(dir = os.path.dirname(os.path.realpath(__file__)))
    start=time.time()
    compressWithSRecMain(connectionToFather, path, save_path.name ,load,
                         resblocks,n_feats , scale,k,crop ,log_likelihood =False,decode =True)  
    end = time.time()
    print(end-start)
    connectionToFather.send(end-start)

    """
    the sys ARgs:
    
    ['/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/src/train.py', '--train-path', '/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r', '--train-file', '/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt', '--eval-path', '/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r', '--eval-file', '/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt', '--plot', '/home/yonathan/Documents/GitHub/SReC/models/MYmodel', '--batch', '20', '--epochs', '10', '--resblocks', '3']
    
    specifics when  training new 
    train_path  /home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r
    eval_path  /home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r
    train_file  <_io.TextIOWrapper name='/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt' mode='r' encoding='UTF-8'>
    eval_file  <_io.TextIOWrapper name='/home/yonathan/Documents/GitHub/SReC/datasets/open_images_val.txt' mode='r' encoding='UTF-8'>
    batch  20
    plot  /home/yonathan/Documents/GitHub/SReC/models/MYmodel
    load  /dev/null
    
    when trainingOld

    """
def TrainSReC( conn,train_path: str, eval_path: str, train_list, eval_list,
        batch: int,  plot: str,load: str,workers=1, epochs=50,
        resblocks=3, n_feats=64, scale=3,
        lr=1e-4, eval_iters=0, lr_epochs =1,
        plot_iters=1000, k=10, clip=0.5,
        crop=128, gd="adam") :
    
    import os
    import sys
    from typing import List

    import click
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import ImageFile
    from torch import nn, optim
    from torch.utils import data, tensorboard

    from src import configs
    from src import data as lc_data
    from src import network
    from src.l3c import timer
    
    from src.train import plot_bpsp,train_loop,run_eval,save
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    configs.n_feats = n_feats
    configs.scale = scale
    configs.resblocks = resblocks
    configs.K = k
    configs.plot = plot

    print(sys.argv)

    #os.makedirs(plot, exist_ok=True)
    #model_load = os.path.join(plot, "train.pth")
    #if os.path.isfile(model_load):
    #   load = model_load
    if os.path.isfile(load) and load != "/dev/null":
        checkpoint = torch.load(load)
        print(f"Loaded model from {load}.")
        print("Epoch:", checkpoint["epoch"])
        if checkpoint.get("best_bpsp") is None:
            print("Warning: best_bpsp not found!")
        else:
            configs.best_bpsp = checkpoint["best_bpsp"]
            print("Best bpsp:", configs.best_bpsp)
    else:
        checkpoint = {}

    compressor = network.Compressor()
    if checkpoint:
        compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()

    optimizer: optim.Optimizer  # type: ignore
    if gd == "adam":
        optimizer = optim.Adam(compressor.parameters(), lr=lr, weight_decay=0)
    elif gd == "sgd":
        optimizer = optim.SGD(compressor.parameters(), lr=lr,
                              momentum=0.9, nesterov=True)
    elif gd == "rmsprop":
        optimizer = optim.RMSprop(  # type: ignore
            compressor.parameters(), lr=lr)
    else:
        raise NotImplementedError(gd)

    starting_epoch = checkpoint.get("epoch") or 0

    print(compressor)


    #  train_dataset = lc_data.ImageFolder(
    #     train_path,
    #     [filename.strip() for filename in train_file],
    #     scale,
    #     T.Compose([
    #         T.RandomHorizontalFlip(),
    #         T.RandomCrop(crop),
    #     ]),
    # )
    train_dataset = lc_data.MySignalFolder(
        filenamesPaths=train_list,
        scale=scale,
    )
    dataset_index = checkpoint.get("index") or 0
    train_sampler = lc_data.PreemptiveRandomSampler(
        checkpoint.get("sampler_indices") or torch.randperm(
            len(train_dataset)).tolist(),
        dataset_index,
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch, sampler=train_sampler,
        num_workers=workers, drop_last=True,
    )
    print(f"Loaded training dataset with {len(train_loader)} batches "
          f"and {len(train_loader.dataset)} images")
    # eval_dataset = lc_data.ImageFolder(
    #     eval_path, [filename.strip() for filename in eval_file],
    #     scale,
    #     T.Lambda(lambda x: x),
    # )
        
    eval_dataset = lc_data.MySignalFolder(
        
         filenamesPaths=eval_list,
        scale=scale  
    )
    eval_loader = data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=workers, drop_last=False,
    )
    print(f"Loaded eval dataset with {len(eval_loader)} batches "
          f"and {len(eval_dataset)} images")

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, lr_epochs, gamma=0.75)

    for _ in range(starting_epoch):
        lr_scheduler.step()  # type: ignore

    train_iter = checkpoint.get("train_iter") or 0
    if eval_iters == 0:
        eval_iters = len(train_loader)

    for epoch in range(starting_epoch, epochs):
        with tensorboard.SummaryWriter(plot) as plotter:
            # input: List[Tensor], downsampled images.
            # sizes: N scale 4
            for _, inputs in train_loader:
                train_iter += 1
                batch_size = inputs[0].shape[0]

                train_loop(inputs, compressor, optimizer, train_iter,
                           plotter, plot_iters, clip)
                # Increment dataset_index before checkpointing because
                # dataset_index is starting index of index of the FIRST
                # unseen piece of data.
                dataset_index += batch_size

                if train_iter % plot_iters == 0:
                    plotter.add_scalar(
                        "train/lr",
                        lr_scheduler.get_lr()[0],  # type: ignore
                        train_iter)
                    save(compressor, train_sampler.indices, dataset_index,
                         epoch, train_iter, plot, "train.pth")

                if train_iter % eval_iters == 0:
                    run_eval(
                        eval_loader, compressor, train_iter,
                        plotter, epoch)

            lr_scheduler.step()  # type: ignore
            dataset_index = 0

    with tensorboard.SummaryWriter(plot) as plotter:
        run_eval(eval_loader, compressor, train_iter,
                 plotter, epochs)
    save(compressor, train_sampler.indices, train_sampler.index,
         epochs, train_iter, plot, "train.pth")
    print("training done")
    pass



def getImageNamesFromFolder(path):
    fileNames=os.listdir(path)
    images=[]
    for image in fileNames:
        full_image_path=os.path.join(path, image)
        _, file_extension = os.path.splitext(full_image_path)
        if os.path.isfile(full_image_path) and file_extension ==".png":
            images.append(image)
    return images


def prepFolders(trainFolder,validFolder):
   
    trainList=getImageNamesFromFolder(trainFolder) 
    validList=getImageNamesFromFolder(validFolder)
    retDict={}
    retDict["tempFolder"]=os.path.join(os.path.dirname(os.path.abspath(__file__)),"training_temp")
    if (os.path.exists(retDict["tempFolder"]) and os.path.isdir(retDict["tempFolder"])):
        import shutil
        shutil.rmtree(retDict["tempFolder"])
    os.mkdir( retDict["tempFolder"])
    retDict["trainFile"]=os.path.join( retDict["tempFolder"],"train_image_names.txt")
    retDict["validFile"]=os.path.join( retDict["tempFolder"],"valid_image_names.txt")
    with open(retDict["trainFile"], "wt") as fout:  
        for image in trainList:
            fout.write(image+"\n")
            
    with open(retDict["validFile"], "wt") as fout:  
        for image in validList:
            fout.write(image+"\n")
            
    return retDict

        
def TrainSReC_casing(conn,
        train_path: str, eval_path: str,
        batch: int,load="",newModel=True,
        
        workers=1, epochs=2,resblocks=3, n_feats=64, scale=3,
        lr=1e-4, eval_iters=0, lr_epochs =1,
        plot_iters=1000, k=10, clip=0.5,
        crop=128, gd="adam"):
    if(newModel and load=="" ):
         load="/dev/null"
    modelFolder=os.path.join(os.path.dirname(os.path.abspath(__file__)),"models")
    modelFolder=os.path.join(modelFolder,"modelFolder_temp")
    
    if (os.path.exists(modelFolder) and os.path.isdir(modelFolder)):
            import shutil
            shutil.rmtree(modelFolder)
            
            
    os.mkdir(modelFolder)        
    

    TrainSReC( conn=conn,train_path=None, eval_path=None, train_list=train_path, eval_list=eval_path,
                batch=batch,  plot =modelFolder,load=load,workers=workers, epochs=epochs,
                resblocks=resblocks, n_feats=n_feats, scale=scale,
                lr=lr, eval_iters=eval_iters, lr_epochs =lr_epochs,
                plot_iters=plot_iters, k=k, clip=clip,
                crop=crop, gd=gd)  
            
    import shutil
    from datetime import datetime

    def moveModel(name,rename=""):
        now = datetime.now()
        if(rename==""):
            rename=((now.strftime("%b-%d-%Y %H:%M:%S")+".pth").replace(" ", "_"))
        shutil.copy(os.path.join(modelFolder,name), os.path.join(os.path.dirname(os.path.abspath(__file__)),"models"))    
        dirname=os.path.join(os.path.dirname(os.path.abspath(__file__)),"models")
        fileName=os.path.join(dirname,name)
        os.rename(fileName,os.path.join(dirname,rename))      
        import copy
        conn.send([copy.deepcopy(rename),copy.deepcopy(os.path.join(dirname,rename))])
            
    if(os.path.exists(os.path.join(modelFolder,"best.pth"))):
        if(newModel):
            moveModel("best.pth")
        else:
            moveModel("best.pth",os.path.basename(load))
    elif(os.path.exists(os.path.join(modelFolder,"train.pth"))):
        if(newModel):
            moveModel("train.pth")
        else:
            moveModel("train.pth",os.path.basename(load))
    
    shutil.rmtree(modelFolder)  
    
#----------------testing------------------------
def testingCompressWithSReCMain():
    from multiprocessing import Pipe
    conn1,conn2=Pipe()
    
    
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b0328075ddf03fc.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0bdcd1cc8c08d4e0.png"
          ]
    #path= os.listdir("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r")
    #for i in range(0,len(path)):
     #   path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",path[i])
        
    save_path="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression"
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    #load ="/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/models/May-30-2021 00:01:29.pth"
    #load ="/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/models/best.pth"

    compressWithSRecMain(conn2,path, save_path ,load,crop=128 )

    
    
def testingDeCompressWithSReCMain():
    from multiprocessing import Pipe
    conn1,conn2=Pipe()
    path=["/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0a7f13330a5d0023.srec","/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0b6ef374714ee86b.srec",
          "/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0b0328075ddf03fc.srec","/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0bdcd1cc8c08d4e0.srec"]
    path= os.listdir("/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression")
    for i in range(0,len(path)):
        path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/",path[i])

 
              
    save_path="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/decompression"
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    #load ="/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/models/May-30-2021 00:01:29.pth"
    #load ="/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/models/best.pth"

    deCompressWithSReCMain(conn2,path, save_path ,load)
    
def testingEvalWithSReCMain():
    from multiprocessing import Pipe
    conn1,conn2=Pipe()
    
    
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b0328075ddf03fc.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0bdcd1cc8c08d4e0.png"
          ]
    path= os.listdir("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r")
    for i in range(0,len(path)):
        path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",path[i])
        
    save_path="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression"
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    evalWithSReCMain(conn2,path=path,load=load) 
    print(conn1.recv())
    
    
def testingTimeTest():
    from multiprocessing import Pipe
    conn1,conn2=Pipe()
    
    
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b0328075ddf03fc.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0bdcd1cc8c08d4e0.png"
          ]
    #path= os.listdir("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r")
    #for i in range(0,len(path)):
     #   path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",path[i])
        
    save_path="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression"
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    timeTest_DeAndCompress(conn2,path,load )
    
def testTraining():
    from os import listdir
    from os.path import isfile, join
    mypath="/home/yonathan/Documents/GitHub/DataSets/WFDB"
    matFilePaths = [join(mypath, f) for f in listdir(mypath) if (isfile(join(mypath, f)) and  os.path.splitext(f)[1]==".mat" )]
    eval_path = matFilePaths[:len(matFilePaths)-100]    
    train_path=  matFilePaths
    batch =20

    load ="/dev/null"
    TrainSReC_casing(conn=None,train_path=train_path, eval_path=eval_path,batch=batch,load=load,newModel=True)
    
def testingECG():
    from multiprocessing import Pipe
    conn1,conn2=Pipe()
    path=["/home/yonathan/Documents/git/filename.png"]
     
    save_path="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/decompression"
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    evalWithSReCMain(conn2, path ,load)
    
def testing():
    import sys
    #testingCompressWithSReCMain() 
    #testingDeCompressWithSReCMain() 
    #testingEvalWithSReCMain()
    #testingTimeTest()
    testTraining()
    #testingECG()
 
#-------------main----------------

def _compressWithSReCMain(conn,argDict):
    compressWithSRecMain(connectionToFather=conn,path =argDict["path"], save_path=argDict["save_path"] ,load= argDict["load"],resblocks=argDict["resblocks"],n_feats = argDict["n_feats"], scale = argDict["scale"]
                         ,k=argDict["k"],crop =argDict["crop"],log_likelihood =argDict["log_likelihood"],decode =argDict["decode"],suffix=argDict["suffix"])
    
def _deCompressWithSReCMain(conn,argDict):
    deCompressWithSReCMain(connectionToFather=conn,paths =argDict["path"], save_path=argDict["save_path"] ,load= argDict["load"],
                           resblocks=argDict["resblocks"],n_feats = argDict["n_feats"], scale = argDict["scale"],suffix=argDict["suffix"],k=argDict["k"])
    
def _evalWithSReCMain(conn ,argDict):
    evalWithSReCMain(conn,path=argDict["path"],load=argDict["load"],workers=argDict["workers"],resblocks=argDict["resblocks"],
                     n_feats=argDict["n_feats"],scale=argDict["scale"],k=argDict["k"],crop=argDict["crop"])
    
def _timeTest_DeAndCompress(conn,argDict):
    timeTest_DeAndCompress(connectionToFather=conn,path =argDict["path"],load= argDict["load"],resblocks=argDict["resblocks"],n_feats = argDict["n_feats"],
                           scale = argDict["scale"],k=argDict["k"],crop =argDict["crop"],log_likelihood =argDict["log_likelihood"])
    
def _TrainSReC_casing(conn,argDict):
  TrainSReC_casing(conn=conn,
        train_path=argDict["train_path"], eval_path=argDict["eval_path"],
        batch=argDict["batch"],load=argDict["load"],newModel=argDict["newModel"],
        workers=argDict["workers"], epochs=argDict["epochs"],resblocks=argDict["resblocks"],
        n_feats=argDict["n_feats"], scale=argDict["scale"],lr=argDict["lr"], eval_iters=argDict["eval_iters"],
        lr_epochs =argDict["lr_epochs"],plot_iters=argDict["plot_iters"], k=argDict["k"], clip=argDict["clip"],
        crop=argDict["crop"], gd=argDict["gd"])
  
  
def main():
    from multiprocessing.connection import Connection
    import socket
    from time import sleep
    import os,sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("child: sys.prefix - ",sys.prefix)
    func= {
        'testing':testing,
        'compress': _compressWithSReCMain,
        'decompress':_deCompressWithSReCMain,
        'eval':_evalWithSReCMain,
        'timeTest':_timeTest_DeAndCompress,
        'train':_TrainSReC_casing   
    }.get(sys.argv[1], testing)
    s2 =socket.socket( family=socket.AF_UNIX,fileno=int(sys.argv[2]),type=socket.SOCK_STREAM,proto=0)
    s2.setblocking(True)
    myConnection2 = Connection(os.dup(s2.fileno()))
    s2.close()
    myConnection2.send("alive")
    sleep(1)
    if(myConnection2.poll(5)):
        try:
            ans=myConnection2.recv()
            func(myConnection2,ans)           
        except EOFError:
            pass      
    
    

if __name__ == "__main__":
    
    #CommandLineEvalWithWithSReC()
    #
    #main()
    testing()