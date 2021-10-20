"""functions directly operating l3c model
"""
import mkl
import  os
import sys






#---------L3C---------------------------------------------------

 

def l3cpyEncryptDecrypt(connectionToFather,pathToModels,modelName,inputPath,action,outPutPath):
    """de/compression for L3C
         note : the function presumes it is a new procces separate from the procces calling it 
    Args:
        connectionToFather (Connection): [Connection in order to communicate with the father process]
        pathToModels (str): [A path to the directory the used model is located at]. Example  "/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir".
        modelName (str): [the name of the trained weights we want to use]. Example  "0306_0001".
        inputPath (list): [full paths to the images to be compressed]. Example  ["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"].
        action (str): ["enc" for using the model to compress, 
            "dec in order to use the model for compresstion "]. 
        outPutPath (list): [a list of full paths for the output of each image]. Example  ["out1.l3c","out2.l3c"].
    """
    
    #changing inherited attributes from father process
    sys.path[0]=os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #imports
    import mkl

    from l3c import _FakeFlags,parse_device_flag,parse_args

    #l3c imports 
    import torch.backends.cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    import pytorch_ext as pe

    import argparse
    from test.multiscale_tester import MultiscaleTester, EncodeError, DecodeError
        
    from torchac import torchac # throws an exception if no backend available!

    
    #working the model
    if(action=="enc"):
        flags = parse_args([pathToModels,modelName,action,"inputPath","out.l3c","--overwrite"])
    else:
        flags = parse_args([pathToModels,modelName,action,"inputPath","out.l3c"])


    parse_device_flag(flags.device)
    print('Testing {} at {} ---'.format(flags.log_date, flags.restore_itr))
    tester = MultiscaleTester(flags.log_date, _FakeFlags(flags), flags.restore_itr, l3c=True)
    
    for imagePath,compressedImagePath in zip(inputPath,outPutPath):
        flags.img_p=imagePath
        if flags.mode == 'enc':
            flags.out_p=compressedImagePath
            try:
                tester.resetFlags(flags)
                tester.encode(flags.img_p, flags.out_p, flags.overwrite)
            except EncodeError as e:
                print('*** EncodeError:', e)
        else:
            flags.out_p_png=compressedImagePath
            try:
                tester.resetFlags(flags)
                tester.decode(flags.img_p, flags.out_p_png)
            except DecodeError as e:
                print('*** DecodeError:', e)
                
        if(connectionToFather.poll()):
            try:
                messege=connectionToFather.recv()
                if(messege=="canceled"):
                    connectionToFather.close()
                    return
            except EOFError:
                print("wrong message")
                pass
    connectionToFather.send("Done")      
                
def l3cpyEncryptDecryptTime(connectionToFather,pathToModels,modelName,inputPath):
    """ time for de/compression for L3C
         note : the function presumes it is a new procces separate from the procces calling it 

    Args:
        connectionToFather (Connection): [Connection in order to communicate with the father process]
        pathToModels (str): [A path to the directory the used model is located at]. Example  "/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir".
        modelName (str): [the name of the trained weights we want to use]. Example  "0306_0001".
        inputPath (list): [full paths to the images to be compressed]. Example  ["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"].
    """
    
    #changing inherited attributes from father process
    sys.path[0]=os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #imports
    from l3c import _FakeFlags,parse_device_flag,parse_args
    import time
    import ntpath
    import tempfile
    #l3c imports 
    import torch.backends.cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    import pytorch_ext as pe

    import argparse
    from test.multiscale_tester import MultiscaleTester, EncodeError, DecodeError
        
    from torchac import torchac # throws an exception if no backend available!

    
    #working the model

    flags = parse_args([pathToModels,modelName,"enc","inputPath","out.l3c","--overwrite"])
       

    parse_device_flag(flags.device)
    print('Testing {} at {} ---'.format(flags.log_date, flags.restore_itr))
    tester = MultiscaleTester(flags.log_date, _FakeFlags(flags), flags.restore_itr, l3c=True)
    tempFolder = tempfile.TemporaryDirectory(dir = os.path.dirname(os.path.realpath(__file__)))
    start=time.time()
    for imagePath in inputPath:

            flags = parse_args([pathToModels,modelName,"enc","inputPath","out.l3c","--overwrite"])
            flags.img_p=imagePath
            flags.out_p=os.path.join(tempFolder.name,ntpath.basename(os.path.splitext(imagePath)[0]) +".l3c" )
            try:
                tester.resetFlags(_FakeFlags(flags))
                tester.encode(flags.img_p, flags.out_p, flags.overwrite)
            except EncodeError as e:
                print('*** EncodeError:', e)
               
            tempImag_p=os.path.splitext(flags.out_p)[0] +".l3c"
            flags = parse_args([pathToModels,modelName,"dec","inputPath","out.l3c"])    
            flags.out_p_png= os.path.join(tempFolder.name,ntpath.basename(os.path.splitext(imagePath)[0]+".png"))
            flags.img_p=tempImag_p
            
            try:
                tester.resetFlags(_FakeFlags(flags))
                tester.decode(flags.img_p, flags.out_p_png)
            except DecodeError as e:
                print('*** DecodeError:', e)
                
                
            if(connectionToFather.poll()):
                try:
                    messege=connectionToFather.recv()
                    if(messege=="canceled"):
                        connectionToFather.close()
                        return
                except EOFError:
                    print("wrong message")
                    pass 
    end = time.time()
    connectionToFather.send(end-start)
    #print("final Time: ",end-start)
                
def CommandLineVlaidationWithWithL3C(
                    pathToL3Cpy="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/src/test.py",
                    pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001",
                    validationData="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r",
                    names="L3C",
                    overwriteCatch=True,
                    recursive="auto"
                    ):
    import subprocess
    originalCwd=os.getcwd()   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    command ="python "+pathToL3Cpy+" "+pathToModels+" "+modelName+ " "+ validationData+ " "+ "--names"+" "+names+ " "+"--overwrite_cache"+" "+ "--recursive="+recursive #+ " "+"--time_report /home/yonathan/Documents"
    subprocess.Popen(command,shell=True)
    os.chdir(os.path.dirname(originalCwd))
    
def CommandLineTrainingWithWithL3C():
    import subprocess
    originalCwd=os.getcwd()   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    command ="python train.py /home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/src/configs/ms/cr.cf /home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/src/configs/dl/oi.cf /home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/logDir"
    command="python train.py /home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/src/configs/ms/cr.cf  /home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/src/configs/dl/oi.cf  logs --restore /home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/logDir/0331_1703* --restore_restart"
    subprocess.Popen(command,shell=True)
    os.chdir(os.path.dirname(originalCwd))

def l3CEvaluation(connectionToFather,pathToModels, modelName,validationData,overwriteCatch,recursive):
    import mkl
    import sys
    #changing inherited attributes from father process
    sys.path[0]=os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ##l3c imports 
    import torch.backends.cudnn
    torch.backends.cudnn.benchmark = True

    import argparse
    from operator import itemgetter

    from helpers.aligned_printer import AlignedPrinter
    from helpers.testset import Testset
    from test.multiscale_tester import MultiscaleTester

  
    # a copy of parse_args function in test.py 
        # done because there is folder with the same name in the diractory 
    def parse_args(args):
        p = argparse.ArgumentParser()

        p.add_argument('log_dir', help='Directory of experiments. Will create a new folder, LOG_DIR_test, to save test '
                                    'outputs.')
        p.add_argument('log_dates', help='A comma-separated list, where each entry is a log_date, such as 0104_1345. '
                                        'These experiments will be tested.')
        p.add_argument('images', help='A comma-separated list, where each entry is either a directory with images or '
                                    'the path of a single image. Will test on all these images.')
        p.add_argument('--match_filenames', '-fns', nargs='+', metavar='FILTER',
                    help='If given, remove any images in the folders given by IMAGES that do not match any '
                            'of specified filter.')
        p.add_argument('--max_imgs_per_folder', '-m', type=int, metavar='MAX',
                    help='If given, only use MAX images per folder given in IMAGES. Default: None')
        p.add_argument('--crop', type=int, help='Crop all images to CROP x CROP squares. Default: None')

        p.add_argument('--names', '-n', type=str,
                    help='Comma separated list, if given, must be as long as LOG_DATES. Used for output. If not given, '
                            'will just print LOG_DATES as names.')

        p.add_argument('--overwrite_cache', '-f', action='store_true',
                    help='Ignore cached test outputs, and re-create.')
        p.add_argument('--reset_entire_cache', action='store_true',
                    help='Remove cache.')

        p.add_argument('--restore_itr', '-i', default='-1',
                    help='Which iteration to restore. -1 means latest iteration. Will use closest smaller if exact '
                            'iteration is not found. Default: -1')

        p.add_argument('--recursive', default='0',
                    help='Either an number or "auto". If given, the rgb configs with num_scales == 1 will '
                            'automatically be evaluated recursively (i.e., the RGB baseline). See _parse_recursive_flag '
                            'in multiscale_tester.py. Default: 0')

        p.add_argument('--sample', type=str, metavar='SAMPLE_OUT_DIR',
                    help='Sample from model. Store results in SAMPLE_OUT_DIR.')

        p.add_argument('--write_to_files', type=str, metavar='WRITE_OUT_DIR',
                    help='Write images to files in folder WRITE_OUT_DIR, with arithmetic coder. If given, the cache is '
                            'ignored and no test output is printed. Requires torchac to be installed, see README. Files '
                            'that already exist in WRITE_OUT_DIR are overwritten.')
        p.add_argument('--compare_theory', action='store_true',
                    help='If given with --write_to_files, will compare actual bitrate on disk to theoretical bitrate '
                            'given by cross entropy.')
        p.add_argument('--time_report', type=str, metavar='TIME_REPORT_PATH',
                    help='If given with --write_to_files, write a report of time needed for each component to '
                            'TIME_REPORT_PATH.')

        p.add_argument('--sort_output', '-s', choices=['testset', 'exp', 'itr', 'res'], default='testset',
                    help='How to sort the final summary. Possible values: "testset" to sort by '
                            'name of the testset // "exp" to sort by experiment log_date // "itr" to sort by iteration // '
                            '"res" to sort by result, i.e., show smaller first. Default: testset')
        return p.parse_args(args)
    
    modelName=modelName.split()[0]
    flags = parse_args([pathToModels,modelName,validationData,overwriteCatch,recursive])

    if flags.compare_theory and not flags.write_to_files:
        raise ValueError('Cannot have --compare_theory without --write_to_files.')
    if flags.write_to_files and flags.sample:
        raise ValueError('Cannot have --write_to_files and --sample.')
    if flags.time_report and not flags.write_to_files:
        raise ValueError('--time_report only valid with --write_to_files.')
    if(type(validationData)!= list):
        testsets = [Testset(images_dir_or_image.rstrip('/'), flags.max_imgs_per_folder,
                            # Append flags.crop to ID so that it creates unique entry in cache
                            append_id=f'_crop{flags.crop}' if flags.crop else None)
                    for images_dir_or_image in flags.images.split(',')]
    else:
        testsets = [Testset(root_dir_or_img=flags.images)]
        
        
    if flags.match_filenames:
        for ts in testsets:
            ts.filter_filenames(flags.match_filenames)

    splitter = ',' if ',' in flags.log_dates else '|'  # support tensorboard strings, too
    results = []
    log_dates = flags.log_dates.split(splitter)
    for log_date in log_dates:
        for restore_itr in map(int, flags.restore_itr.split(',')):
            print('Testing {} at {} ---'.format(log_date, restore_itr))
            tester = MultiscaleTester(log_date, flags, restore_itr)
            tempRes,mulResults=   tester.test_all(testsets)
            results += tempRes
    print(mulResults[0].mean())        
    connectionToFather.send(mulResults[0].mean())

    # if --names was passed: will print 'name (log_date)'. otherwise, will just print 'log_date'
    if flags.names:
        names = flags.names.split(splitter) if flags.names else log_dates
        names_to_log_date = {log_date: f'{name} ({log_date})'
                             for log_date, name in zip(log_dates, names)}
    else:
        # set names to log_dates if --names is not given, i.e., we just output log_date
        names_to_log_date = {log_date: log_date for log_date in log_dates}
    if not flags.write_to_files:
        print('*** Summary:')
        with AlignedPrinter() as a:
            sortby = {'testset': 0, 'exp': 1, 'itr': 2, 'res': 3}[flags.sort_output]
            a.append('Testset', 'Experiment', 'Itr', 'Result')
            for testset, log_date, restore_itr, result in sorted(results, key=itemgetter(sortby)):
                a.append(testset.id,  names_to_log_date[log_date], str(restore_itr), result)



def l3CTraining(connectionToFather,newModel,ms_config_path,dl_config_path,pathToModel,
                trainDataList=[],valiDataLIst=[]):
    
    from helpers.config_checker import DEFAULT_CONFIG_DIR, ConfigsRepo
    configs_dir=DEFAULT_CONFIG_DIR
    
    import torch

    # seed at least the random number generators.
    # doesn't guarantee full reproducability: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    import argparse
    import sys

    import torch.backends.cudnn
    from fjcommon import no_op

    import pytorch_ext as pe
    from helpers.global_config import global_config
    from helpers.saver import Saver
    from train.multiscale_trainer import MultiscaleTrainer
    from train.train_restorer import TrainRestorer
    from train.trainer import LogConfig
    
    torch.backends.cudnn.benchmark = True
    
    def _print_debug_info():
        print('*' * 80)
        print(f'DEVICE == {pe.DEVICE} // PyTorch v{torch.__version__}')
        print('*' * 80)
    
    def parse_args(args):
        p = argparse.ArgumentParser()

        p.add_argument('ms_config_p', help='Path to a multiscale config, see README')
        p.add_argument('dl_config_p', help='Path to a dataloader config, see README')
        p.add_argument('log_dir_root', default='logs', help='All outputs (checkpoints, tensorboard) will be saved here.')
        p.add_argument('--temporary', '-t', action='store_true',
                    help='If given, outputs are actually saved in ${LOG_DIR_ROOT}_TMP.')
        p.add_argument('--log_train', '-ltrain', type=int, default=100,
                    help='Interval of train output.')
        p.add_argument('--log_train_heavy', '-ltrainh', type=int, default=5, metavar='LOG_HEAVY_FAC',
                    help='Every LOG_HEAVY_FAC-th time that i % LOG_TRAIN is 0, also output heavy logs.')
        p.add_argument('--log_val', '-lval', type=int, default=500,
                    help='Interval of validation output.')

        p.add_argument('-p', action='append', nargs=1,
                    help='Specify global_config parameters, see README')

        p.add_argument('--restore', type=str, metavar='RESTORE_DIR',
                    help='Path to the log_dir of the model to restore. If a log_date ('
                            'MMDD_HHmm) is given, the model is assumed to be in LOG_DIR_ROOT.')
        p.add_argument('--restore_continue', action='store_true',
                    help='If given, continue in RESTORE_DIR instead of starting in a new folder.')
        p.add_argument('--restore_restart', action='store_true',
                    help='If given, start from iteration 0, instead of the iteration of RESTORE_DIR. '
                            'Means that the model in RESTORE_DIR is used as pretrained model')
        p.add_argument('--restore_itr', '-i', type=int, default=-1,
                    help='Which iteration to restore. -1 means latest iteration. Will use closest smaller if exact '
                            'iteration is not found. Only valid with --restore. Default: -1')
        p.add_argument('--restore_strict', type=str, help='y|n', choices=['y', 'n'], default='y')

        p.add_argument('--num_workers', '-W', type=int, default=8,
                    help='Number of workers used for DataLoader')

        p.add_argument('--saver_keep_tmp_itr', '-si', type=int, default=250)
        p.add_argument('--saver_keep_every', '-sk', type=int, default=10)
        p.add_argument('--saver_keep_tmp_last', '-skt', type=int, default=3)
        p.add_argument('--no_saver', action='store_true',
                    help='If given, no checkpoints are stored.')

        p.add_argument('--debug', action='store_true')

        return p.parse_args(args)
    
    if(newModel):
        flags = parse_args([ms_config_path,dl_config_path,pathToModel])
    else:
        flags = parse_args([ms_config_path,dl_config_path,"--restore "+pathToModel,'--restore_continue'])

        

    _print_debug_info()

    if flags.debug:
        flags.temporary = True

    global_config.add_from_flag(flags.p)
    print(global_config)

    ConfigsRepo(configs_dir).check_configs_available(flags.ms_config_p, flags.dl_config_p)

    saver = (Saver(flags.saver_keep_tmp_itr, flags.saver_keep_every, flags.saver_keep_tmp_last,
                   verbose=True)
             if not flags.no_saver
             else no_op.NoOp())

    restorer = TrainRestorer.from_flags(flags.restore, flags.log_dir_root, flags.restore_continue, flags.restore_itr,
                                        flags.restore_restart, flags.restore_strict)

    trainer = MultiscaleTrainer(flags.ms_config_p, flags.dl_config_p,
                                flags.log_dir_root + ('_TMP' if flags.temporary else ''),
                                LogConfig(flags.log_train, flags.log_val, flags.log_train_heavy),
                                flags.num_workers,
                                saver=saver, restorer=restorer,
                                trainDataList=trainDataList,
                                valiDataLIst=valiDataLIst)
    if not flags.debug:
        trainer.train()
    else:
        trainer.debug()
        
          
  

    import copy
    connectionToFather.send([copy.deepcopy(trainer.log_date),copy.deepcopy(trainer.log_dir)])
    #clean the memory
    import gc
    del trainer
    gc.collect()
    

def dlConfig(batchsize_train="30",
                        batchsize_val="30",
                        crop_size="128",
                        max_epochs="2",
                        image_cache_pkl='/home/yonathan/Documents/GitHub/L3C-PyTorch/logDir_test',
                        train_imgs_glob='/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r',
                        val_glob='/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r',
                        num_val_batches='5'):
    
        workPath=os.path.join(os.path.dirname(os.path.abspath(__file__)),"configs")
        workPath=os.path.join(workPath,"dl")  
        with  open( os.path.join(workPath,"oi (For Setting).cf"), "rt") as fin:
            with open(os.path.join(workPath,"oi.cf"), "wt") as fout:
                for line in fin:
                    #batchsize_train
                    line=line.replace('BS_T', str(batchsize_train))
                    #batchsize_val
                    line=line.replace('BS_V', str(batchsize_val))
                    #crop_size
                    line=line.replace('CS', str(crop_size))
                    #max_epochs
                    line=line.replace('ME', str(max_epochs))
                    #image_cache_pkl
                    line=line.replace('ICP',str(image_cache_pkl ))
                    #train_imgs_glob
                    line=line.replace('TIG', str(train_imgs_glob))
                    #val_glob
                    line=line.replace('VG',str(val_glob ))
                    #num_val_batches
                    fout.write(line.replace('VB',str( num_val_batches)))
        return os.path.join(workPath,"oi.cf")
    
def msConfig(optim,lr_initial,lr_schedule,weight_decay,num_scales,shared_across_scales,Cf,kernel_size,dmll_enable_grad,rgb_bicubic_baseline,enc_cls,
             enc_num_blocks,enc_feed_F,enc_importance_map,learned_L,dec_cls,dec_num_blocks,dec_skip,q_cls,q_C,q_L,q_levels_range,q_sigma,prob_K,
             after_q1x1,x4_down_in_scale0):
    workPath=os.path.join(os.path.dirname(os.path.abspath(__file__)),"configs")
    workPath=os.path.join(workPath,"ms")
    with  open( os.path.join(workPath,"cr (For Setting).cf"), "rt") as fin:
        with open(os.path.join(workPath,"cr.cf"), "wt") as fout:  
            for line in fin:
                line=line.replace('OPT', "'"+str(optim)+"'")
                line=line.replace('_LR_INT_', str(lr_initial))
                line=line.replace('_LR_SC_', "'"+str(lr_schedule)+"'")
                line=line.replace('_WD_', str(weight_decay))
                line=line.replace('_NS_', str(num_scales))
                line=line.replace('_SAS_', str(shared_across_scales))
                line=line.replace('_CF_', str(Cf))
                line=line.replace('_KS_', str(kernel_size))
                line=line.replace('_DEG_', str(dmll_enable_grad))
                line=line.replace('_RBB_', str(rgb_bicubic_baseline))
                line=line.replace('ENC_CLS', "'"+str(enc_cls)+"'")
                line=line.replace('ENC_NUM', str(enc_num_blocks))
                line=line.replace('ENC_F', str(enc_feed_F))
                line=line.replace('ENC_IM', str(enc_importance_map))
                line=line.replace('_LL_', str(learned_L))
                line=line.replace('DEC_CLS', "'"+str(dec_cls)+"'")
                line=line.replace('DEC_BN', str(dec_num_blocks))
                line=line.replace('DEC_SKIP', str(dec_skip))
                line=line.replace('Q_CLS', "'"+str(q_cls)+"'")
                line=line.replace('_QC_', str(q_C))
                line=line.replace('_QL_', str(q_L))
                line=line.replace('Q_LR', str(q_levels_range))
                line=line.replace('_QS_', str(q_sigma))
                line=line.replace('PR_K', str(prob_K))
                line=line.replace('A_Q1Q', str(after_q1x1))
                fout.write(line.replace('X4DIS', str(x4_down_in_scale0)))
    return os.path.join(workPath,"cr.cf")
                
                
def L3CPreOp(config):
    
    image_cache_pkl=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"logDir_test")
    #TODO change 
    train_imgs_glob=''
    val_glob=''
    
    dlDict=config["dl"]
    msDict=config["ms"]
    
    dl_config_path=dlConfig(batchsize_train=dlDict["batchsize_train"],
                            batchsize_val=dlDict["batchsize_val"],
                            crop_size=dlDict["crop_size"],
                            max_epochs=dlDict["max_epochs"],
                            image_cache_pkl=dlDict["image_cache_pkl"],
                            train_imgs_glob=dlDict["train_imgs_glob"],
                            val_glob=dlDict["val_glob"],
                            num_val_batches=dlDict["num_val_batches"])
    
    ms_config_path=msConfig(optim=msDict["optim"],lr_initial=msDict["lr.initial"],lr_schedule=msDict["lr.schedule"],weight_decay=msDict["weight_decay"],num_scales=msDict["num_scales"],
                            shared_across_scales=msDict["shared_across_scales"],
                            Cf=msDict["Cf"],kernel_size=msDict["kernel_size"],dmll_enable_grad=msDict["dmll_enable_grad"],rgb_bicubic_baseline=msDict["rgb_bicubic_baseline"],
                            enc_cls=msDict["enc.cls"],
                            enc_num_blocks=msDict["enc.num_blocks"],enc_feed_F=msDict["enc.feed_F"],enc_importance_map=msDict["enc.importance_map"],
                            learned_L=msDict["learned_L"],dec_cls=msDict["dec.cls"],dec_num_blocks=msDict["dec.num_blocks"],
                            dec_skip=msDict["dec.skip"],q_cls=msDict["q.cls"],q_C=msDict["q.C"],q_L=msDict["q.L"],q_levels_range=msDict["q.levels_range"],q_sigma=msDict["q.sigma"],
                            prob_K=msDict["prob.K"],
                            after_q1x1=msDict["after_q1x1"],x4_down_in_scale0=msDict["x4_down_in_scale0"])

    return dl_config_path,ms_config_path
    
    
                
def L3CTrainingCasing(connectionToFather,trainDataList,valiDataLIst,pathToModel,ms_config_path,dl_config_path,newModel=True):
    l3CTraining(connectionToFather,newModel,ms_config_path,dl_config_path,pathToModel,
                trainDataList=trainDataList ,valiDataLIst=valiDataLIst)
#----------------------testing-------------------------
def testConfigPrep():
    #DL
    ###############################################################################
    dlDict={}
    dlDict["batchsize_train"]  = 30
    dlDict["batchsize_val"]  = 30
    dlDict["crop_size"]  = 128

    dlDict["max_epochs"]  = 2

    dlDict["image_cache_pkl"]  = None
    dlDict["train_imgs_glob"]  = 'path/to/train/ (check readme)'
    dlDict["val_glob"] = 'path/to/val (check readme)'

    dlDict["val_glob_min_size"] = None  # for cache_p
    dlDict["num_val_batches"] = 5


    #MS
    ###############################################################################
    msDict={}
    

    msDict["optim"]  = 'RMSprop'

    msDict["lr.initial"]  = 0.0001
    msDict["lr.schedule" ] = 'exp_0.75_e5'
    msDict["weight_decay"]  = 0

    msDict["num_scales"]  = 3
    msDict["shared_across_scales"]  = False

    msDict["Cf"]  = 64
    msDict["kernel_size"]  = 3

    msDict["dmll_enable_grad"]  = 0

    msDict["rgb_bicubic_baseline"]  = False

    msDict["enc.cls"]  = 'EDSRLikeEnc'
    msDict["enc.num_blocks"]  = 8
    msDict["enc.feed_F"]  = True
    msDict["enc.importance_map"]  = False

    msDict["learned_L"]  = False

    msDict["dec.cls"]  = 'EDSRDec'
    msDict["dec.num_blocks"]  = 8
    msDict["dec.skip"]  = True

    msDict["q.cls"]  = 'Quantizer'
    msDict["q.C"]  = 5
    msDict["q.L"]  = 25
    msDict["q.levels_range"]  = (-1, 1)
    msDict["q.sigma"]  = 2

    msDict["prob.K"]  = 10

    msDict["after_q1x1"]  = True
    msDict["x4_down_in_scale0"]  = False

    return{"ms":msDict,"dl":dlDict}

def testConfig():
    config=testConfigPrep()
    a,b=L3CPreOp(config)
    os.remove(a)
    os.remove(b)


    
def  testingL3CpyEncryptDecrypt():
    from multiprocessing import Pipe
    from time import sleep
    conn1,conn2=Pipe()
    #l3cpyEncryptDecrypt( conn2,    
     #               pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
      #              modelName="0306_0001",
       #             inputPath=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"],
        #            action="enc",
         #           outPutPath=["out1.l3c","out2.l3c"]            
          #          )
    l3cpyEncryptDecrypt( conn2,    
                    pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001",
                    inputPath=["/home/yonathan/Documents/git/NeuralNetworkCompression/Nature Background Image.l3c_0306_0001_cr_oi.part0"],
                    action="dec",
                    outPutPath=["test.png"]            
                    )
def  testingL3CpyEncryptDecryptTime():
    from multiprocessing import Pipe

    conn1,conn2=Pipe()
    l3cpyEncryptDecryptTime( conn2,    
                    pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001",
                    inputPath=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"],
                    )
    
    
     
def testingValidationWithWithL3C():
    #CommandLineVlaidationWithWithL3C()
    from multiprocessing import Pipe

    conn1,conn2=Pipe()
    if(False):
        l3CEvaluation( pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001",
                    validationData="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)",
                    overwriteCatch="--overwrite_cache",
                    recursive="--recursive=auto",connectionToFather=conn2)
    else:
        l3CEvaluation( pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001",
                    validationData=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)/e01263c97c5382b0.png",
                                    "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)/e97858d8e422d201.png",
                                    "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)/e4548333f23fa923.png",
                                    "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)/ea5e2b9eaa6e4e49.png",
                                    "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)/eada6632f433eebc.png"],
                    overwriteCatch="--overwrite_cache",
                    recursive="--recursive=auto",connectionToFather=conn2)
    print(conn1.recv())
    
def testingTrainingWithL3C():
    ms_config_path="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/src/configs/backups/cr.cf"
    dl_config_path="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/src/configs/backups/oi.cf"
    pathToModel="/home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/logDir"
    trainDataList=os.listdir("/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r")
    valiDataLIst=os.listdir("/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r")
    
    if(False):
        for i in range(len(trainDataList)):
            trainDataList[i]=os.path.join("/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r",trainDataList[i])
        for i in range(len(valiDataLIst)):
                valiDataLIst[i]=os.path.join("/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r",valiDataLIst[i])
    else:
         trainDataList="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r"
         valiDataLIst="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r"
         
    #pathToModel="/home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/logDir/0331_1703*"
     
    L3CTrainingCasing(connectionToFather=None,pathToModel=pathToModel,ms_config_path=ms_config_path,dl_config_path=dl_config_path,
                      trainDataList=trainDataList,
                      valiDataLIst=valiDataLIst,newModel=True)

    
def testing():
    #testingL3CpyEncryptDecrypt()
    #testingValidationWithWithL3C()
    #CommandLineVlaidationWithWithL3C()
    #testingL3CpyEncryptDecryptTime()
    testingTrainingWithL3C()
    #testConfig()
    
def _l3cpyEncryptDecrypt(conn,argDict):
    dl,ms=L3CPreOp(argDict["config"])
    l3cpyEncryptDecrypt(connectionToFather=conn,pathToModels=argDict["pathToModels"],modelName=argDict["modelName"],inputPath=argDict["inputPath"],action=argDict["action"],outPutPath=argDict["outPutPath"])
    os.remove(dl)
    os.remove(ms)

    
def _l3cpyEncryptDecryptTime(conn,argDict):
    dl,ms=L3CPreOp(argDict["config"])
    l3cpyEncryptDecryptTime(connectionToFather=conn,pathToModels=argDict["pathToModels"],modelName=argDict["modelName"],inputPath=argDict["inputPath"])
    os.remove(dl)
    os.remove(ms)
    
def _l3CEvaluation(conn,argDict):    
    dl,ms=L3CPreOp(argDict["config"])
    l3CEvaluation(connectionToFather=conn,pathToModels=argDict["pathToModels"],modelName=argDict["modelName"],validationData=argDict["validationData"],overwriteCatch=argDict["overwriteCatch"],recursive=argDict["recursive"])
    os.remove(dl)
    os.remove(ms)

def _L3CTrainingCasing(conn,argDict):
    dl,ms=L3CPreOp(argDict["config"])
    L3CTrainingCasing(connectionToFather=conn,ms_config_path=ms,dl_config_path=dl,trainDataList=argDict["trainDataList"],
                      valiDataLIst=argDict["valiDataLIst"],pathToModel=argDict["pathToModel"],newModel=argDict["newModel"])
    os.remove(dl)
    os.remove(ms)
    


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
        'de/compress': _l3cpyEncryptDecrypt,
        'eval':_l3CEvaluation,
        'timeTest':_l3cpyEncryptDecryptTime,
        'train':_L3CTrainingCasing,
    }.get(sys.argv[1], testing)
    s2 =socket.socket( family=socket.AF_UNIX,fileno=int(sys.argv[2]),type=socket.SOCK_STREAM,proto=0)
    s2.setblocking(True)
    myConnection2 = Connection(os.dup(s2.fileno()))
    s2.close()
    myConnection2.send("alive")
    sleep(1)
    if(myConnection2.poll(1)):
        try:
            ans=myConnection2.recv()
            func(myConnection2,ans)           
        except EOFError:
            pass      
    
    

if __name__ == "__main__":
    #CommandLineTrainingWithWithL3C()
    #testing()
    main()
     
