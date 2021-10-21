#---------------------------subProcess based callers------------------------------------


def subProcessForSReC(environmentPath,action,argDict):
    import socket,subprocess,time,os
    from multiprocessing.connection import Connection
   
    s1, s2 = socket.socketpair()
    s1.setblocking(True)
    s2.setblocking(True)
    fileNum=os.dup(s2.fileno())
    pathToSReC_operations=os.path.join(os.path.dirname(os.path.abspath(__file__)),"SReC_operations.py")
    child = subprocess.Popen([environmentPath,pathToSReC_operations, action,str(fileNum)], pass_fds=(fileNum,))
    myConnection1 = Connection(os.dup(s1.fileno()))
    s1.close()
    s2.close()
    time.sleep(1)
    if(myConnection1.poll(5)):
        try:
            ans=myConnection1.recv()
            if(ans=="alive"):
                myConnection1.send(argDict)
                return child,myConnection1           
        except EOFError:
            pass    
    return None


def compressWithSReC(path,save_path,load,config,decode =False,log_likelihood=False,suffix=".srec",environmentPath="/home/yonathan/anaconda3/envs/srec_env_test/bin/python"):
    return subProcessForSReC(environmentPath=environmentPath,action='compress',argDict={"path":path,"save_path":save_path,"load":load,"resblocks":config["resblocks"],"n_feats":config["n_feats"],
                                                                                        "scale":config["scale"],"k":config["k"],"crop":0,"log_likelihood":log_likelihood,
                                                                                        "decode":decode,"suffix":suffix})

def DeCompressWithSReC(path,save_path,load,config,suffix=".srec",environmentPath="/home/yonathan/anaconda3/envs/srec_env_test/bin/python"):
    return subProcessForSReC(environmentPath=environmentPath,action='decompress',argDict={"path":path,"save_path":save_path,"load":load,"resblocks":config["resblocks"],
                                                                                          "n_feats":config["n_feats"],"scale":config["scale"],
                                                                                          "suffix":suffix,"k":config["k"]})

def evalWithSReC(path,load,config,environmentPath="/home/yonathan/anaconda3/envs/srec_env_test/bin/python"):
    return subProcessForSReC(environmentPath=environmentPath,action='eval',argDict={"path":path,"load":load,"workers":config["workers"],
                                                                                    "resblocks":config["resblocks"],"n_feats":config["n_feats"],
                                                                                    "scale":config["scale"],"k":config["k"],"crop":0})

def timeTestSReC(path,load,config,log_likelihood=False,environmentPath="/home/yonathan/anaconda3/envs/srec_env_test/bin/python"):
    return subProcessForSReC(environmentPath=environmentPath,action='timeTest',argDict={"path":path,"load":load,"resblocks":config["resblocks"],
                                                                                        "n_feats":config["n_feats"],"scale":config["scale"],"k":config["k"],
                                                                                        "crop":0,"log_likelihood":log_likelihood})
    

def trainSReC(train_path,eval_path,batch,load,newModel,config,epochs,environmentPath="/home/yonathan/anaconda3/envs/srec_env_test/bin/python"):
    return subProcessForSReC(environmentPath=environmentPath,action='train',argDict={"train_path":train_path,"eval_path":eval_path,"batch":batch,
                                                                                     "load":load,"newModel":newModel,"workers":config["workers"],
                                                                                     "epochs":epochs,"resblocks":config["resblocks"],"n_feats":config["n_feats"],
                                                                                     "scale":config["scale"],"lr":config["lr"],"eval_iters":config["eval_iters"],
                                                                                     "lr_epochs":config["lr_epochs"],"plot_iters":config["plot_iters"],"k":config["k"],
                                                                                     "clip":config["clip"],"crop":config["crop"],"gd":config["gd"]})
#--------------------------- testing subProcess based callers------------------------------------
def createConfigSrec():
    SReCDict={}
    SReCDict["Name"]="Default"
    SReCDict["resblocks"] =3
    SReCDict["n_feats"]  = 64
    SReCDict["scale"]  = 3
    SReCDict["k"] =10
    SReCDict["workers"]  =2 
    SReCDict["crop"]  =128
    SReCDict["lr"]=1e-4
    SReCDict["eval_iters"]=1e-4
    SReCDict["lr_epochs"]=1
    SReCDict["plot_iters"]=1000
    SReCDict["clip"]=0.5
    SReCDict["gd"]="adam"
    return SReCDict
    

def testingAssit_waitForSubProcessToGiveAnswer(p,c):
        from time import sleep
        sleep(2)
        while(p.poll()== None or c.poll()):
            if(c.poll()):
                try:
                    answer=c.recv()
                    if(answer=="Done"):
                        break
                    else:
                        print(answer) 
                #Raises EOFError if there is nothing left to receive and the other end was closed.
                except EOFError  as e:
                    #print(e.__traceback__)
                    break
            sleep(2)
        return answer

def testinCompressWithSReC():
    SReCDict=createConfigSrec()
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"]
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b0328075ddf03fc.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0bdcd1cc8c08d4e0.png"
          ]
    import os
    path= os.listdir("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r")
    for i in range(0,len(path)):
        path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",path[i])
    save_path="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression"
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    p,c=compressWithSReC(path,save_path,load,config=SReCDict)
    testingAssit_waitForSubProcessToGiveAnswer(p,c)
    
    
def testinDeCompressWithSReC():
    SReCDict=createConfigSrec()
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"]
    path=["/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0a7f13330a5d0023.srec","/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0b6ef374714ee86b.srec",
          "/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0b0328075ddf03fc.srec","/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/0bdcd1cc8c08d4e0.srec"]
    import os
    path= os.listdir("/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression")
    for i in range(0,len(path)):
        path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/compression/",path[i])
    save_path="/home/yonathan/Documents/GitHub/SReC/Compression&Decompression/decompression"
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    p,c=DeCompressWithSReC(path,save_path,load,config=SReCDict)
    testingAssit_waitForSubProcessToGiveAnswer(p,c)
    
def testingEvalWithSRedc():
    import os
    SReCDict=createConfigSrec()
    path= os.listdir("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r")
    for i in range(0,len(path)):
        path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",path[i])
        
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"  
    p,c=evalWithSReC(path,load,config=SReCDict)
    testingAssit_waitForSubProcessToGiveAnswer(p, c)
  
def testingTimeWithSReC():
    SReCDict=createConfigSrec()
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"]
    path=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b0328075ddf03fc.png",
          "/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0bdcd1cc8c08d4e0.png"
          ]
    import os
    path= os.listdir("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r")
    for i in range(0,len(path)):
        path[i]= os.path.join("/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r",path[i])
    load ="/home/yonathan/Documents/GitHub/SReC/models/openimages.pth"
    p,c=timeTestSReC(path,load,config=SReCDict)
    testingAssit_waitForSubProcessToGiveAnswer(p,c)  
    
def testTraining():
    
    train_path=  "/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r"
    eval_path="/home/yonathan/Documents/GitHub/SReC/AtualDatasets/validation/val_oi_500_r"
    batch =20
    load ="/dev/null"
    load ="/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/models"
    load ="/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/models/openimages.pth"
    load ="/home/yonathan/Documents/git/NeuralNetworkCompression/models/SReC/models/May-30-2021_11:53:00.pth"
    load = "" 
    newModel=True
    config=createConfigSrec()
    epoch=2
    trainSReC(train_path,eval_path,batch,load,newModel,config,epoch,environmentPath="/home/yonathan/anaconda3/envs/srec_env_test/bin/python")   
if __name__ == "__main__":
    from time import sleep
    #testinCompressWithSReC()
    #testinDeCompressWithSReC()
    #testingEvalWithSRedc()
    #testingTimeWithSReC()
    testTraining()
