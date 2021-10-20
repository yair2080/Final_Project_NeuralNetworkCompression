""" 
functions called by the GUI to operate create a procces for the model 
TODO: make more abstract to be used by all modles 
"""
import os, sys




#_--------------------------Process based callers-------------------------- 
def ProcessCompressWithL3C(pathToModels,modelName,inputPath,action,outPutPath):
        """Uses the L3C model to de/compress 
        Args:
            pathToModels (str): [A path to the directory the used model is located at]. example="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir"
            modelName (str): [the name of the trained weights we want to use ]. example ="0306_0001"
            inputPath (list): [full paths to the images to be compressed].example =["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"]
            action (str): ["enc" for using the model to compress, 
            "dec in order to use the model for compresstion "]..
            outPutPath (list): [a list of full paths for the output of each image].example:["out1.l3c","out2.l3c"]

        Returns:
            [tuple->(Process,Connection)]: connection is a bi pipe
        """
        import torch.multiprocessing as mp
        from torch.multiprocessing import set_start_method
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from L3C_operations import l3cpyEncryptDecrypt
        
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        conn1, conn2 = mp.Pipe()
        x = mp.Process(target=l3cpyEncryptDecrypt, daemon=False,
                             kwargs=dict(inputPath=inputPath,action=action,outPutPath=outPutPath,modelName=modelName,pathToModels=pathToModels,connectionToFather=conn1)) 
        x.start()
        
        sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
        return x,conn2
    
   
def ProcessEvalWithL3C(pathToModels, modelName,validationData,overwriteCatch="--overwrite_cache",recursive="--recursive=auto"):


    import torch.multiprocessing as mp
    from torch.multiprocessing import set_start_method
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from L3C_operations import l3CEvaluation
        
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    conn1, conn2 = mp.Pipe()
    x = mp.Process(target=l3CEvaluation, daemon=False,
                             kwargs=dict(pathToModels=pathToModels, modelName=modelName,validationData=validationData,overwriteCatch=overwriteCatch,recursive=recursive,connectionToFather=conn1)) 
    x.start()
        
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
    return x,conn2
        
        
def ProcessCompressDecompressTimeWithL3C(pathToModels,modelName,inputPath):
        """Uses the L3C model to de/compress 
        Args:
            pathToModels (str): [A path to the directory the used model is located at]. example="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir"
            modelName (str): [the name of the trained weights we want to use ]. example ="0306_0001"
            inputPath (list): [full paths to the images to be compressed].example =["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"]
            action (str): ["enc" for using the model to compress, 
            "dec in order to use the model for compresstion "]..
        Returns:
            [tuple->(Process,Connection)]: connection is a bi pipe
        """
        import torch.multiprocessing as mp
        from torch.multiprocessing import set_start_method
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from L3C_operations import l3cpyEncryptDecryptTime
        
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        conn1, conn2 = mp.Pipe()
        x = mp.Process(target=l3cpyEncryptDecryptTime, daemon=False,
                             kwargs=dict(inputPath=inputPath,modelName=modelName,pathToModels=pathToModels,connectionToFather=conn1)) 
        x.start()
        
        sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
        return x,conn2
 
 
 
#---------------------------subProcess based callers------------------------------------
def subProcessForL3C(environmentPath,action,argDict):
    import socket,subprocess,time,os
    from multiprocessing.connection import Connection
   
    s1, s2 = socket.socketpair()
    s1.setblocking(True)
    s2.setblocking(True)
    fileNum=os.dup(s2.fileno())
    pathToL3C_operations=os.path.join(os.path.dirname(os.path.abspath(__file__)),"L3C_operations.py")
    child = subprocess.Popen([environmentPath,pathToL3C_operations, action,str(fileNum)], pass_fds=(fileNum,))
    myConnection1 = Connection(os.dup(s1.fileno()))
    s1.close()
    s2.close()
    time.sleep(1)
    if(myConnection1.poll(1)):
        try:
            ans=myConnection1.recv()
            if(ans=="alive"):
                myConnection1.send(argDict)
                return child,myConnection1           
        except EOFError:
            pass    
    return None


def compressWithL3C(pathToModels,modelName,inputPath,action,outPutPath,config,environmentPath="/home/yonathan/anaconda3/envs/l3c_env_test/bin/python"):
    #return subProcessForL3C(environmentPath=environmentPath,action='de/compress',argList=[pathToModels,modelName,inputPath,action,outPutPath,config])
    return subProcessForL3C(environmentPath=environmentPath,action='de/compress',argDict={"pathToModels":pathToModels,"modelName":modelName,"inputPath":inputPath,"action":action,"outPutPath":outPutPath,"config":config})

    
def evalWithL3C(pathToModels, modelName,validationData,config,overwriteCatch="--overwrite_cache",recursive="--recursive=auto",environmentPath="/home/yonathan/anaconda3/envs/l3c_env_test/bin/python"):
    #return subProcessForL3C(environmentPath=environmentPath,action='eval',argList=[pathToModels,modelName,validationData,overwriteCatch,recursive])
    return subProcessForL3C(environmentPath=environmentPath,action='eval',argDict={"pathToModels":pathToModels,"modelName":modelName,"validationData":validationData,"overwriteCatch":overwriteCatch,"recursive":recursive,"config":config})


def compressDecompressTimeWithL3C(pathToModels,modelName,inputPath,config,environmentPath="/home/yonathan/anaconda3/envs/l3c_env_test/bin/python"):
    #return subProcessForL3C(environmentPath=environmentPath,action='timeTest',argList=[pathToModels,modelName,inputPath])
    return subProcessForL3C(environmentPath=environmentPath,action='timeTest',argDict={"pathToModels":pathToModels,"modelName":modelName,"inputPath":inputPath,"config":config})


def trainWithL3C(trainDataList,valiDataLIst,pathToModel,config,newModel=True,environmentPath="/home/yonathan/anaconda3/envs/l3c_env_test/bin/python"):
  
    #return subProcessForL3C(environmentPath=environmentPath,action='train',argList=[ms_config_path,trainDataList,valiDataLIst,pathToModel,newModel,batchsize_train, batchsize_val,crop_size,max_epochs, num_val_batches])
    return subProcessForL3C(environmentPath=environmentPath,action='train',
                            argDict={"trainDataList":trainDataList,"valiDataLIst":valiDataLIst,"pathToModel":pathToModel,"newModel":newModel,"config":config})




#--------------------------------testing Process based callers--------------------------
def testingCompressWithL3C ():
    if(False):
        x,conn2=ProcessCompressWithL3C(pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                        modelName="0306_0001",
                        inputPath=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"],
                        action="enc",
                        outPutPath=["out1.l3c","out2.l3c"])
        x.join()
        conn2.close()
    x,conn2=ProcessCompressWithL3C(inputPath=["/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/src/out1.l3c",
                               "/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/src/out2.l3c"],
                    action="dec"
                    ,outPutPath=["out1.png","out2.png"],
                    pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001")


def testingEvaluationWithL3C():
    #CommandLineVlaidationWithWithL3C()
    p,c=ProcessEvalWithL3C( pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                modelName="0306_0001",
                validationData="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)",
                overwriteCatch="--overwrite_cache",
                recursive="--recursive=auto")
    waitForProcessToGiveAnswer(p,c)


def testingAssit_waitForProcessToGiveAnswer(p,c):
        from time import sleep
        sleep(1)
        while(p.is_alive()):
            if(c.poll()):
                try:
                    answer=c.recv() 
                    #Raises EOFError if there is nothing left to receive and the other end was closed.
                except EOFError  as e:
                    #print(e.__traceback__)
                    break
        return answer
    
    

def testignWhoIsTheBestModel():
    bpsDict={}
    
    p,c=evalWithL3C( pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                modelName="0306_0001",
                validationData="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)",
                overwriteCatch="--overwrite_cache",
                recursive="--recursive=auto")
    
    bpsDict["0306_0001"]=testingAssit_waitForProcessToGiveAnswer(p,c)
    
    p,c=evalWithL3C( pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                modelName="0331_1256",
                validationData="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)",
                overwriteCatch="--overwrite_cache",
                recursive="--recursive=auto")
    
    bpsDict["0331_1256"]=testingAssit_waitForProcessToGiveAnswer(p,c)
    
def testingTimeWithL3C():
    #CommandLineVlaidationWithWithL3C()
    p,c=ProcessCompressDecompressTimeWithL3C( pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001",
                    inputPath=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"])
    ans=testingAssit_waitForProcessToGiveAnswer(p,c)
    print("testingTimeWithL3C: ",ans)
    
  
    
#--------------------------- testing subProcess based callers------------------------------------
def testingAssit_waitForSubProcessToGiveAnswer(p,c):
        from time import sleep
        sleep(2)
        while(p.poll()== None or c.poll()):
            if(c.poll()):
                try:
                    answer=c.recv()
                    break 
                #Raises EOFError if there is nothing left to receive and the other end was closed.
                except EOFError  as e:
                    #print(e.__traceback__)
                    break
            sleep(2)
        return answer
def testingSUBcompressWithL3C():
    from time import sleep
    p,c=compressWithL3C(pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                    modelName="0306_0001",
                    inputPath=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"],
                    action="enc",
                    outPutPath=["out1.l3c","out2.l3c"])
    assert testingAssit_waitForSubProcessToGiveAnswer(p,c) =="Done"


def testingSUBevalWithL3C():
        from time import sleep
        import numpy as np
        from L3C_operations import testConfigPrep
        p,c=evalWithL3C( pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",
                modelName="0306_0001",
                validationData="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r (1dev10)",
                overwriteCatch="--overwrite_cache",
                recursive="--recursive=auto",config=testConfigPrep())
        
        ans=testingAssit_waitForSubProcessToGiveAnswer(p,c)
        np.testing.assert_almost_equal(ans,2.736480015516281)
    
def testingSUBcompressDecompressTimeWithL3C():
    from L3C_operations import testConfigPrep
    p,c=compressDecompressTimeWithL3C(pathToModels="/home/yonathan/Documents/git/NeuralNetworkCompression/models/L3C/logDir",config=testConfigPrep(),
                    modelName="0306_0001",
                    inputPath=["/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0a7f13330a5d0023.png","/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r/0b6ef374714ee86b.png"])
    ans =testingAssit_waitForSubProcessToGiveAnswer(p,c)
    print(ans)
    
def testingTrainWithL3C():
    from L3C_operations import testConfigPrep
    pathToModel="/home/yonathan/Documents/GitHub/L3C-PyTorch_fixed/logDir"
    trainDataList="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r"
    valiDataLIst="/home/yonathan/Documents/GitHub/DataSets/val_oi_500_r"
    
    p,c=trainWithL3C(trainDataList,valiDataLIst,pathToModel,config=testConfigPrep())
    ans =testingAssit_waitForSubProcessToGiveAnswer(p,c)
    print(ans)
    
if __name__ == "__main__": 
    #note: daemon=True 
    #testingCompressWithL3C()
    #testingValidationWithL3C()
    #testignWhoIsTheBestModel()
    #testingTimeWithL3C()
    print("parent: sys.prefix - ",sys.prefix)
    #testingSUBcompressWithL3C()
    #testingSUBevalWithL3C()
    #testingSUBcompressDecompressTimeWithL3C()
    testingTrainWithL3C()
    
    