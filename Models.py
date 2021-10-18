

"""
contains classes used by the presentation layer to interact with the models 
"""


class Models():
    """ holds all models and operates them
    """
    def __init__(self):
        self.loadedModlesDict={}
        self.ExistingModelsDict={}
        self.EvalMethodsDict={"bpsp":True,
                              "Time":False}
        self.availableConfigs={}
        self.ExistingModelsDict={}
        self.addExistingModels()
        
        
    def loadModel(self,name):
        self.loadedModlesDict[name]=self.ExistingModelsDict[name]
        
    
    def removeModel(self,name):
        self.loadedModlesDict.pop(name)
    
    def evaluation(self,name,validationData):
        import threading ,queue,copy,multiprocessing
        conn1, conn2 =multiprocessing.Pipe()
        t=threading.Thread(target =self.evaluationThread,kwargs=dict(inputPaths=copy.deepcopy(validationData),connection=conn2))
        t.start()
        return conn1
        
    def evaluationThread(self,inputPaths,connection):
        answerDict={}
        if(len(self.loadedModlesDict)==0):
            connection.send("DONE") 
            return
        evalMethod=self.getEvalMethod()
        if(len(self.loadedModlesDict)>1):
            for model in self.loadedModlesDict:
                connection.send("Evaluating "+ model)
                try:
                    if(evalMethod=="bpsp"):
                        p,c= self.loadedModlesDict[model].modelEvaluate_bpsp(inputPaths)
                    elif (evalMethod=="Time"):
                        #for now only for the first image
                        #select randomly?
                        p,c= self.loadedModlesDict[model].modelEvaluate_Time([inputPaths[0]])  

                    else:
                        print(evalMethod," dose not exist")
                        continue
                except NotImplementedError:
                    print(evalMethod," dose not exist for ",model)
                    continue  
                temp=waitForSubProcessToGiveAnswer(p,c,connection,True)
                #if user canceld
                if(temp==False):
                    return
                elif(temp==None):
                    print("model crushed")
                else:
                    answerDict[model]=temp
            return min(answerDict,key=answerDict.get)
        else:
             return next(iter(self.loadedModlesDict))
        
    def getEvalMethod(self):
        for method in self.EvalMethodsDict:
            if(self.EvalMethodsDict[method]==True):
                return method
        return "bpsp"   
        
    
    def compression(self,inputPaths,outputPaths):
        import threading ,queue,copy,multiprocessing
        conn1, conn2 =multiprocessing.Pipe()
        t=threading.Thread(target =self.compressionThread,kwargs=dict(inputPaths=copy.deepcopy(inputPaths),outputPaths=copy.deepcopy(outputPaths),connection=conn2))
        t.start()
        return conn1
 
    
    def compressionThread(self,inputPaths,outputPaths,connection):

        bestModelForTheJob=self.evaluationThread(inputPaths,connection)
        connection.send("Compressing With "+ bestModelForTheJob) 
        p,c= self.loadedModlesDict[bestModelForTheJob].modelCompress(inputPaths,outputPaths)  
        temp=waitForSubProcessToGiveAnswer(p,c,connection)
        if(temp==False):
                return            
        connection.send("DONE") 
    
    
    def decompression(self,inputPaths,outputPaths):
        import threading ,queue,copy,multiprocessing
        conn1, conn2 =multiprocessing.Pipe()
        t=threading.Thread(target =self.decompressionThread,kwargs=dict(inputPaths=copy.deepcopy(inputPaths),outputPaths=copy.deepcopy(outputPaths),connection=conn2))
        t.start()
        return conn1
 
    
    def decompressionThread(self,inputPaths,outputPaths,connection):
        from time import sleep
        modelProcessDict={}
        compressingModelsDict={}
        answerDict={}
        for model in self.ExistingModelsDict:
            if not inputPaths:
                break             
            connection.send("Decompressing With "+ model) 
            p,c= self.ExistingModelsDict[model].modelDecompress(inputPaths,outputPaths) 
            if(p  !=  None and c != None): 
                temp=waitForSubProcessToGiveAnswer(p,c,connection)
                if(temp==False):
                    return
                    
        connection.send("DONE")
        
    def training(self, trainingPath,validPath,batchSize,epoch,selectedModel,newModel=False,config=None,name="someName"):
        import threading ,queue,copy,multiprocessing
        conn1, conn2 =multiprocessing.Pipe()
        if(newModel==False):
            t=threading.Thread(target =self.trainingThread,kwargs=dict(trainingPath=copy.deepcopy(trainingPath),validPath=copy.deepcopy(validPath),batchSize=copy.deepcopy(batchSize),epoch=copy.deepcopy(epoch),selectedModel=copy.deepcopy(selectedModel),connection=conn2))
        else:
            t=threading.Thread(target =self.NewModeltrainingThread,kwargs=dict(trainingPath=copy.deepcopy(trainingPath),
                                                                               validPath=copy.deepcopy(validPath),batchSize=copy.deepcopy(batchSize),
                                                                               epoch=copy.deepcopy(epoch),selectedModel=copy.deepcopy(selectedModel),
                                                                               selectedConfig=copy.deepcopy(config),name=copy.deepcopy(name),
                                                                               connection=conn2))
            
        t.start()
        return conn1
 
    
    def trainingThread(self,trainingPath,validPath,batchSize,epoch,selectedModel,connection):

        connection.send("Training With "+ selectedModel) 
        p,c= self.ExistingModelsDict[selectedModel].trainModel( trainingPath,validPath,batchSize,epoch)  
        temp=waitForSubProcessToGiveAnswer(p,c,connection,OnCancelTerminate=True)
        if(temp==False):
                return            
        connection.send("DONE")
        
    def NewModeltrainingThread(self,trainingPath,validPath,batchSize,epoch,selectedModel,selectedConfig,connection,name):
        
        import os
        def unknownModel(model):
            print("Unknown Model: " +model)
            
        def newL3C(trainingPath,validPath,batchSize,epoch,selectedConfig,selectedModel,name):       
            pathToModel=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models" )
            pathToModel=os.path.join(pathToModel, "L3C" )
            pathToModel=os.path.join(pathToModel, "logDir" )
            config=self.availableConfigs["L3C"][selectedConfig]
            p,c=L3C.trainNewModel(self, trainingPath,validPath,batchSize,epoch,pathToModel,config,pathToModel)
            temp=waitForSubProcessToGiveAnswer(p,c,connection)
            if(temp==False):
                connection.send("FAIL")  
                return 
            else:
                giveName,modelDir=temp
                newModelDir=modelDir
                #newModelDir=modelDir.replace(giveName,name)
                #os.rename(modelDir,newModelDir)
                self.modelsConfigDict["L3C"][os.path.basename(newModelDir)]=config
                connection.send("DONE")
                
       
        def newSReC(trainingPath,validPath,batchSize,epoch,selectedConfig,selectedModel,name):
            pathToModel=""
            config=self.availableConfigs["SReC"][selectedConfig]
            p,c=SReC.trainNewModel(self, trainingPath,validPath,batchSize,epoch,pathToModel,config,pathToModel)
            temp=waitForSubProcessToGiveAnswer(p,c,connection)
            if(temp==False):
                connection.send("FAIL")  
                return 
            else:
                giveName,modelDir=temp
                newModelDir=modelDir
                #newModelDir=modelDir.replace(giveName,name)
                #os.rename(modelDir,newModelDir)
                h,t=os. path. splitext(giveName)
                self.modelsConfigDict["SReC"][h]=config
                connection.send("DONE")


        def modelDirToModel(model):
            funcChooser={
                "L3C":newL3C,
                "SReC":newSReC
            }
            
            return funcChooser.get(model, unknownModel)

        connection.send("Training With "+ selectedModel) #,,,,,
        (modelDirToModel(selectedModel.split()[1]))(trainingPath=trainingPath,validPath=validPath,batchSize=batchSize,epoch=epoch,selectedConfig=selectedConfig,selectedModel=selectedModel,name=name)  

        import json
        with open("modelsConfig.json", "w") as write_file:
            json.dump( self.modelsConfigDict, write_file)
        self.addExistingModels()
          

    
    
            
    def addExistingModels(self):
        from Models import L3C
        from os import walk
        import os
        import json
        
        with open("modelsConfig.json", "r") as read_file:
            modelsConfigDict = json.load(read_file)
        
        with open("Configs.json", "r") as read_file:
            self.availableConfigs = json.load(read_file)
    
        def unknownModel(pathToModelDir):
            print("Unknown Model: " +pathToModelDir)

        def initL3C(pathToModelDir):
            pathToModelDir=os.path.join(pathToModelDir,"logDir")
            _, modelDirNames, _ = next(walk(os.path.join(pathToModelDir)))
            modelList=[]
            for model in modelDirNames:
                modelList.append(L3C(modelDirPath=pathToModelDir,name=model ,config=(modelsConfigDict["L3C"])[model]))
            return modelList
        
        def initSReC(pathToModelDir):
            pathToModelDir=os.path.join(pathToModelDir,"models")
            modelDirNames=  os.listdir((os.path.join(pathToModelDir)))
            modelList=[]
            for model in modelDirNames:
                if(model=="imagenet64.pth"): continue
                modelList.append(SReC(modelDirPath=pathToModelDir,name=model,config=(modelsConfigDict["SReC"])[os.path.splitext(model)[0]]))
            return modelList

        def modelDirToModel(modelDir):
            funcChooser={
                "L3C":initL3C,
                "SReC":initSReC
            }
            return funcChooser.get(modelDir, unknownModel)
            
   
            
        modelsDir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models" )
        _, modelDirNames, _ = next(walk(os.path.join(modelsDir)))
        modelList=[]
        
        for modelDir in modelDirNames:
            initModel=modelDirToModel(modelDir)
            modelList.extend(initModel(os.path.join(modelsDir,modelDir)))
                            
        for  model in modelList:
            self.ExistingModelsDict[model.name]=model
            
        self.modelsConfigDict=modelsConfigDict
        
    def addNewConfig(self,ModelName,config):
        if(config['Name'] in self.availableConfigs[ModelName]):
            return None
        
        def addL3C(config):
            #dl
            dlDict={}
            dlDict["batchsize_train"]  = int(config['dl']["batchsize_train"])
            dlDict["batchsize_val"]  = int(config['dl']["batchsize_val"])
            dlDict["crop_size"]  = int(config['dl']["crop_size"])
            dlDict["max_epochs"]  = None
            dlDict["image_cache_pkl"]  = None
            dlDict["train_imgs_glob"]  = str(config['dl']["train_imgs_glob"])
            dlDict["val_glob"] = str(config['dl']["val_glob"])
            dlDict["val_glob_min_size"] = None  # for cache_p
            dlDict["num_val_batches"] = int(config['dl']["num_val_batches"])            
            config['dl']=dlDict
            #ms
            msDict={}
            msDict["optim"]  = str(config['ms']["optim"])
            msDict["lr.initial"]  = float(config['ms']["lr.initial"])
            msDict["lr.schedule" ] = str(config['ms']["lr.schedule" ])
            msDict["weight_decay"]  = int(config['ms']["weight_decay"])
            msDict["num_scales"]  = int(config['ms']["num_scales"])
            msDict["shared_across_scales"]  = bool(config['ms']["shared_across_scales"])
            msDict["Cf"]  = int(config['ms']["Cf"])
            msDict["kernel_size"]  = int(config['ms']["kernel_size"])
            msDict["dmll_enable_grad"]  = int(config['ms']["dmll_enable_grad"] )
            msDict["rgb_bicubic_baseline"]  = bool(config['ms']["rgb_bicubic_baseline"])
            msDict["enc.cls"]  = str(config['ms']["enc.cls"])
            msDict["enc.num_blocks"]  = int(config['ms']["enc.num_blocks"])
            msDict["enc.feed_F"]  = bool(config['ms']["enc.feed_F"])
            msDict["enc.importance_map"]  = bool(config['ms']["enc.importance_map"])
            msDict["learned_L"]  = bool(config['ms']["learned_L"])
            msDict["dec.cls"]  = str(config['ms']["dec.cls"])
            msDict["dec.num_blocks"]  = int(config['ms']["dec.num_blocks"])
            msDict["dec.skip"]  = bool(config['ms']["dec.skip"])
            msDict["q.cls"]  = str(config['ms']["q.cls"])
            msDict["q.C"]  = int(config['ms']["q.C"])
            msDict["q.L"]  = int(config['ms']["q.L"])
            msDict["q.levels_range"]  =eval(config['ms']["q.levels_range"]) 
            msDict["q.sigma"]  = int(config['ms']["q.sigma"])
            msDict["prob.K"]  = int(config['ms']["prob.K"])
            msDict["after_q1x1"]  = bool(config['ms']["after_q1x1"])
            msDict["x4_down_in_scale0"]  = bool(config['ms']["x4_down_in_scale0"])
            config['ms']=msDict
            return config

        def addSReC(config):
            SReCDict={}
            SReCDict["Name"]=str(config["Name"])
            SReCDict["resblocks"] =int(config["resblocks"])
            SReCDict["n_feats"]  = int(config["n_feats"])
            SReCDict["scale"]  = int(config["scale"])
            SReCDict["k"] =int(config["k"])
            SReCDict["workers"]  =int(config["workers"]) 
            SReCDict["crop"]  =int(config["crop"])
            SReCDict["lr"]=float(config["lr"])
            SReCDict["eval_iters"]=float(config["eval_iters"])
            SReCDict["lr_epochs"]=int(config["lr_epochs"])
            SReCDict["plot_iters"]=int(config["plot_iters"])
            SReCDict["clip"]=float(config["clip"])
            SReCDict["gd"]=str(config["gd"])
            config=SReCDict
            return config

            
   
                
        func={
                "L3C":addL3C,
                "SReC":addSReC
                }[ModelName]
        
        config=func(config)   
        self.availableConfigs[ModelName][config['Name']]=config
        import json
        with open("Configs.json", "w") as write_file:
            json.dump( self.availableConfigs, write_file)

        
        
        
        
def checkIfConnectionAnswer(c):
    if(c.poll()):
            try:
                answer=c.recv() 
                    #Raises EOFError if there is nothing left to receive and the other end was closed.
            except EOFError:
                pass
            return answer
    return False
   
                    
                
      
def waitForProcessToGiveAnswer(modelProcess,modelConnection,GuiThreadConnection,OnCancelTerminate=False):
        from time import sleep
        answer=None
        while(modelProcess.is_alive() or  modelConnection.poll() ):
            if(modelConnection.poll()):
                try:
                    answer=modelConnection.recv() 
                    #Raises EOFError if there is nothing left to receive and the other end was closed.
                except EOFError  as e:
                    #print(e.__traceback__)
                    break
            test=checkIfConnectionAnswer(GuiThreadConnection)
            if(test=="canceled"):
                    if( OnCancelTerminate==False):
                        modelConnection.send(test)
                        return False
                    else:
                        modelConnection.close()
                        modelProcess.terminate()
                        return False
                    
            sleep(1)
        return answer
        

def waitForSubProcessToGiveAnswer(modelProcess,modelConnection,GuiThreadConnection,OnCancelTerminate=False):
        from time import sleep
        answer=None
        while(modelProcess.poll()== None or  modelConnection.poll() ):
            if(modelConnection.poll()):
                try:
                    answer=modelConnection.recv() 
                    #Raises EOFError if there is nothing left to receive and the other end was closed.
                except EOFError  as e:
                    #print(e.__traceback__)
                    break
            test=checkIfConnectionAnswer(GuiThreadConnection)
            if(test=="canceled"):
                    if( OnCancelTerminate==False):
                        try:
                            modelConnection.send(test)
                        except BrokenPipeError:
                            #add message model crushed ?
                            pass
                        return False
                    else:
                        modelConnection.close()
                        modelProcess.kill()
                        return False
                    
            sleep(1)
        return answer       
#### models####

class AbstractModel():
    
    def __init__(self,modelDirPath,name,verName,modelSuffix,config,envName):
        self.modelDirPath= modelDirPath
        self.name=name
        self.verName=verName
        self.modelSuffix=modelSuffix.replace(" ", "_")
        self.environmentPath=AbstractModel.getEnvDir(envName)
        self.config=config
         
        
    def modelCompress(self, inputPaths,outputPaths):
        raise NotImplementedError

    def modelDecompress(self, sale,inputPaths,outputPaths):
        raise NotImplementedError
    
    def trainModel(self, trainingPath,validPath,batchSize,epoch,newModel=False):
        raise NotImplementedError
    
    @staticmethod
    def trainNewModel( trainingPath,validPath,batchSize,epoch,config,newModel=True):
        raise NotImplementedError

    
    def modelEvaluate_bpsp(self,data):
        raise NotImplementedError
    
    def modelEvaluate_Time(self,data):
        raise NotImplementedError
    
    @staticmethod
    def getEnvDir(envName):
        import os
        import sys
        path=sys.prefix    
        while(1):
            h,t = os.path.split(path)
            if(t==""):
                return None
            if (t=="anaconda3"):
                break
            path=h
        
        path=os.path.join(path,"envs")
        path=os.path.join(path,envName)
        path=os.path.join(path,"bin")
        path=os.path.join(path,"python")
        return path
    

class ModelZipMock(AbstractModel):
    def __init__(self,modelDirPath,name):
        AbstractModel.__init__(self,"","zip","zip","zip")
        
    def modelCompress(self, inputPaths,outputPaths):
        import zipfile
        import os 
        """
        function : file_compress
        args : inputPaths : list of filenames to be zipped
        outputPaths : output zip file
        return : none
        assumption : Input file paths and this code is in same directory.
        """
        outputPaths=os.path.join(outputPaths, "images.zip")
        # Select the compression mode ZIP_DEFLATED for compression
        compression = zipfile.ZIP_DEFLATED
        # create the zip file first parameter path/name, second mode
        zf = zipfile.ZipFile(outputPaths, mode="w")
        try:
            for file_to_write in inputPaths:
                # Add file to the zip file
                zf.write(file_to_write, arcname=os.path.basename(file_to_write), compress_type=compression)
        except FileNotFoundError as e:
            print(f' *** Exception occurred during zip process - {e}')
        finally:
            zf.close()

    def modelDecompress(self,inputPaths,outputPaths):
        import zipfile
        for path in inputPaths:
            with zipfile.ZipFile(path, 'r') as zip_ref: 
                zip_ref.extractall(outputPaths)
    

class L3C(AbstractModel):
    
    def __init__(self,modelDirPath,name,config):
        AbstractModel.__init__(self,modelDirPath=modelDirPath, name=L3C.__name__+" "+name,verName=name,modelSuffix=".l3c"+"_"+name,config=config,envName= "l3c_env_test")


    def modelCompress(self, inputPaths,outputPaths):   #note:  inputPaths is a list of path and ouputPaths is on path to the folder were the compressed files be put        
        
            from models.L3C.src.L3C_callers import compressWithL3C
            import os
            import ntpath
            
            

            outPutPathList=[]
            inPutPathList=[]

            for targetFiles in inputPaths:
                #TODO: check all available ad 
                #testing files are competable
                if(os.path.splitext(targetFiles)[1] !=".png"):
                    continue
                
                outPutPathList.append(os.path.join(outputPaths,
                                        ntpath.basename(os.path.splitext(targetFiles)[0]) +self.modelSuffix))
                inPutPathList.append(targetFiles)
            
            #testing if there are files to compress
            if(not outPutPathList):
                return None,None    
            return compressWithL3C(inputPath=inPutPathList,outPutPath=outPutPathList,action="enc",pathToModels=self.modelDirPath,modelName=self.verName,config=self.config,environmentPath=self.environmentPath)
                
                
    def modelDecompress(self,inputPaths,outputPaths):
            from models.L3C.src.L3C_callers import compressWithL3C
            import os,ntpath
            import copy
            outPutPathList=[]
            inPutPathList=[]
            inputPathsCopy= copy.deepcopy(inputPaths)
            
            for targetFiles in inputPathsCopy:
                #testing files are competable
                if(os.path.splitext(targetFiles)[1] !=self.modelSuffix):
                    continue
                outPutPathList.append(os.path.join(outputPaths,
                                        ntpath.basename(os.path.splitext(targetFiles)[0]) + ".png"))
                inPutPathList.append(targetFiles)
                inputPaths.remove(targetFiles)
            #testing if there are files to compress
            if(not outPutPathList):
                return None,None
            return compressWithL3C(inputPath=inPutPathList,outPutPath=outPutPathList,action="dec",pathToModels=self.modelDirPath,modelName=self.verName,environmentPath=self.environmentPath,config=self.config)
    
    def modelEvaluate_bpsp(self,data):
        from models.L3C.src.L3C_callers import evalWithL3C
        return evalWithL3C(pathToModels=self.modelDirPath,modelName=self.verName,validationData=data,environmentPath=self.environmentPath,config=self.config)
    
    def modelEvaluate_Time(self,data):
        from models.L3C.src.L3C_callers import compressDecompressTimeWithL3C
        return compressDecompressTimeWithL3C(pathToModels=self.modelDirPath,modelName=self.verName,inputPath=data,environmentPath=self.environmentPath,config=self.config)
    
    def trainModel(self, trainingPath,validPath,batchSize,epoch,newModel=False):
        from models.L3C.src.L3C_callers import trainWithL3C
        import os
        newModel =False
        pathToModel=os.path.join(self.modelDirPath,self.verName)
        self.config["dl"]["max_epochs"]= epoch
        self.config["dl"]["batchsize_train"]=batchSize
        self.config["dl"]["batchsize_val"]=batchSize
        
        return trainWithL3C(trainDataList=trainingPath,valiDataLIst=validPath,pathToModel=pathToModel,newModel=newModel,
                            environmentPath=self.environmentPath,config=self.config)
        
    @staticmethod
    def trainNewModel(self, trainingPath,validPath,batchSize,epoch,pathToModel,config,modelDirPath,environmentPath="l3c_env_test",newModel=True):
        from models.L3C.src.L3C_callers import trainWithL3C
        import os
        config["dl"]["max_epochs"]= epoch
        config["dl"]["batchsize_train"]=batchSize
        config["dl"]["batchsize_val"]=batchSize
        
        return trainWithL3C(trainDataList=trainingPath,valiDataLIst=validPath,pathToModel=pathToModel,newModel=newModel,
                            environmentPath=AbstractModel.getEnvDir(environmentPath),config=config)
    
    
    
class SReC(AbstractModel):
    
    def __init__(self,modelDirPath,name,config):
        import os
        clrname=os.path.splitext(name)[0]
        AbstractModel.__init__(self,modelDirPath=modelDirPath, name=SReC.__name__+" "+clrname,verName=name,modelSuffix=".srec"+"_"+clrname,config=config,envName= "srec_env_test")
         
        
    def modelCompress(self, inputPaths,outputPaths):
            from models.SReC.SReC_callers import compressWithSReC
            import os
            import ntpath            

            inPutPathList=[]

            for targetFiles in inputPaths:
                #TODO: check all available ad 
                #testing files are competable
                if(os.path.splitext(targetFiles)[1] !=".png"):
                    continue
                inPutPathList.append(targetFiles)
            
            #testing if there are files to compress
            if(not inPutPathList):
                return None,None 
            return compressWithSReC(path=inPutPathList,save_path=outputPaths,
                                    load=os.path.join(self.modelDirPath,self.verName),suffix=self.modelSuffix,
                                    environmentPath=self.environmentPath,config=self.config)

    def modelDecompress(self,inputPaths,outputPaths):
            from models.SReC.SReC_callers import DeCompressWithSReC
            import os,ntpath
            import copy
            inPutPathList=[]
            inputPathsCopy= copy.deepcopy(inputPaths)
            
            for targetFiles in inputPathsCopy:
                #testing files are competable
                if(os.path.splitext(targetFiles)[1] != self.modelSuffix):
                    continue
                inPutPathList.append(targetFiles)
                inputPaths.remove(targetFiles)
            #testing if there are files to compress
            if(not inPutPathList):
                return None,None
            return DeCompressWithSReC(path=inPutPathList,save_path=outputPaths,load=os.path.join(self.modelDirPath,self.verName),
                                      suffix=self.modelSuffix,environmentPath=self.environmentPath,config=self.config)

    
    def modelEvaluate_bpsp(self,data):
        import os
        from models.SReC.SReC_callers import evalWithSReC
        #(,, scale = 3,k=10,crop =0,workers=2,)
        return evalWithSReC(path=data, load=os.path.join(self.modelDirPath,self.verName),environmentPath=self.environmentPath,config=self.config)
    
    def modelEvaluate_Time(self,data):
        import os
        from models.SReC.SReC_callers import timeTestSReC
        
        return timeTestSReC(path=data, load=os.path.join(self.modelDirPath,self.verName),environmentPath=self.environmentPath,config=self.config)       
    
    
    def trainModel(self, trainingPath,validPath,batchSize,epoch,newModel=False):
        from models.SReC.SReC_callers import trainSReC
        import os
        return trainSReC(train_path=trainingPath,eval_path=validPath,batch=batchSize,
                         load=os.path.join(self.modelDirPath,self.verName),newModel=False,config=self.config,epochs=epoch,environmentPath=self.environmentPath)
        
    @staticmethod
    def trainNewModel(self, trainingPath,validPath,batchSize,epoch,pathToModel,config,modelDirPath,environmentPath="srec_env_test",newModel=True):
        from models.SReC.SReC_callers import trainSReC
        import os
        
        return trainSReC(train_path=trainingPath,eval_path=validPath,batch=batchSize,load=pathToModel,newModel=newModel,config=config,epochs=epoch,
                         environmentPath=AbstractModel.getEnvDir(environmentPath))
