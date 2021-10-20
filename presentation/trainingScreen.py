import tkinter as tk
from tkinter import ttk 
from tkinter import filedialog,Listbox
import os
from presentation.presentationHelper  import  UploadImage,UploadFolder,VerticalScrolledFrame,LoadingScreen


class TrainingScreen(tk.Frame):
    
    def __init__(self, parent):
        
        self.trainImagePath=None
        self.validImagePath=None
        self.trainListbox=None
        self.validListbox=None

        tk.Frame.__init__(self, parent)
        self.pack(fill=tk.BOTH, expand=True)
        self.master.title("Training")
    
        self.AllowedFiletypes = [ (' Portable Network Graphics files', '*.png')]
        
                
        #upper part #relief=tk.RAISED,borderwidth=5
        upperFrame=ttk.Frame(self)
        upperFrame.pack(fill=tk.X,pady=2, padx=5)
        ttk.Button(upperFrame, text="Help",style='AccentButton',command=lambda: self.master.helpScreen(helpName=TrainingScreen.__name__)).pack(side="left")

        ##training
            #rightpart
        rightFrame1=ttk.Frame(self)
        rightFrame1.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.TRUE,anchor=tk.E)
        ttk.Label(rightFrame1, text="Training Images",font=' Times 15 bold').pack(side="top",anchor=tk.NW,pady=5 )
        subrightFrame1=ttk.Frame(rightFrame1)
        subrightFrame1.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.E,side="left")

        def getTrainFolder():
            self.trainListbox.delete(0,tk.END) 
            self.trainImagePath=UploadFolder(imagePaths=[],listbox=self.trainListbox)
        ttk.Button(subrightFrame1, text="Add Folder",
                   command=lambda:getTrainFolder()
                   ).pack(side="top",anchor=tk.NW )
        def clearTrain():
            self.trainListbox.delete(0,tk.END) 
            self.trainImagePath=None
            
        ttk.Button(subrightFrame1, text="Clear",
                   command= lambda : clearTrain()    
                   ).pack(side="top",anchor=tk.W,pady=7,padx=2 )
        #ttk.Separator(rightpartBottom1,orient=tk.HORIZONTAL).pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.S)

    
        self.trainListbox = Listbox(rightFrame1,selectmode='multiple')
        scrollbar = ttk.Scrollbar(rightFrame1)
        scrollbar.pack(side = 'right',fill=tk.BOTH) 
        self.trainListbox.pack(expand=tk.TRUE,side='left',fill=tk.BOTH, padx=5)
        self.trainListbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.trainListbox.yview)

       
        ##validtion
            #rightpart
        rightFrame2=ttk.Frame(self)
        ttk.Label(rightFrame2, text="Validation Images",font=' Times 15 bold').pack(side="top",anchor=tk.NW,pady=5 )
        rightFrame2.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.TRUE,anchor=tk.E)
        subrightFrame2=ttk.Frame(rightFrame2)
        subrightFrame2.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.E,side="left")
       
        def getValidFolder():
            self.validListbox.delete(0,tk.END) 
            self.validImagePath=UploadFolder(imagePaths=[],listbox=self.validListbox)
        ttk.Button(subrightFrame2, text="Add Folder",
                   command=lambda:  getValidFolder()
                   ).pack(side="top",anchor=tk.NW )
        
        def clearValid():
            self.validListbox.delete(0,tk.END) 
            self.validImagePath=None
        ttk.Button(subrightFrame2, text="Clear",
                   command= lambda : clearValid()
                   ).pack(side="top",anchor=tk.NW,pady=7,padx=2 )
        self.validListbox = Listbox(rightFrame2,selectmode='multiple')
        scrollbar = ttk.Scrollbar(rightFrame2)
        scrollbar.pack(side = 'right',fill=tk.BOTH) 
        self.validListbox.pack(expand=tk.TRUE,side='left',fill=tk.BOTH, padx=5)
        self.validListbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.validListbox.yview)
        ##ttk.Separator(rightFrame2,orient=tk.HORIZONTAL).pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.S,side="bottom")

        

     

        #Bottom
            
        theBottom=ttk.Frame(self)
        theBottom.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.S)
            #top bottom part 
                #top
        topBottom=ttk.Frame(theBottom,relief=tk.RAISED,borderwidth=2)
        topBottom.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.TRUE,anchor=tk.E,side="top")
        firstTopBottom=ttk.Frame(topBottom)
        firstTopBottom.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.TRUE,anchor=tk.E,side="top")
        
        self.box_value_model = tk.StringVar()
        self.box_value_config = tk.StringVar()
        modelsList=["new L3C","new SReC"]
        modelsList.extend(list(dict.keys(self.master.models.ExistingModelsDict)))
        self.model_combobox=ttk.Combobox(firstTopBottom, textvariable=self.box_value_model,values=modelsList,state='readonly')
        self.model_combobox.pack(side="left",anchor=tk.CENTER,pady=2,padx=20,expand=tk.FALSE )
        self.config_combobox=ttk.Combobox(firstTopBottom, textvariable=self.box_value_config, values=[],state='readonly')
        self.config_combobox.pack(side="right",anchor=tk.CENTER,pady=2,padx=20,expand=tk.FALSE )
        self.model_combobox.bind("<<ComboboxSelected>>",self.model_combobox_selected)
        self.box_value_model.set("Select Model")
        self.box_value_config.set("Select Configuration")
                #bottom 
        rightBottom=ttk.Frame(topBottom)
        rightBottom.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.TRUE,anchor=tk.E,side="top")
        
        vcmd = (parent.register(self.validate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        
        self.epoch_var = tk.StringVar()
        self.batchSize_var = tk.StringVar()
        ttk.Label(rightBottom, text="Epochs:").pack(side="left",anchor=tk.W,pady=2,padx=5 )
        ttk.Entry(rightBottom,width=5, validate = 'key', validatecommand = vcmd, textvariable=self.epoch_var).pack(side="left",anchor=tk.W,pady=2,padx=5,expand=tk.FALSE )  
        ttk.Label(rightBottom, text="Batchsize:").pack(side="left",anchor=tk.W,pady=2 ,padx=5 )
        ttk.Entry(rightBottom,width=5, validate = 'key', validatecommand = vcmd, textvariable=self.batchSize_var).pack(side="left",anchor=tk.W,pady=2,padx=5,expand=tk.FALSE )
        
        
        trainB=ttk.Button(rightBottom, text="Train",width=50,
                   command= lambda : self.train()
                   ).pack(side="left",anchor=tk.E,pady=2,padx=20,expand=True )
        
            #bottom bottom 
        leftBottom=ttk.Frame(theBottom)
        leftBottom.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.W,side="bottom")

 
        ttk.Button(leftBottom, text="Back",
                   command= lambda : self.master.show_frame("MainScreen")
                   ).pack(side="bottom",anchor=tk.SW,pady=2,padx=2 )
        ttk.Button(leftBottom, text="Advanced",
                   command= lambda : self.master.show_frame("ConfigurationScreen")
                   ).pack(side="bottom",anchor=tk.SW,pady=2,padx=2 )
        
    def updateModels(self):
        modelsList=["new L3C","new SReC"]
        modelsList.extend(list(dict.keys(self.master.models.ExistingModelsDict)))
        self.model_combobox['values']=modelsList
        
        
    def validate(self, action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if value_if_allowed:
            try:
                int(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False
        
    def model_combobox_selected(self,event):
        if(self.box_value_model.get().split()[0]!="new"):
            self.box_value_config.set(self.master.models.ExistingModelsDict[self.box_value_model.get()].config["Name"])
            self.config_combobox['state']='disabled'
        else: 
            model=self.master.models.availableConfigs[self.box_value_model.get().split()[1]]
            self.config_combobox['values']=list(model.keys())
            self.config_combobox['state']='readonly'
            
            
    def train(self,):
        
        # testing all required fields are filled 
        if( self.trainImagePath==None ):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="training message", message="No training images selected\n\nPress Add Files to add the files you wish to train on")
            return
        
        if(self.validImagePath==None ):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="training message", message="No validation images selected\n\nPress Add Files to add the files you wish to validate with")
            return
        
        if(self.box_value_model.get()=="Select Model"):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="training message", message="No model  selected\n\nUse the left combobox to select which model you wish to train")
            return
        
        if(self.box_value_config.get()=="Select Configuration" and self.config_combobox['state']=='readonly'):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="training message", message="No Configuration  selected\n\nUse the right combobox to select which Configuration you wish to use")
            return     
        
        if(self.epoch_var.get()=='' or int(self.epoch_var.get())==0):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="training message", message="No epoch\n\nEnter the number of epochs you wish to do in the left entry ")
            return
        
        if(self.batchSize_var.get()=='' or int(self.batchSize_var.get())==0):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="training message", message="No batch size \n\nEnter the number of batch size you wish to do in the right entry ")
            return
        
        if(self.box_value_model.get().split()[0]=="new"):
                                            #training(self, trainingPath,validPath,batchSize,epoch,selectedModel,newModel=False,config=None,name="someName"
            connection=self.master.models.training(trainingPath=self.trainImagePath,validPath=self.validImagePath,
                                                   batchSize=int(self.batchSize_var.get()),epoch=int(self.epoch_var.get()),selectedModel=self.box_value_model.get(),
                                                   config=self.box_value_config.get(),newModel=True,name="THE L3C")
        else:
            connection=self.master.models.training(self.trainImagePath,self.validImagePath,int(self.batchSize_var.get()),int(self.epoch_var.get()),self.box_value_model.get())

        #chose folder
        # true the model did nothing 
        if(connection==None):
            import tkinter.messagebox
            tkinter.messagebox.showerror(title="training message", message="Model problems  \n\nContact support for help")
            return
         
        #cleaning 
        self.master.loadingScreenForProcess(connection,"")
        self.batchSize_var.set('')
        self.epoch_var.set('')
        self.trainListbox.delete(0,tk.END) 
        self.trainImagePath=None
        self.validListbox.delete(0,tk.END) 
        self.validImagePath=None
        #TODO add cleaning


        
        
            
        
        
        

        
        
       
