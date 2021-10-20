import tkinter as tk
from tkinter import ttk
from presentation.presentationHelper  import  VerticalScrolledFrame
class ConfigurationScreen(tk.Frame):

    def __init__(self, parent):
        
        tk.Frame.__init__(self, parent)
        self.pack(fill=tk.BOTH, expand=True)
        self.master.title("Configuration")
        self.configVarDict={}

        #Upper part
        upperFrame=ttk.Frame(self)
        upperFrame.pack(fill=tk.X,pady=10, padx=5)
        ttk.Button(upperFrame, text="Help",style='AccentButton',command=lambda: self.master.helpScreen(helpName=ConfigurationScreen.__name__)).pack(side="left")
        #Center Frame Top
        centerFrameTop=ttk.Frame(self)
        centerFrameTop.pack(fill=tk.BOTH,pady=1, padx=1,expand=tk.FALSE)
        #left Center Frame Top
        leftCenterFrame=ttk.Frame(centerFrameTop)
        leftCenterFrame.pack(fill=tk.BOTH,pady=1, padx=1,expand=tk.FALSE,side ="left")
        ttk.Label(leftCenterFrame, text="Model").pack(side="top",anchor=tk.W,pady=2 ,padx=5 )
        ttk.Label(leftCenterFrame, text="Configuration").pack(side="left",anchor=tk.W,pady=2 ,padx=5 )
        #right center frame top
        rightFrameTop=ttk.Frame(centerFrameTop)
        rightFrameTop.pack(fill=tk.BOTH,pady=1, padx=1,expand=tk.FALSE,side="left")
        
        
        modelsList=[]
        modelsList.extend(list(dict.keys(self.master.models.availableConfigs)))
        self.box_value_model = tk.StringVar()
        self.model_combobox=ttk.Combobox(rightFrameTop, textvariable=self.box_value_model, values=modelsList,state='readonly')
        self.model_combobox.pack(side="top",anchor=tk.CENTER,pady=2 ,padx=5 )
        self.model_combobox.bind("<<ComboboxSelected>>",self.model_combobox_selected)
        
        self.box_value_config = tk.StringVar()
        self.config_combobox=ttk.Combobox(rightFrameTop, textvariable=self.box_value_config, values=[],state='readonly')
        self.config_combobox.pack(side="top",anchor=tk.CENTER,pady=2 ,padx=5 )
        self.config_combobox.bind("<<ComboboxSelected>>",self.config_combobox_selected)

        self.scrollFrame = VerticalScrolledFrame(self,relief=tk.RAISED,borderwidth=1)
        self.scrollFrame.pack(fill=tk.BOTH,expand=tk.TRUE,anchor=tk.CENTER,pady=5, padx=5)
        self.inFrame=tk.Frame(self.scrollFrame.interior)

     
        #bottom bottom  
        leftBottom=ttk.Frame(self)
        leftBottom.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.W,side="bottom") 
        ttk.Button(leftBottom, text="Back",
                   command= lambda : self.master.show_frame("TrainingScreen")
                   ).pack(side="bottom",anchor=tk.SW,pady=2,padx=2 )
        #bottom
        bottom=ttk.Frame(self)
        bottom.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.FALSE,anchor=tk.W,side="bottom") 
        self.saveButton=ttk.Button(bottom, text="Save",command= lambda : self.saveConfig())
        self.saveButton.pack(side="bottom",anchor=tk.CENTER,pady=2,padx=2 )
        self.saveButton["state"] = "disabled"

        
    def reset_config_combobox(self):
        configList={
            "L3C":(list(dict.keys(self.master.models.availableConfigs["L3C"]))),
            "SReC":(list(dict.keys(self.master.models.availableConfigs["SReC"])))
        }[self.box_value_model.get()]
        configList.extend(['New'])
        self.config_combobox['values']=configList
        
        
        
    def model_combobox_selected(self,event):
        self.reset_config_combobox() 
        
        
            
    def config_combobox_selected(self,event):
        self.configVarDict={}
        self.inFrame.pack_forget()
        self.inFrame=tk.Frame(self.scrollFrame.interior)
        self.inFrame.pack()
        
        configName=self.box_value_config.get()
        if(configName!="New"):
            state='disabled'
            self.saveButton["state"] = "disabled"
        else:
            state='enabled'
            self.saveButton["state"] = "enabled"

            
        if(configName=="New"):
            configName="Default"
            pass
            
        def addLineToInFrame(labelName,sub=False,masterName=""):
            inFrameLine=tk.Frame(self.inFrame)
            inFrameLine.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.TRUE)
            ttk.Label(inFrameLine, text=labelName).pack(side="left",anchor=tk.W,pady=2 ,padx=5 )
            if(sub):
                self.configVarDict[masterName][labelName] = tk.StringVar()
                self.configVarDict[masterName][labelName].set(self.master.models.availableConfigs[self.box_value_model.get()][configName][masterName][labelName])
                ttk.Entry(inFrameLine,width=7,state=state,textvariable= self.configVarDict[masterName][labelName]).pack(side="right",anchor=tk.E,pady=2,padx=5,expand=tk.FALSE )

            else:
                self.configVarDict[labelName] = tk.StringVar()
                self.configVarDict[labelName].set(self.master.models.availableConfigs[self.box_value_model.get()][configName][labelName])
                ttk.Entry(inFrameLine,width=7,state=state,textvariable= self.configVarDict[labelName]).pack(side="right",anchor=tk.E,pady=2,padx=5,expand=tk.FALSE )

            
        def addSeparatorToInFrame(labelName):
                ttk.Label(self.inFrame, text=labelName).pack(anchor=tk.W,pady=2 ,padx=5 )
                separator = ttk.Separator(self.inFrame, orient='horizontal')
                separator.pack(fill='x')
        
        def l3cSubs(config):
                mid={"ms":"ms","dl":"dl"}[str(config)]
                self.configVarDict[mid]={}
                addSeparatorToInFrame(mid)
                for msConfig in self.master.models.availableConfigs["L3C"][configName][mid]:
                    addLineToInFrame(msConfig,sub=True,masterName=mid)
                    
        helpInFrameLine=tk.Frame(self.inFrame)
        helpInFrameLine.pack(fill=tk.BOTH,pady=2, padx=5,expand=tk.TRUE ,anchor=tk.W)
        ttk.Button(helpInFrameLine, text="Help",style='AccentButton',command=lambda: self.master.helpScreen(helpName=(self.box_value_model.get()+"_config"))).pack(side="left")
           
        for config in self.master.models.availableConfigs[self.box_value_model.get()][configName]:           
            if(self.box_value_model.get()=="L3C" and str(config)!="Name" ):
                l3cSubs(config)
                continue
            addLineToInFrame(config)

    def saveConfig(self):
        
        if(((self.configVarDict["Name"]).get())in self.master.models.availableConfigs[self.box_value_model.get()] ):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="Configuration message", message="Configuration name taken \n Please write a diffrent one ")
            return
        
        configDict={}
        def l3cSubs(config):
                mid={"ms":"ms","dl":"dl"}[str(config)]
                configDict[mid]={}
                for msConfig in self.configVarDict[config]:
                     configDict[mid][msConfig]=(self.configVarDict[config][msConfig]).get()
                    
        
        for config in self.configVarDict:
            if(self.box_value_model.get()=="L3C" and str(config)!="Name" ):
                l3cSubs(config)
                continue
            configDict[config]=(self.configVarDict[config]).get()
            
        self.master.models.addNewConfig(self.box_value_model.get(),configDict)
        
        self.reset_config_combobox()