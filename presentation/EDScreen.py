import tkinter as tk
from tkinter import ttk 
from tkinter import filedialog,Listbox
import os
from presentation.presentationHelper  import  UploadImage,UploadFolder,VerticalScrolledFrame,LoadingScreen


class EDScreen(tk.Frame):
    
    def __init__(self, parent):
        
        self.imagePaths=[]
        self.listbox=None
        tk.Frame.__init__(self, parent)
        self.pack(fill=tk.BOTH, expand=True)
        self.master.title("Compress/Decompress")
    
        self.AllowedFiletypes = [ (' Portable Network Graphics files', '*.png')]
        
        for key in self.master.models.ExistingModelsDict:
            self.AllowedFiletypes.append(tuple([key,self.master.models.ExistingModelsDict[key].modelSuffix +"*"]))
            
        self.AllowedFiletypes=tuple(self.AllowedFiletypes)
            

            
        #upper part #relief=tk.RAISED,borderwidth=5
        upperFrame=ttk.Frame(self)
        upperFrame.pack(fill=tk.X,pady=10, padx=5)
        ttk.Button(upperFrame, text="Help",style='AccentButton',command=lambda: self.master.helpScreen(helpName=EDScreen.__name__)).pack(side="left")
        ttk.Label(upperFrame, text="Added Files").pack(side="bottom",anchor=tk.CENTER)
        #Left part 
        leftFrame=ttk.Frame(self)
        leftFrame.pack(fill=tk.BOTH,pady=1, padx=1,anchor=tk.W ,side= 'left')
        ttk.Button(leftFrame, text="Add Folder",
                   command=lambda: UploadFolder(imagePaths=self.imagePaths,listbox=self.listbox)
                   ).pack(side="top",anchor=tk.CENTER,pady=5 ,padx=5,fill=tk.X)
        ttk.Button(leftFrame, text="Add File",
                   command=lambda: UploadImage(imagePaths=self.imagePaths,listbox=self.listbox,AllowedFiletypes=self.AllowedFiletypes)
                   ).pack(side="top",anchor=tk.CENTER,pady=5 ,padx=5,fill=tk.X  )
        ttk.Button(leftFrame, text="Compress",
                   command= lambda: self.compress()
                   ).pack(side="top",anchor=tk.CENTER,pady=5,padx=5,fill=tk.X )
        ttk.Button(leftFrame, text="Decompress",
                   command=lambda: self.decompress()
                   ).pack(side="top",anchor=tk.CENTER,pady=5,padx=5,fill=tk.X )
        ttk.Button(leftFrame, text="Back",command= lambda : self.master.show_frame("MainScreen")).pack(side="bottom",anchor=tk.SW,pady=5,padx=5 ) 
        ttk.Button(leftFrame, text="Advanced",
                   command= lambda : self.master.show_frame("EDAdvanced")
                   ).pack(side="bottom",anchor=tk.SW,pady=5,padx=5 )
        
        #rightpart
        rightFrame=ttk.Frame(self)
        rightFrame.pack(fill=tk.BOTH,pady=5, padx=5,expand=tk.TRUE,anchor=tk.E)
        self.listbox = Listbox(rightFrame,selectmode='multiple')
        scrollbar = ttk.Scrollbar(rightFrame)
        scrollbar.pack(side = 'right',fill=tk.BOTH) 
        self.listbox.pack(expand=tk.TRUE,side='left',fill=tk.BOTH)
        self.listbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.listbox.yview)

        #rightpartBottom
        rightpartBottom=ttk.Frame(self)
        rightpartBottom.pack(fill=tk.BOTH,pady=5, padx=5,expand=tk.FALSE,anchor=tk.S)
        ttk.Button(rightpartBottom, text="Delete",
                   command= lambda : self.deleteItemFromList()
                   ).pack(side="bottom",anchor=tk.S,pady=2,padx=2 )
        
        
    def compress(self):
        
        if( not self.imagePaths ):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="compression message", message="no files selected\n\nPress Add Files to add the files you wish to de/compress")
            return
        
        if( not self.master.models.loadedModlesDict ):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="compression message", message="No model selected \nGo to the Advanced screen to select one")
            return
        
        
        
        foldername = filedialog.askdirectory()
        #true if the user canceled
        if(foldername == () or foldername ==""):
            return

        
        #chose folder
        connection=self.master.models.compression(self.imagePaths,foldername)
        # true the model did nothing 
        if(connection==None):
            return


        
        self.master.loadingScreenForProcess(connection,"")
        self.listbox.delete(0,tk.END)
        self.imagePaths.clear()
        
    def decompress(self):
        
        if( not self.imagePaths ):
            import tkinter.messagebox
            tkinter.messagebox.showinfo(title="decompression message", message="no files selected")
            return
        
        foldername = filedialog.askdirectory()
        #true if the user canceled
        if(foldername == () or foldername ==""):
            return
        #chose folder
        connection=self.master.models.decompression(self.imagePaths,foldername)
        # true the model did nothing 
        if(connection==None):
            return

        self.master.loadingScreenForProcess(connection,"")
        self.listbox.delete(0,tk.END)
        self.imagePaths.clear()
        
    def deleteItemFromList(self):
        from copy import deepcopy
        import os,ntpath
        sel = self.listbox.curselection()
        paths=self.imagePaths[::-1]
        for index in sel[::-1]:
            removedItem = self.listbox.get(index)
            self.listbox.delete(index)
            tempImagePath=deepcopy(self.imagePaths)
            for path in self.imagePaths:   
                name=ntpath.basename(path)
                if(name == removedItem):
                    tempImagePath.remove(path)
                    break
            self.imagePaths=deepcopy(tempImagePath)
           
class EDAdvanced(tk.Frame):

    def __init__(self, parent):
        
        tk.Frame.__init__(self, parent)
        self.pack(fill=tk.BOTH, expand=True)
        self.master.title("Compress/Decompress-Advanced")
            
        #Upper part
        upperFrame=ttk.Frame(self)
        upperFrame.pack(fill=tk.X,pady=10, padx=5)
        ttk.Button(upperFrame, text="Help",style='AccentButton',command=lambda: self.master.helpScreen(helpName=EDAdvanced.__name__)).pack(side="left")
        #Center Frame
        centerFrame=ttk.Frame(self)
        centerFrame.pack(fill=tk.BOTH,pady=1, padx=1,expand=tk.TRUE)
        # Center Left part 
        leftFrame=ttk.Frame(centerFrame)
        leftFrame.pack(fill=tk.BOTH,pady=1, padx=1,expand=tk.TRUE,anchor=tk.W ,side= 'left')
        subLeftFrame=ttk.Frame(leftFrame)
        subLeftFrame.pack(fill=tk.BOTH,pady=25, padx=15,expand=tk.TRUE,anchor=tk.W ,side= 'left')
        ttk.Label(subLeftFrame, text="Select model:").pack(side="top",anchor=tk.W)
        leftScrollFrame = VerticalScrolledFrame(subLeftFrame,relief=tk.RAISED,borderwidth=1)
        leftScrollFrame.pack(fill=tk.BOTH,expand=tk.TRUE,anchor=tk.CENTER,pady=5, padx=5)
            #loading the models
        self.checkButtonsDict={}
        for i in self.master.models.ExistingModelsDict:
            chkValue = tk.BooleanVar() 
            tempButton=ttk.Checkbutton(leftScrollFrame.interior, text=i,variable=chkValue,takefocus = 0)
            tempButton.pack(anchor=tk.W)  
            if  i in self.master.models.loadedModlesDict:
                chkValue.set(True)
            else:
                chkValue.set(False)
            self.checkButtonsDict[i]=chkValue                          
        # Center Right part
        rightFrame=ttk.Frame(centerFrame)
        rightFrame.pack(fill=tk.BOTH,pady=1, padx=1,expand=tk.TRUE,anchor=tk.E,side= 'right')
        subRightFrame=ttk.Frame(rightFrame)
        subRightFrame.pack(fill=tk.BOTH,pady=25, padx=15,expand=tk.TRUE,anchor=tk.E ,side= 'right')
        ttk.Label(subRightFrame, text="Choose evaluation:").pack(side="top",anchor=tk.W)
        rightScrollFrame = VerticalScrolledFrame(subRightFrame,relief=tk.RAISED,borderwidth=1)
        rightScrollFrame.pack(fill=tk.BOTH,expand=tk.TRUE,anchor=tk.CENTER,side='bottom',pady=5, padx=5)
        self.radioVal = tk.StringVar()
            #loading evaluations 
        self.radioVal.set("Time")
        for i in self.master.models.EvalMethodsDict:   
            radioButton=ttk.Radiobutton(rightScrollFrame.interior, text=i,value=i,variable=self.radioVal)
            radioButton.pack( anchor = tk.W)
            if self.master.models.EvalMethodsDict[i]==True:
                self.radioVal.set(i)
                #radioButton.select()

        #Bottom part
        bottomFrame=ttk.Frame(self)
        bottomFrame.pack(fill=tk.BOTH,pady=10, padx=5, side ='bottom',anchor=tk.S)
        ttk.Button(bottomFrame, text="Back",command= lambda : self.backBottonPressed()).pack(side="left")
                
    def backBottonPressed(self):
        self.master.show_frame("EDScreen")
        #Updating chosen evalMethod
        for i in self.master.models.EvalMethodsDict:
            if i == self.radioVal.get():
                self.master.models.EvalMethodsDict[i]=True
            else:
                self.master.models.EvalMethodsDict[i]=False
                
        #Updating chosen models
        for i in self.checkButtonsDict:
            if self.checkButtonsDict[i].get() and  i not in self.master.models.loadedModlesDict:
                self.master.models.loadModel(i)
            elif  not self.checkButtonsDict[i].get() and i in self.master.models.loadedModlesDict:
                self.master.models.removeModel(i)
                
