"""the lunching point for the application 
"""
#tkinter imports
import tkinter as tk 
from tkinter import ttk 
#presentation imports
from presentation.MainScreen import MainScreen
from presentation.EDScreen import EDScreen, EDAdvanced  
from presentation.trainingScreen import  TrainingScreen
from presentation.configurationScreen import ConfigurationScreen

from Models import Models
class FrameMaster(tk.Tk):

    def __init__(self, *args, **kwargs):
        #config window  
        tk.Tk.__init__(self, *args, **kwargs)
        self.resizable(tk.FALSE,tk.FALSE)
        self.geometry('330x420')
        self.models=Models()
        

        style = ttk.Style(self)
        # Import the tcl file
        import os
        azurePath=os.path.dirname(os.path.abspath(__file__))
        azurePath=os.path.join(azurePath,"presentation")
        azurePath=os.path.join(azurePath,"Azure-ttk-theme")
        azurePath=os.path.join(azurePath,"azure.tcl")
        self.tk.call('source', azurePath)
        style.theme_use('azure')
           
        #adding the frames
        self.frames = {}
        for F in (MainScreen,EDScreen,EDAdvanced,TrainingScreen,ConfigurationScreen):
            page_name = F.__name__
            self.frames[page_name] = F
        
        #main
        frame = self.frames["MainScreen"]
        self.currentFrame=frame(parent=self)       


    def show_frame(self, page_name):
        self.currentFrame.pack_forget()
        frame = self.frames[page_name]
        if page_name == "MainScreen":
           self.geometry('330x420')
           self.currentFrame=frame(parent=self) 
        else:
             self.geometry('450x600')
             self.currentFrame=frame(parent=self)
             
    def loadingScreenForProcess(self,connection,labelName):
        
        from presentation.presentationHelper import LoadingScreen
        self.newWindow = tk.Toplevel(self)
        LoadingScreen(self.newWindow,connection,labelName)
        
    def helpScreen(self,helpName): 
        from presentation.Help import HelpImage
        self.newHelperWindow = tk.Toplevel(self)
        HelpImage(self.newHelperWindow,helpName).pack_forget()
        
        
        


if __name__ == "__main__":
                
    app = FrameMaster()
    app.mainloop()