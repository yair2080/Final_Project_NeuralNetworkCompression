import tkinter as tk
from tkinter import ttk 
import tkinter.font as tkFont

class MainScreen(tk.Frame):

    def __init__(self, parent):
        
        tk.Frame.__init__(self, parent)
        self.pack(fill=tk.BOTH, expand=True)
        self.master.title("Main")
        #upper part
        upperFrame=ttk.Frame(self)
        upperFrame.pack(fill=tk.X,pady=10, padx=5)
        ttk.Button(upperFrame, text="Help",style='AccentButton',command=lambda: self.master.helpScreen(helpName=MainScreen.__name__)).pack(side="left")
        #space
        lowerFrame=ttk.Frame(self,height = 70)
        lowerFrame.pack(fill=tk.BOTH,pady=5, padx=5)
        label = tk.Label(lowerFrame, text="Neural networks\n for lossy and lossless\n data compression", font=' Times 18 bold')
        label.pack(anchor=tk.CENTER,pady=10, padx=5)

        #lower part
        lowerFrame1=ttk.Frame(self)
        lowerFrame1.pack(fill=tk.X,pady=5, padx=5)
        ttk.Button(lowerFrame1, text="Compress/Decompress",command= lambda : self.master.show_frame("EDScreen")).pack(side="top")

        lowerFrame2=ttk.Frame(self)
        lowerFrame2.pack(fill=tk.X,pady=5, padx=5)
        ttk.Button(lowerFrame2, text="Train",command= lambda : self.master.show_frame("TrainingScreen")).pack(side="top")

        lowerFrame3=ttk.Frame(self)
        lowerFrame3.pack(fill=tk.X,pady=5, padx=5)
        ttk.Button(lowerFrame3,text="Exit",command=parent.destroy).pack(side="top")
        

   



if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(tk.FALSE,tk.FALSE)
    root.geometry('330x420')
    MainScreen(root,None)
    root.mainloop()