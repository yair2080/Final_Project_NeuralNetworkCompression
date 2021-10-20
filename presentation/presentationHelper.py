
import os
from tkinter import filedialog,Listbox
import tkinter as tk
from tkinter import ttk

"""
presentationHelper.py 
holds functions and classes used in a  number of frames
"""


def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)

#TODO: switch listbox to something more abstract
def UploadImage(imagePaths,listbox,AllowedFiletypes):
    """[gets the paths from user added images]

    Args:
        imagePaths ([List]): [The list of image paths the form holds]
        listbox ([Listbox]): [The list box to witch the image be added to ]
        AllowedFiletypes ([tupels of tupels]): [
            the suffix of the files we want 
            example:
            ((' Portable Network Graphics files', '*.png'),('All files', '*.*'))
        ]
    """
    filenames = filedialog.askopenfilenames(filetypes=AllowedFiletypes)
    for afile in filenames:
        if(afile != ""):
            imagePaths.append(afile)
            listbox.insert(tk.END, os.path.basename(afile))

#TODO: switch listbox to something more abstract- or by type 
def UploadFolder(imagePaths,listbox,ImageType='.png'):
    """[summary]

    Args:
        imagePaths ([List]): [The list of image paths the form holds]
        listbox ([Listbox]): [The list box to witch the image be added to ]
        ImageType (str, optional): [the suffix of the files we want
            example '.txt']. Defaults to '.png'.
    """
    foldername = filedialog.askdirectory()
    if foldername == ():
        return
    filenames=list_files(foldername,ImageType)
    for afile in filenames:
        if(afile != ""):
            imagePaths.append(afile)
            listbox.insert(tk.END, os.path.basename(afile))
    return foldername  
            
            
            



class LoadingScreen(tk.Frame):
    def __init__(self, master,connection,labelName):
        import tkinter.ttk as ttk
        import threading
        self.connection=connection
        self.master = master
        self.master.geometry('350x120')
        self.frame = tk.Frame(self.master)
        self.frame.pack(anchor=tk.CENTER,expand=True,fill=tk.BOTH)
        self.label=ttk.Label(self.frame, text=labelName)
        self.label.pack(side="top",anchor=tk.CENTER,pady=5,padx=5)
        self.progress = ttk.Progressbar(self.frame, orient = tk.HORIZONTAL,
            length = 100, mode = 'indeterminate')
        self.progress.pack( fill=tk.BOTH ,pady=5,padx=5)
        ttk.Button(self.frame, text="Cancel",command= lambda : self.cancelProcess()
                   ).pack(pady=5,padx=5 )
        
        
        
        self.progressbarThread = threading.Thread(target=self.barProgress, daemon=True) 
        self.processEnded=False
        self.progressbarThread.start()
        
        
        
   
    def barProgress(self):
        import time
        import tkinter.ttk as ttk
        def isItDone():
            
            if(self.connection.poll()):
                try:
                    answer=self.connection.recv() 
                    if(answer=="DONE" or answer=="FAIL"):
                        self.processEnded=True
                        if(answer=="FAIL"):
                            import tkinter.messagebox
                            tkinter.messagebox.showerror(title="training message", message="Model problems  \n\nContact support for help")    
                        
                    else:
                            self.label['text']=answer
                except Exception:
                    pass                    
                
        try:
            while(self.processEnded == False ):
                
                for prog in range(0,100,1):
                    self.progress['value'] = prog
                    self.frame.update_idletasks()
                    time.sleep(0.1)
                    isItDone()
                    if(self.processEnded==True):
                        break
                    
                for prog in range(100,0,1):
                    self.progress['value'] = prog
                    self.frame.update_idletasks()
                    time.sleep(0.1)
                    isItDone()
                    if(self.processEnded==True):
                        break
             

                           
        #user canceld manually
        except Exception:
            self.processEnded=True
            
        self.cancelProcess()          

       
        
            
        
    def cancelProcess(self):
        #if process ended
        if(self.processEnded != True ):
            self.connection.send("canceled")
            self.connection.close()
        self.master.destroy()
        
    
    

class VerticalScrolledFrame(tk.Frame):
    # http://tkinter.unpythonic.net/wiki/VerticalScrolledFrame
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """
    #found on: https://stackoverflow.com/a/16198198/15638952
    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        canvas = tk.Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=tk.NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)