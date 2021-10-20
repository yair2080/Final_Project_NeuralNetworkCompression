import tkinter as tk
import os
from PIL import Image, ImageTk
from presentation.presentationHelper  import  VerticalScrolledFrame
#from presentationHelper  import  VerticalScrolledFrame



class HelpPDF(tk.Frame):
  def __init__(self, parent,name):
    from tkPDFViewer import tkPDFViewer as pdf
    import gc
    gc.collect()
    parent.title("Help_Main")
    parent.geometry("720x500")
    v1 = pdf.ShowPdf()
    helpFile=os.path.join(os.path.dirname(__file__),"help_pdf_files")
    helpFile=os.path.join(helpFile,name+".pdf")
    v2 = v1.pdf_view(parent,pdf_location = helpFile,width = 100, height = 100)
    v2.pack()


class HelpImage(tk.Frame):
  def __init__(self, parent,name):
        tk.Frame.__init__(self, parent)
        parent.title("Help")
        parent.geometry("720x500")
        self.scrollFrame = VerticalScrolledFrame(parent)
        self.pack(fill=tk.BOTH, expand=tk.TRUE)
        self.scrollFrame.pack(fill=tk.BOTH, expand=tk.TRUE)
        helpFile=os.path.join(os.path.dirname(__file__),"help_image_files")
        helpFile=os.path.join(helpFile,name+".jpg")      
        image = Image.open(helpFile)
        (width,height)=image.size
        new_height = 720 * height / width
        new_width  = new_height * width / height
        image = image.resize((int(new_width), int(new_height)), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(image)
        img = tk.Label(self.scrollFrame.interior, image=render)
        img.image = render
        img.pack()


if __name__ == "__main__":               
  root = tk.Tk()
  p1=HelpImage(parent=root,name="MainScreen")
  root.mainloop()
