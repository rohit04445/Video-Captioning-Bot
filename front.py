from tkinter import *
import time
from tkinter.filedialog import askopenfile 
from pred import *
from utils import *
import pyttsx3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import threading
import warnings
from PIL import Image

class VideoCaptioning(Tk):
	def __init__(self):
		super().__init__()
		self.geometry("700x600")
		self.resizable(False, False)
		self.threadWork=None
		self.clg_name="Dr B R Ambedkar \nNational Institute of Technology, Jalandhar"
		self.count=0
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.predict=pred()

	def changetext(self):
		self.ans = self.ans.replace('<BOS>','')
		self.ans = self.ans.replace('<EOS>','')
		self.ans=self.ans.upper()
		self.ans=self.ans.replace('\n',".\n")


	def Labels(self):
		self.Background_Image=PhotoImage(file='./GUI FOLDER/nit.png')
		self.Background_Image_Label=Label(self,image=self.Background_Image)
		self.Background_Image_Label.place(x=0,y=0,relwidth=1,relheight=1)
		self.logo_image=PhotoImage(file='./GUI FOLDER/logo.png')

		self.canvas=Canvas(self,bg='white', width=500, height=400,border=0)
		self.canvas.place(x=100,y=100)
		self.logo_image_Label=Label(self,image=self.logo_image,border=0)
		self.logo_image_Label.place(x=100,y=100)
		self.clg_name_logo=Label(self,bg='white',text=self.clg_name,font='bold, 15')
		self.clg_name_logo.place(x=205,y=110)
		self.canvas.create_line(0,130,500,130, fill="black", width=3)
		self.Title=Label(self,bg='white',text='Video Captioning',font='bold, 30') 
		self.Title.place(x=200,y=190)
        
		self.Upload_Label=Label(self,bg='white',text='Upload the video to be captioned',font='8')
		self.Upload_Label.place(x=200,y=250)
		self.Video_File=Label(self,bg='white',text="",font="8")           
		self.Video_File.place(x=230,y=270)
		self.Output_Label=Label(self,bg="white",text="",font="bold, 10")
		self.Output_Label.place(x=160,y=350)
		self.Button=Button(self,text=" CHOOSE FILE ",font="bold, 7",command=self.open_file,bg="#0966c3",fg="white",activebackground='#0d3267',border=0)
		self.Button.place(x=450,y=255)     
		self.Button=Button(self,text="Start Captioning",font="bold, 15",bg="#0966c3",activebackground='#0d3267',fg="white",border=0,command=self.thrun)
		self.Button.place(x=280,y=300)

	def pred(self,file_name):
		start = time.perf_counter()
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.ans=self.predict.predict(file_name.name)
		end = time.perf_counter()
		print(f'Finished in {round(end-start, 2)} second(s)') 


	def show(self):
		playVideoFromPath(self.file_name.name)

	def speak(self):
		engine = pyttsx3.init()
		engine.say(self.ans)
		engine.runAndWait()

	def find_output(self):
		self.count+=1
		self.Output_Label.config(text="Captioning.....")
		self.pred(self.file_name)
		self.changetext()
		self.Output_Label.config(text=("Caption: "+self.ans))
		process1 = threading.Thread(target=self.speak,name=str(self.count))
		process2 = threading.Thread(target=self.show,name=str(self.count))
		process1.start()
		process2.start()
		process1.join()
		process2.join()
		print(self.ans)
        

	def thrun(self):
			self.threadWork=threading.Thread(target=self.find_output,name=str(self.count))
			self.threadWork.start()


	def open_file(self):
		self.file_name = askopenfile(mode='r', filetypes=[('all files', '.*')])
		print(self.file_name)  
		self.Output_Label.config(text="")
		self.Video_File.config(text=os.path.basename(self.file_name.name))

if __name__=="__main__":
	window=VideoCaptioning()
	window.Labels()
	window.mainloop()