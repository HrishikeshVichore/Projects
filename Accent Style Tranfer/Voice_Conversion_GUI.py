import tkinter as tk
import os
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Toplevel
from subprocess import Popen, CREATE_NEW_CONSOLE
from queue import Queue
import sounddevice as sd
import soundfile as sf
import threading
from datetime import datetime
from tkinter import Radiobutton


class Analyser:
    def __init__(self,parent):
        
        self.root = parent
        self.root.title("Speech-to-Speech Conversion")
        
        self.frame = tk.Frame(parent)
        self.frame.grid()
        
        tk.Label(self.frame, text = 'Content').grid(row = 1 , column = 0)
        tk.Label(self.frame, text = 'Style').grid(row = 2 , column = 0)
        
        self.content_dir = tk.StringVar()
        self.style_dir = tk.StringVar() 
        
        self.content_dir.set("D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Recording.wav")
        self.style_dir.set("D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Audio_Data_Wav/us_english/abbreviation.wav")
        
        self.cname=tk.Entry(self.frame, state=tk.DISABLED, textvariable=self.content_dir, width=100)
        self.cname.grid(row = 1 , column = 1)
        
        self.sname=tk.Entry(self.frame, state=tk.DISABLED, textvariable=self.style_dir, width=100)
        self.sname.grid(row = 2 , column = 1)
        
        self.cbutton = tk.Button(self.frame,text = "Browse...", command = lambda: self.new('content'))
        self.sbutton = tk.Button(self.frame,text = "Browse...", command = lambda: self.new('style'))
        
        self.crbutton = tk.Button(self.frame,text = "Record", command = lambda: self.new_window('content'))
        self.srbutton = tk.Button(self.frame,text = "Record", command = lambda: self.new_window('style'))
        
        self.cbutton.grid(row = 1 , column = 2)
        self.sbutton.grid(row = 2 , column = 2)
        
        self.crbutton.grid(row = 1 , column = 3)
        self.srbutton.grid(row = 2 , column = 3)
        
        self.convert = tk.Button(self.frame, text = 'Run...', command=self.run)
        self.convert.grid(row = 3 , column = 3)
        
        self.moreOptions = tk.Button(self.frame,text = "More Options", command = self.options)
        self.moreOptions.grid(row = 3 , column = 0)        
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        
    def new(self,button_name):
        filetypes = (
        ('Wav files', '*.wav'),
        ('MP3 files', '*.mp3'),
        ('All files', '*.*')
        )
        file_path = filedialog.askopenfilename(title='Open a file',filetypes=filetypes)
        if 'content' in button_name:
            self.content_dir.set(file_path)
        elif 'style' in button_name:
            self.style_dir.set(file_path)
    
    def run(self, options = False):
        content = self.content_dir.get()
        style = self.style_dir.get()
        
        if options:
            kernels = ['Random', 'Content', 'Style']
            kernel = kernels[self.var.get()]
            content_weight = self.cw.get()
            style_weight = self.sw.get()
            lr = self.lr.get()
            epochs = self.epochs.get()
            print_interval = self.pi.get()
            output = f'{os.path.dirname(self.content_dir.get())}/{self.output_filename.get()}{self.file_type}'
            cmd = ['python', 'torch_train_gui.py', "-content", content, "-style", style, "-content_weight", content_weight, "-style_weight", style_weight, '-learning_rate', lr, '-epochs', epochs, '-print_interval', print_interval, '-kernel', kernel, '-output', output]
        else:
            cmd = ['python','torch_train_gui.py', "-content", content, "-style", style]
        #cmd = python torch_train_gui.py -content "D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Recording.wav" -style "D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Audio_Data_Wav/us_english/abbreviation.wav" -content_weight 100.0 -style_weight 1 -learning_rate 0 -epochs 20000 -print_interval 1000 -kernel "Content" -output "D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Recording_MyNet.wav"
        print(cmd)   
        Popen(cmd, creationflags=CREATE_NEW_CONSOLE)
        
    
    def new_window(self, button_name):
        self.root.withdraw()
        
        voice_rec = Toplevel(self.root)
        voice_rec.title("Recording Window")
        voice_rec.config(bg="#107dc2")
        
        self.q = Queue()
        
        self.recording = False
        self.file_exists = False
        self.filename = f'Recordings/Recording_{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}.wav'
        
        if 'content' in button_name:
            self.content_dir.set(self.filename)
        elif 'style' in button_name:
            self.style_dir.set(self.filename)
        
        tk.Label(voice_rec, text="Voice Recorder", bg="#107dc2").grid(row=0, column=0, columnspan=4)
        
        record_btn = tk.Button(voice_rec, text="Record Audio", command=lambda m=1:self.threading_rec(m))
        stop_btn = tk.Button(voice_rec, text="Stop Recording", command=lambda m=2:self.threading_rec(m))
        play_btn = tk.Button(voice_rec, text="Play Recording", command=lambda m=3:self.threading_rec(m))
        del_btn = tk.Button(voice_rec, text="Delete Recording", command=lambda m=4:self.threading_rec(m))
        back_btn = tk.Button(voice_rec, text="Go Back", command=lambda window=voice_rec:self.goBack(window))
        
        record_btn.grid(row=1,column=1)
        stop_btn.grid(row=1,column=0)
        play_btn.grid(row=1,column=2)
        del_btn.grid(row=1,column=3)
        back_btn.grid(row=1,column=4)
        
        voice_rec.rowconfigure(0, weight=1)
        voice_rec.columnconfigure(0, weight=1)
        voice_rec.mainloop()
    
    def record_audio(self):
        self.recording= True  
        #Create a file to save the audio
        messagebox.showinfo(message="Recording Audio. Speak into the mic")
        if not os.path.isdir('Recordings'):
            os.mkdir('Recordings')
        with sf.SoundFile(self.filename, mode='w', samplerate=44100,
                            channels=2) as file:
        #Create an input stream to record audio without a preset time
                with sd.InputStream(samplerate=44100, channels=2, callback=self.callback):
                    while self.recording == True:
                        #Set the variable to True to allow playing the audio later
                        self.file_exists =True
                        #write into file
                        file.write(self.q.get())
    
    def threading_rec(self,x):
        if x == 1:
            t1=threading.Thread(target= self.record_audio)
            t1.start()
        elif x == 2:
            self.recording = False
            messagebox.showinfo(message="Recording finished")
        elif x == 3:
            if self.file_exists:
                data, fs = sf.read(self.filename, dtype='float32')
                sd.play(data,fs)
                sd.wait()
            else:
                messagebox.showerror(message="Record something to play")
        elif x == 4:
            if self.file_exists:
                os.remove(self.filename)
                self.file_exists=False
                messagebox.showinfo(message=f"Deleted the recording {self.filename}")
            else:
                messagebox.showerror(message="Nothing to delete")
                
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())
    
    def goBack(self, window):
        window.destroy()
        self.root.iconify()
        self.root.deiconify()
    
    def options(self):
        self.convert.grid_forget()
        
        tk.Label(self.frame, text = 'Content Weight').grid(row = 4 , column = 0)
        tk.Label(self.frame, text = 'Style Weight').grid(row = 5 , column = 0)
        tk.Label(self.frame, text = 'Learning Rate').grid(row = 6 , column = 0)
        tk.Label(self.frame, text = 'Print Interval').grid(row = 7 , column = 0)
        tk.Label(self.frame, text = 'Epochs').grid(row = 8 , column = 0)
        tk.Label(self.frame, text = 'Output Filename').grid(row = 9 , column = 0)
        tk.Label(self.frame, text = 'Choose a Kernel:').grid(row = 10 , column = 0)
        tk.Label(self.frame, text = '0 for dynamic lr').grid(row = 6 , column = 2)
        
        self.cw=tk.Entry(self.frame, width=100)
        self.cw.grid(row = 4 , column = 1)
        self.cw.insert(0,str(1e2))
        
        self.sw=tk.Entry(self.frame, width=100)
        self.sw.grid(row = 5 , column = 1)
        self.sw.insert(0,str(1))
        
        self.lr=tk.Entry(self.frame, width=100)
        self.lr.grid(row = 6 , column = 1)
        self.lr.insert(0,str(0.002))
        
        self.pi=tk.Entry(self.frame, width=100)
        self.pi.grid(row = 7 , column = 1)
        self.pi.insert(0,str(1000))
        
        self.epochs=tk.Entry(self.frame, width=100)
        self.epochs.grid(row = 8 , column = 1)
        self.epochs.insert(0,str(20000))
        
        self.output_filename=tk.Entry(self.frame, width=100)
        self.output_filename.grid(row = 9 , column = 1)
        name, self.file_type = os.path.splitext(os.path.basename(self.content_dir.get()))
        self.output_filename.insert(0,name)
        
        self.var = tk.IntVar()
        self.var.set(1)
        R1 = Radiobutton(self.frame, text="Random", variable=self.var, value=0)
        R1.grid(row = 10, column = 1)
        
        R2 = Radiobutton(self.frame, text="Content", variable=self.var, value=1)
        R2.grid(row = 10, column = 2)
        
        R3 = Radiobutton(self.frame, text="Style", variable=self.var, value=2)
        R3.grid(row = 10, column = 3)
        
        tk.Button(self.frame, text = 'Default...', command = self.options).grid(row = 11, column = 0)
                
        self.convert.configure(command=lambda options=True:self.run(options))
        self.convert.grid(row = 11, column = 3)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = Analyser(root)
    root.mainloop()