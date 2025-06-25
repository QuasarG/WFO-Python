import numpy as np
import tkinter as tk
import random
from CEC2022 import cec22_test_func
from math import pi, cos
from tkinter import ttk
from WFO import WFO
from tkinter import scrolledtext
from tkinter import Menu
from tkinter import messagebox as mbox
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import threading
from tkinter import filedialog as fd
import webbrowser
import os

from numpy import zeros,transpose
from math import inf,pi,cos
from random import random,uniform,randrange


#设置提示框
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
 
    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 30
        y = y + cy + self.widget.winfo_rooty() +3
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
 
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
 
    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def createToolTip( widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def deleteToolTip(widget):
    toolTip = ToolTip(widget)
    def always(event):
        toolTip.hidetip()
    widget.bind('<Enter>', always)
    widget.bind('<Enter>', always)


#起始界面
begintop = tk.Tk()
begintop.title('WFO')
sw = begintop.winfo_screenwidth()
sh =  begintop.winfo_screenheight()
ww=630
wh=495
x = (sw-ww)/2
y = (sh-wh)/2
begintop.geometry("%dx%d+%d+%d" %(ww,wh,x,y-50))
begintop.resizable(width=False,height=False)
try:
    begintop.iconbitmap('icon\Water flow.ico')
except:
    pass


startflag=0
ttk.Label(begintop, text="Welcome",font=('Arial',10)).place(relx=0.01,rely=0.01)
ttk.Label(begintop, text="WFO: Water Flow Optimization",font=('Times New Roman',20,'bold')).place(relx=0.20,rely=0.1)
photo = tk.PhotoImage(file="icon\image.png")
ttk.Label(begintop,image=photo,compound='center').place(relx=0.31,rely=0.22)
ttk.Label(begintop, text="Welcome use the WFO toolbox!",font=('Arial',12)).place(relx=0.325,rely=0.72)
def start():
    global startflag
    startflag=1
    begintop.destroy()
startbutt = tk.Button(begintop,text="Start",width=8,command=start,bg='yellow',font=('Arial',12,'bold')).place(relx=0.7,rely=0.7)
ttk.Label(begintop, text="@copyright Kaiping Luo, Beihang University. kaipingluo@buaa.edu.cn",font=('Arial',13)).place(relx=0.1,rely=0.82)
def ref():
    webbrowser.open('https://ieeexplore.ieee.org/document/9352564')
refbutt = tk.Button(begintop,text="Reference: Kaiping Luo. Water Flow Optimizer: a natured-inspired evolutionary algorithm for global optimization.",width=100,command=ref,fg='red',font=('Arial',8)).place(relx=0.017,rely=0.94)


begintop.mainloop()


if startflag==0:
    exit()
#构建GUI界面
top = tk.Tk()
top.title('WFO')
top.geometry("%dx%d+%d+%d" %(ww,wh,x,y-50))
top.resizable(width=False,height=False)
try:
    top.iconbitmap('icon\Water flow.ico')
except:
    pass


a=ttk.LabelFrame(top, text="WFO: Water Flow Optimizer")
a.grid(column=0, row=0, padx=8, pady=4)
b = ttk.LabelFrame(a, text="Problem's Parameters")
b.grid(column=0, row=0, padx=8, pady=4)
c = ttk.LabelFrame(b, text="Objective Function")
c.grid(column=0, row=0, padx=8, pady=4)
d = ttk.LabelFrame(b, text="Criterion")
d.grid(column=1, row=0, padx=8, pady=4)
e = ttk.LabelFrame(a, text="Algorithmic Parameters")
e.grid(column=0, row=1, padx=8, pady=4)


#Objective Function
Funclist = ["CEC2022", "Your Problem"]

def radCall():
    radSel=rad1Var.get()
    if radSel == 0:
        fitFuncEntered.delete(0,tk.END)
        fitFuncEntered.config(state='disabled')
        rad2Var.set(0) 
        rad21butt.config(state='disabled')
        rad22butt.config(state='disabled')
        enterbutt.config(state='normal')
        expandbutt.config(state='disabled')
        createToolTip(dimEntered,'2,10,20 only')
    elif radSel == 1:
        fitFuncEntered.config(state='normal')
        rad21butt.config(state='normal')
        rad22butt.config(state='normal')
        expandbutt.config(state='normal')
        fitFuncEntered.delete(0,tk.END)
        enterbutt.config(state='disabled')
        deleteToolTip(dimEntered)
 
rad1Var = tk.IntVar()
rad1Var.set(0)    
rad11butt = tk.Radiobutton(c, text=Funclist[0], variable=rad1Var, value=0, command=radCall)
rad11butt.grid(column=0, row=0,sticky='W', columnspan=3)
rad12butt = tk.Radiobutton(c, text=Funclist[1], variable=rad1Var, value=1, command=radCall)
rad12butt.grid(column=0, row=1,sticky='W', columnspan=3)


#Criterion
Critlist = ["Minimization", "Maximization",]
 
rad2Var = tk.IntVar()
rad2Var.set(0)    
rad21butt = tk.Radiobutton(d, text=Critlist[0], variable=rad2Var, value=0, command=None,state='disabled')
rad21butt.grid(column=0, row=0,sticky='W', columnspan=3)
rad22butt = tk.Radiobutton(d, text=Critlist[1], variable=rad2Var, value=1, command=None,state='disabled')
rad22butt.grid(column=0, row=1,sticky='W', columnspan=3)


#Function Label
ttk.Label(b, text="Function Label").grid(column=0, row=1,padx=5,pady=5,sticky='W')
func = tk.IntVar()
funcChosen = ttk.Combobox(b, width=4, textvariable=func)
funcChosen['values'] = (1,2,3,4,5,6,7,8,9,10,11,12)
funcChosen.place(relx=0.55,rely=0.36)
funcChosen.current(0)
funcChosen.config(state='readonly')


#enterbutt
def enter():
    try:
        num=int(funcChosen.get())
        fitFuncEntered.config(state='normal')
        fitFuncEntered.delete(0,tk.END)
        fitFuncEntered.insert(0,'cec22_test_func(x = x, nx = n, mx = 1, func_num = '+'%d'%(num)+')')
        fitFuncEntered.config(state='disabled')
    except:
        pass
    
enterbutt = ttk.Button(b,text="Enter",width=6,command=enter)
enterbutt.place(relx=0.793,rely=0.35)


#Fitness Function
ttk.Label(b, text="Fitness Function").grid(column=0, row=2,padx=5,pady=5,sticky='W')
fitFunc = tk.StringVar()
fitFuncEntered = ttk.Entry(b, width=7, textvariable=fitFunc)
fitFuncEntered.grid(column=1, row=2,padx=5,pady=5,sticky='W')
fitFuncEntered.config(state='disabled')


#entrybutt
def expand():
    exptop = tk.Tk()
    eww=500
    ewh=35
    ex = (sw-eww)/2
    ey = (sh-ewh)/2
    exptop.geometry("%dx%d+%d+%d" %(eww,ewh,ex,ey-50))
    exptop.resizable(width=False,height=False)
    exptop.attributes("-topmost",1)
    exptop.title('Fitness Function')
    try:
        exptop.iconbitmap('icon\Water flow.ico')
    except:
        pass
    expfunc = tk.StringVar()
    expfuncEntered = ttk.Entry(exptop, width=60,textvariable=expfunc)
    expfuncEntered.grid(column=0, row=0, padx=5,pady=6)
    expfuncEntered.insert(0,fitFuncEntered.get())
    def expenter():
        fitFuncEntered.delete(0,tk.END)
        fitFuncEntered.insert(0,expfuncEntered.get())
        exptop.destroy()
    expenterbutt=ttk.Button(exptop,text="Enter",width=6,command=expenter)
    expenterbutt.grid(column=1, row=0, padx=5,pady=6)
    exptop.mainloop()
expandbutt = ttk.Button(b,text="Expand",width=6,command=expand,state='disabled')
expandbutt.place(relx=0.793,rely=0.48)


#Variable Dimensions
ttk.Label(b, text="Variable Dimensions").grid(column=0, row=3,padx=5,pady=5,sticky='W')
vdim = tk.StringVar()
dimEntered = ttk.Entry(b, width=16, textvariable=vdim)
dimEntered.grid(column=1, row=3, padx=5,pady=5,sticky='E')


#Upper Bound
ttk.Label(b, text="Upper Bound").grid(column=0, row=4,padx=5,pady=5,sticky='W')
uppBound = tk.StringVar()
uppBoundEntered = ttk.Entry(b, width=16, textvariable=uppBound)
uppBoundEntered.grid(column=1, row=4, padx=5,pady=5,sticky='E')


#Lower Bound
ttk.Label(b, text="Lower Bound").grid(column=0, row=5,padx=5,pady=5,sticky='W')
lowBound = tk.StringVar()
lowBoundEntered = ttk.Entry(b, width=16, textvariable=lowBound)
lowBoundEntered.grid(column=1, row=5, padx=5,pady=5,sticky='E')


#Water Particle Numbers
ttk.Label(e, text="Water Particle Numbers").grid(column=0, row=0, padx=5,pady=5,sticky='W')
wpNum = tk.StringVar()
wpNumEntered = ttk.Entry(e, width=8, textvariable=wpNum)
wpNumEntered.grid(column=1, row=0, padx=5,pady=5,sticky='E')
wpNumEntered.insert(0, '50')


#Laminar Probability: 0<pl<1
ttk.Label(e, text="Laminar Probability: 0<pl<1").grid(column=0, row=1, padx=5,pady=5,sticky='W')
lamPro = tk.StringVar()
lamProEntered = ttk.Entry(e, width=8, textvariable=lamPro)
lamProEntered.grid(column=1, row=1, padx=5,pady=5,sticky='E')
lamProEntered.insert(0, '0.3')


#Eddying Probability: 0<pe<1
ttk.Label(e, text="Eddying Probability: 0<pe<1").grid(column=0, row=2, padx=5,pady=5,sticky='W')
eddPro = tk.StringVar()
eddProEntered = ttk.Entry(e, width=8, textvariable=eddPro)
eddProEntered.grid(column=1, row=2, padx=5,pady=5,sticky='E')
eddProEntered.insert(0, '0.7')


#Max Function Evaluations
ttk.Label(e, text="Max Function Evaluations").grid(column=0, row=3, padx=5,pady=5,sticky='W')
maxfuncEval = tk.StringVar()
maxfuncEvalEntered = ttk.Entry(e, width=8, textvariable=maxfuncEval)
maxfuncEvalEntered.grid(column=1, row=3,padx=5,pady=5, sticky='E')
maxfuncEvalEntered.insert(0, '100000')


#Infomation
ttk.Label(a, text="Infomation").place(relx=0.49,rely=0.1)
def gethelp():
    info.insert(tk.END,
'''Welcome to use Water Flow Optimizer!

If you choose "CEC2022", please select a benchmark problem from the CEC2022 suite through the dropdown menu!

If you choose "Your Problem", please input your problem in light of the folowing examples:

e.g.#1: min f=x1^2 + x2^2
where, -10<= x1, x2 <=10

Input "x[0]**2+x[1]**2" in the "Fitness Function" edit field
Input "2" in the "Variable Dimensions" edit field
Input "10" or "[10,10]" in the "Upper Bound" edit field
Input "-10" or "[-10,-10]" in the "Lower Bound" edit field

e.g.#2: min f=(x-4)^2 + (x+y)^2
s.t.
x+2y<=5
3x-y=5
-10<= x, y <=10

Input "(x[0]-4)**2+(x[0]+x[1])**2 + 1000*max(x[0]+2*x[1]-5,0)**2+1000*abs(3*x[0]-x[1]-5)" in the "Fitness Function" edit field
Input "2" in the "Variable Dimensions" edit field
Input "10" or "[10,10]" in the "Upper Bound" edit field
Input "-10" or "[-10,-10]" in the "Lower Bound" edit field

''')
helpbutt=ttk.Button(a,text="Help",width=5,command=gethelp)
helpbutt.place(relx=0.64,rely=0.09)
def empty():
    info.delete(1.0,tk.END)
emptybutt=ttk.Button(a,text="Empty",width=6,command=empty)
emptybutt.place(relx=0.743,rely=0.09)
info = scrolledtext.ScrolledText(a, width=48, height=22, wrap=tk.WORD,font=('Arial',8))
info.place(relx=0.48,rely=0.17)
info.insert(tk.INSERT,
'''Welcome to use Water Flow Optimizer!

If you choose "CEC2022", please select a benchmark problem from the CEC2022 suite through the dropdown menu!

If you choose "Your Problem", please input your problem in light of the folowing examples:

e.g.#1: min f=x1^2 + x2^2
where, -10<= x1, x2 <=10

Input "x[0]**2+x[1]**2" in the "Fitness Function" edit field
Input "2" in the "Variable Dimensions" edit field
Input "10" or "[10,10]" in the "Upper Bound" edit field
Input "-10" or "[-10,-10]" in the "Lower Bound" edit field

e.g.#2: min f=(x-4)^2 + (x+y)^2
s.t.
x+2y<=5
3x-y=5
-10<= x, y <=10

Input "(x[0]-4)**2+(x[0]+x[1])**2 + 1000*max(x[0]+2*x[1]-5,0)**2+1000*abs(3*x[0]-x[1]-5)" in the "Fitness Function" edit field
Input "2" in the "Variable Dimensions" edit field
Input "10" or "[10,10]" in the "Upper Bound" edit field
Input "-10" or "[-10,-10]" in the "Lower Bound" edit field

''')


#Tip
createToolTip(dimEntered,'2,10,20 only')
createToolTip(uppBoundEntered,'Real numbers or list only')
createToolTip(lowBoundEntered,'Real numbers or list only')
createToolTip(lamProEntered,'Real numbers in the range of (0,1) only')
createToolTip(eddProEntered,'Real numbers in the range of (0,1) only')


#进度条
bar = ttk.Progressbar(a, length=50,mode="indeterminate",orient=tk.HORIZONTAL)
bar.grid(column=3,row=0,sticky='N',padx=20,pady=20)


#构建多线程类
class MyThread(threading.Thread):
    def __init__(self, func, *args):
        super().__init__()
        
        self.func = func
        self.args = args
        
        self.setDaemon(True)
        self.start()
        
    def run(self):
        self.func(*self.args)


#Solve
def solve():
    global n,fb,xb,con,alg,prob,func,kind
    func=fitFuncEntered.get()


    #Judging file
    if rad1Var.get()==0 and os.path.exists('input_data')==False:
        mbox.showerror(title='Error', message='Please make sure the "input_data" folder is under the same content!')
        return

    
    #Judging Variable Dimensions
    if rad1Var.get()==0 and (vdim.get()=='2' or vdim.get()=='10' or vdim.get()=='20'):
        pass
    elif rad1Var.get()==1 and vdim.get().isdigit()==True and int(vdim.get())>=1:
        pass
    else:
        mbox.showerror(title='Error', message='Please enter a correct "Variable Dimensions"!')
        return
    n=int(vdim.get())
    

    #Judging Upper Bound and Lower Bound
    try:
        if type(eval(uppBound.get()))==int:
            up=[int(uppBound.get()) for i in range(n)]
        elif type(eval(uppBound.get()))==float:
            up=[float(uppBound.get()) for i in range(n)]
        elif type(eval(uppBound.get()))==list:
            if len(eval(uppBound.get()))==n:
                up=eval(uppBound.get())
            else:
                mbox.showerror(title='Error', message='"Upper Bound" doesn\'t match to the dimension!')
                return
        else:
            mbox.showerror(title='Error', message='Please enter a correct "Upper Bound"!')
            return
    except:
        mbox.showerror(title='Error', message='Please enter a correct "Upper Bound"!')
        return
    try:
        if type(eval(lowBound.get()))==int:
            low=[int(lowBound.get()) for i in range(n)]
        elif type(eval(lowBound.get()))==float:
            low=[float(lowBound.get()) for i in range(n)]
        elif type(eval(lowBound.get()))==list:
            if len(eval(lowBound.get()))==n:
                low=eval(lowBound.get())
            else:
                mbox.showerror(title='Error', message='"Lower Bound" doesn\'t match to the dimension!')
                return
        else:
            mbox.showerror(title='Error', message='Please enter a correct "Lower Bound"!')
            return
    except:
        mbox.showerror(title='Error', message='Please enter a correct "Lower Bound"!')
        return
    gap=np.array(up)-np.array(low)
    boundflag=1
    for i in gap:
        if i<0:
            boundflag=0
    if boundflag==0:
        mbox.showerror(title='Error', message='"Upper Bound" can\'t be lower than "Lower Bound"!')
        return


    #Judging Water Particle Numbers
    if wpNum.get().isdigit()==True:
        if int(wpNum.get())>=30:
            pass
        else:
            mbox.showerror(title='Error', message='"Water Particle Numbers" have better be larger than 30!')
            return
    else:
        mbox.showerror(title='Error', message='Please enter a correct "Water Particle Numbers"!')
        return


    #Judging Max Function Evaluations
    if maxfuncEval.get().isdigit()==True:
        if int(maxfuncEval.get())>=100000:
            pass
        else:
            mbox.showerror(title='Error', message='"Max Function Evaluations" have better be larger than 100000!')
            return
    else:
        mbox.showerror(title='Error', message='Please enter a correct "Max Function Evaluations"!')
        return


    #Judging Laminar Probability
    try:
        if type(eval(lamPro.get()))==float:
            if float(lamPro.get())>0 and float(lamPro.get())<1:
                pass
            else:
                mbox.showerror(title='Error', message='"Laminar Probability" must be in the range of (0,1)!')
                return
        else:
            mbox.showerror(title='Error', message='Please enter a correct "Laminar Probability"!')
            return
    except:
        mbox.showerror(title='Error', message='Please enter a correct "Laminar Probability"!')
        return


    #Judging Eddying Probability
    try:
        if type(eval(eddPro.get()))==float:
            if float(eddPro.get())>0 and float(eddPro.get())<1:
                pass
            else:
                mbox.showerror(title='Error', message='"Eddying Probability" must be in the range of (0,1)!')
                return
        else:
            mbox.showerror(title='Error', message='Please enter a correct "Eddying Probability"!')
            return
    except:
        mbox.showerror(title='Error', message='Please enter a correct "Eddying Probability"!')
        return

    #Judging Fitness Function
    if 'x[%d]'%(n-1) not in fitFunc.get() and rad1Var.get() == 1:
        mbox.showerror(title='Error', message='"Fitness Function" doesn\'t match to the dimension!')
        return

    bar.start(3)
    solvebutt.config(state='disabled')
    enterbutt.config(state='disabled')
    rad11butt.config(state='disabled')
    rad12butt.config(state='disabled')
    #Operating WFO
    class alg:
        NP=int(wpNum.get())
        max_nfe=int(maxfuncEval.get())
        pl=float(lamPro.get())
        pe=float(eddPro.get())
    class prob:
        dim=n
        lb=low
        ub=up
        def fobj(x):
            rplfunc=(fitFunc.get()).replace('^','**')
            if rad2Var.get() == 0:
                f=eval(rplfunc)
            else:
                f=eval('-('+rplfunc+')')
            return f
    try:
        fb,xb,con=WFO(alg,prob)
    except SyntaxError:
        bar.stop()
        mbox.showerror(title='Error', message='Please enter a correct "Fitness Function"!')
        solvebutt.config(state='normal')
        rad11butt.config(state='normal')
        rad12butt.config(state='normal')
        return
    except RuntimeError:
        return
    except TypeError:
        bar.stop()
        mbox.showerror(title='Error', message='"Data error! Please restart the application!')
        return

    kind=rad2Var.get()
    if kind == 0:
        info.insert(tk.INSERT,'The minimal objective function value: '+str(fb)+'\nThe best solution: '+str(xb)+'\n\n')
    else:
        info.insert(tk.INSERT,'The maximal objective function value: '+str(-fb)+'\nThe best solution: '+str(xb)+'\n\n')
    bar.stop()
    grabutt.config(state='normal')
    exrbutt.config(state='normal')
    solvebutt.config(state='normal')
    rad11butt.config(state='normal')
    rad12butt.config(state='normal')


solvebutt = ttk.Button(a,text="Solve",width=8,command=lambda:MyThread(solve))
solvebutt.grid(column=1,row=0,sticky='N',padx=10,pady=5)


#Export results
def expr():
    global alg,prob,func,kind,fb,xb
    path = fd.asksaveasfilename(filetypes = [('txt','.txt')],defaultextension = [('txt','.txt')])
    try:
        f=open(path,'w')
        f.write('Fitness Function:'+func+'\nVariable Dimensions:'+str(prob.dim)+'\nUpper Bound:'+str(prob.ub)+'\nLower Bound:'+str(prob.lb)+'\n')
        if kind == 0:
            f.write('The minimal objective function value: '+str(fb)+'\nThe best solution: '+str(xb)+'\n')
        else:
            f.write('The maximal objective function value: '+str(-fb)+'\nThe best solution: '+str(xb)+'\n')
    except:
        return

exrbutt = ttk.Button(a,text="Export results",width=15,command=expr,state='disabled')   
exrbutt.grid(column=2,row=0,sticky='N',padx=15,pady=5)


#Gragh
def creategragh():
    global con
    plt.plot(con)
    plt.xlabel('Number of function evaluation')
    plt.ylabel('Function value')
    plt.title('Convergence')
    plt.show()
grabutt = ttk.Button(a,text="Show the line gragh about the convergence",width=45,command=creategragh,state='disabled')
grabutt.place(relx=0.46,rely=0.91)


#Reference
def ref():
    webbrowser.open('https://ieeexplore.ieee.org/document/9352564')
refbutt = tk.Button(top,text="Reference: Kaiping Luo. Water Flow Optimizer: a natured-inspired evolutionary algorithm for global optimization.",width=100,command=ref,fg='red',font=('Arial',8))
refbutt.place(relx=0.017,rely=0.94)
createToolTip(refbutt,'Click to open the web')


top.mainloop()

