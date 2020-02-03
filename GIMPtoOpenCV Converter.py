GIMPmax = (360, 100, 100)
OpenCVmax = (180,255,255)

gimp = (193,32,100)

h = (gimp[0]/GIMPmax[0])*OpenCVmax[0]
s = (gimp[1]/GIMPmax[1])*OpenCVmax[1]
v = (gimp[2]/GIMPmax[2])*OpenCVmax[2]

openCV = (h,s,v)
print("for ", gimp, "\nopenCV value is = ",openCV)