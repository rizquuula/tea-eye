GIMPmax = (360, 100, 100)
OpenCVmax = (180,255,255)

gimps = (
    (196,32,92),
    (194,35,89),
    (185,21,99),
    (182,13,100),
    (186,39,99),
    (179,31,100),
    (181,25,71),
    (182,30,89),
    (172,11,77),
    (180,21,90),
    (184,26,97)
)
for gimp in gimps:
    h = (gimp[0]/GIMPmax[0])*OpenCVmax[0]
    s = (gimp[1]/GIMPmax[1])*OpenCVmax[1]
    v = (gimp[2]/GIMPmax[2])*OpenCVmax[2]

    openCV = (int(h),int(s),int(
        
    ))
    print("for ", gimp, "\nopenCV value is = ",openCV)