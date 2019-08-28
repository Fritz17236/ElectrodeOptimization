# Import Statements

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import scipy.fftpack
import scipy.io
from scipy import ndimage
import scipy.optimize.nnls as nnls
import sklearn
import scipy.cluster.vq as spc
from matplotlib import cm
import cvxpy as cp
import copy
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
import datetime



#  Class Declarations 
class StimSweepData:
        
        
    def __init__(self):
        self.Ts        = None 
        self.mseImgSet = None 
        self.wmsImgSet = None
        self.ssmImgSet = None
         
        self.mseActSet = None
        self.wmsActSet = None
        self.ssmActSet = None
         
        self.mseRecSet = None
        self.wmsRecSet = None
        self.ssmRecSet = None
class ImageData:
    def __init__(self):
        self.numImgs    = None
        self.imgSet     = None
        self.filtImgSet = None
        self.xs         = None
        self.ys         = None
        self.zoomFac    = None
        self.origImg    = None
        self.selecDims  = None

def metricCompar(imgData,simParams,psychParams, electrode):
    # Compare Error Metrics Side-by-Side for the same set of images    
    img    = imgData.origImg
    imgSet = imgData.imgSet
    fltSet = imgData.filtImgSet
    xs     = imgData.xs
    ys     = imgData.ys
    
    if electrode:
        print('Solving for Electrode Activities...')
    else:
        print('Solving for Cellular Activities...')    
    
    print('MSE Activity Reconsruction:')
    mseImgs, mseActs = reconsImgSet(imgSet,imgData, simParams, psychParams, "mse", electrode)
    print('wMSE Activity Reconstruction')
    wmsImgs, wmsActs = reconsImgSet(imgSet,imgData, simParams, psychParams, "wms", electrode)
    print('SSIM Activity Reconstruction')
    ssmImgs, ssmActs = reconsImgSet(imgSet,imgData, simParams, psychParams, "ssm", electrode)
    
    print('Activities Solved. Rebuilding Images ...')
    pixelDims = simParams["pixelDims"]
    
    mseRecons = rebuildImg(img,mseImgs,xs,ys,pixelDims,psychParams)
    wmsRecons = rebuildImg(img,wmsImgs,xs,ys,pixelDims,psychParams)
    ssmRecons = rebuildImg(img,ssmImgs,xs,ys,pixelDims,psychParams)

    print('Images rebuilt.')
    print('Simulation Complete')
    
    return (
            mseImgs, wmsImgs, ssmImgs,
            mseActs, wmsActs, ssmActs,
            mseRecons, wmsRecons, ssmRecons
           )

def reconsImgSet(imgSet, imgData, simParams, psychParams, metric, electrode):
    # Given a set of images (imgSet) as a 2d Matrix, and a metric, reconstruct
    # the image set according to the given image in parallel according to the available cpu cores    
    if electrode:
        activityLength = simParams["P"].shape[1]
    else:
        activityLength = simParams["A"].shape[1]
    
    numPixels = imgSet.shape[0]
    numImgs   = imgSet.shape[1]
    
    # convert imgSet to list for parallelization
    imgList = []
    for i in np.arange(numImgs):
        imgList.append(imgSet[:,i])
     
    num_cores = mp.cpu_count()
    
    # run reconstructions in parallel
    results = np.asarray(Parallel(n_jobs=num_cores)(delayed(actSolver)(i,imgData,simParams,psychParams,metric,electrode) for i in tqdm(imgList)))

    #convert results back to 2 variables separating activity and the reconstructed image
    imgs = np.zeros((numPixels,numImgs))
    acts = np.zeros((activityLength,numImgs))
    for i in np.arange(numImgs):
        imgs[:,i] = results[i,0]
        acts[:,i] = results[i,1]
    return imgs, acts   

def loadRawImg(fName):
    # given a filename for an rgb image, load and preprocess the image by doing the following:
    # convert to grayscale --> zero-mean --> normalize to +/-5 .5 intensity
    img = plt.imread(fName)
    img = np.sum(img,2)/3

    # Normalize to +/- 1 intensity range and zero mean
    img -= np.mean(img)
    img = img / np.max((np.abs(img))) 

    return img
    
def dct2(a):
    # 2D Discrete Cosine Transform and Its Inverse
    lDim = a.shape[0]
    rDim = a.shape[1]
    # build the matrix
    n, k = np.ogrid[1:2*lDim+1:2, :lDim]
    m, l = np.ogrid[1:2*rDim+1:2, :rDim]
    Dl = 2 * np.cos(np.pi/(2*lDim) * n * k)
    Dr = 2 * np.cos(np.pi/(2*rDim) * m * l)
    return (Dl.T @ a @ Dr)

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def genStixel( height, width, s ):
# % genStiheightelImg: Generate a zero-mean white-noise stixelated image of specified
# % dimension.
# %   This function generates an image of size specified bwidth (height,
# %   width), and divides the image into s height s squares
# %   each stiheightel having the same Gaussian Generated white noise value. 
# %   The Gaussian values range from [-0.5, 0.5]. 


    heightStixel = np.floor(height/s).astype(int)  #% full number of stixels
    widthStixel = np.floor(width/s).astype(int)
    remWidth = width - s*widthStixel #% remainder that specifies padding
    remHeight = height - s*heightStixel

    #% Depending whether there is remainder after full stixels, determine
    #% if we need to pad. Otherwise, set pad variables to 0
    if ( remWidth != 0): 
        wpad = 1
    else: 
        wpad = 0

    if (remHeight != 0):
        hpad = 1
    else: 
        hpad = 0


    # pad the image to fit to remainder size
    img = np.zeros((height+remHeight,width+remWidth)) # %initialize image

    #% Fill in the full stixel 
    for i in np.arange(heightStixel+hpad+1):   # For each stixel block
        for j in np.arange(widthStixel+wpad+1):
            #% Generate a Gaussian White Noise value between [-0.5,0.5]
            val = np.random.normal(0,1)
            # Assign Block the Gaussian Value
            img[(i-1)*s:i*s,(j-1)*s:j*s] = val


    # clip image to original dimensions
    img = img[0:height,0:width]
    #normalize img to lie on interval [-0.5,0.5]
    if (np.max(img)) != 0:
        img = img / (2*np.max(img))
        img[img > 0] = .5
        img[img <= 0 ] = -.5
    return img

def flatDCT(pixelDims):
    # build and return a flattened dct matrix specifically for (80,40) images flattened with fortran ordering
    # Build 80 x 40 2D DCT-II Matrix
    numPixels = pixelDims[0]*pixelDims[1]
    D1 = np.zeros((numPixels,numPixels))
    D2 = np.zeros((numPixels,numPixels))
    # build a flattened form of a  1d DCT matrix 
    lDim = pixelDims[0]
    rDim = pixelDims[1]
    n, k = np.ogrid[1:2*lDim+1:2, :lDim]
    m, l = np.ogrid[1:2*rDim+1:2, :rDim]
    Dl = 2 * np.cos(np.pi/(2*lDim) * n * k)
    Dr = 2 * np.cos(np.pi/(2*rDim) * m * l)

#     imRows = 80
#     imCols = 40
    # build D1
    for i in np.arange(lDim):
        for j in np.arange(rDim):
            D1[j*lDim + i,j*lDim:(j+1)*lDim] = Dl.T[i,:]


    # build D2
    for i in np.arange(rDim):
        for k in np.arange(lDim):
            for j in np.arange(rDim):
                D2[k+j*pixelDims[0],i*pixelDims[0]+k] = Dr[i,j]
    D = D2@D1
    return D

def flatW(psychParams,pixelDims,imgData): 
    # build and return a flattned W matrix for images (img) flattned with fortran ordering
    # (1/2) * N / D where D is horizontal degrees, N is number of blocks 
    XO = psychParams["XO"]
    N  =  int(imgData.origImg.shape[0]/imgData.sDims[0]) # number of selection blocks (number of samples of DC terms of each subImage)
    offset = (1/2) * (N / XO)
    Wp = csf(psychParams,pixelDims,offset=offset) #offset frequency b
    flatW = np.reshape(Wp,(pixelDims[0]*pixelDims[1],),order='F')
    W = np.diag(flatW)
    return W

def csf(psychParams,pixelDims,offset=0):
    # given a peak sensitivity frequency pf, and a psychophysically determined pixels-per-degree of viusal field ppd,
    # and and image, return a mask that has the same shape as the image and applies a weighting to each pixel in the image
    # according to the contrast sensitivity function 
    def getNg(psychParams):
        e = psychParams["e"]
        Ng0 = psychParams["Ng0"]
        eg = psychParams["eg"]
        term1 = .85 / (1 + (e/.45)**2)
        term2 = .15 / (1 + (3/eg)**2)
        return Ng0*term1*term2
    
    def Mopt(f,psychParams):
        #given a spatial frequency f and psychophysical parameters,
        # return the frequnecy filetered by the optical transfer function
        # of the retina
        sigma00 = .30           # Non-retinal optical linespread constant (arcmin)
        sigmaRet = 1 / np.sqrt(7.2*np.sqrt(3)*getNg(psychParams))
        sigma_0 = np.sqrt(sigma00**2 + sigmaRet**2) # (arcmin) std deviation of linespread (function of eccentricity)
        Cab = .08    # (arcmin / mm ) dimensionality constant
        d = psychParams["d"] # pupil size in mm
        sigma = np.sqrt(sigma_0**2 + (Cab*d)**2)
        return np.exp(-2*(np.pi**2)*((sigma/60)**2)*(f**2))
        
    def intTerm(f,psychParams):
        # given spatial frequency f and psychophysical paratmeters,
        # calculate the visual-angle integration term of the CSF
        e = psychParams["e"]
        Xmax = 12   # (degrees) maximum visual integration area  
        term1 = .85 / (1 + (e/4)**2)
        term2 = .15 / (1 + (e/12)**2)
        Xmax=Xmax*(term1+term2)**-.5
        Ymax = Xmax
        Nmax = 15  # (cycles) maximum number of cycles function of eccentriicty
        XO = psychParams["XO"]
        YO = psychParams["YO"]
        
        term1 = (.5*XO)**2 + 4*e**2
        term2 = (.5*XO)**2 + e**2
        NmaxFac = term1/term2
        
        return 1/(XO*YO) + 1/(Xmax*Ymax) + NmaxFac*(f/Nmax)**2
    
    def illumTerm(psychParams):
        #given spatial frequency f and psychophysical parameters,
        # calculate the  illumance term of the CSF
        n = .03  #quantum efficiency term (function of eccentricity)
        e = psychParams["e"]
        term1 = .4 / (1 + (e/7)**2)
        term2 = .48 / (1 + (e/20)**2) 
        n = n*(term1 + term2 +.12)
        p = 1.24 # photon conversion factor (function of incident light)
        d = psychParams["d"]
        L = psychParams["L"]
        E = np.pi/4 * d**2 * L * (1 - (d/9.7)**2 + (d/12.4)**4)
        return 1/(n*p*E)
        
    def inhibTerm(f,psychParams):
        # given spatial frequency f and psychophysical parameters,
        # calculate the lateral inhibition term of the CSF
        Ng0 = psychParams["Ng0"]
        e = psychParams["e"]
        u0 = 7  #(cycles/deg) stop frequency of lateral inhibition
        term1 = .85 / (1 + (e/4)**2)
        term2 = .13 / (1 + (e/20)**2)
        u0 = u0 * (getNg(psychParams)/Ng0)**.5 * (term1 + term2 + .02)**-.5
        return 1 - np.exp(-(f/u0)**2)
    
    k  = psychParams["k"]
    X0 = psychParams["elecXO"]
    Y0 = psychParams["elecYO"]
    T  = psychParams["T"]
    sfRes = 1/pixelDims[0]
    Ng = getNg(psychParams)
    Ng0 = psychParams["Ng0"]
    ph0= 3*10**-8*Ng0/Ng  # neural noise term (sec / deg^2)
    fxx,fyy = np.meshgrid(np.arange(pixelDims[1]),np.arange(pixelDims[0]))
    ppd = pixelDims[0]/X0
    fs = (sfRes * ppd *((fxx)**2+(fyy)**2)**.5  ) + offset
    
    num   = Mopt(fs,psychParams) / k
    
    if not psychParams["binocular"]:
        num = num /  np.sqrt(2)
    
    denom = np.sqrt( 
        (2/T)
        *intTerm(fs,psychParams)
        *(illumTerm(psychParams) + ph0 / inhibTerm(fs,psychParams)) 
    )
    W = np.divide(num,denom)
    return W

def pruneDecoder(A,P):
    # remove the columns of A corresponding to the cells which don't change the image
    # reconstruction
    # also delete the corresponding rows of P
    # if a column of A has a norm of 0 it must be all 0, so delete the column. 
    delList = []
    for i in np.arange(A.shape[1]):
        if (np.linalg.norm(A[:,i])) <= 10**-6:
            delList.append(i)
            
    return np.delete(A,delList,axis=1),np.delete(P,delList,axis=0)

def pruneDict(P,eActs,threshold=.05):
    # Given a dictionary and a threshol value, remove any dictionary elements whose maximum value is 
    # below the threshold.  Append an element of zeros to the pruned dictionary. 
    pp = P.copy()
    pp[pp <= threshold] = 0
    
    dictLength = pp.shape[1]
    toDel = []
    for i in  np.arange(dictLength):
        if ~np.any(pp[:,i]):
            toDel.append(i)
    if ~np.any(pp[:,dictLength-1]):
            toDel.append(dictLength-1)
    
    
    pp = np.delete(pp,toDel,axis=1)
    eActs = np.delete(eActs,toDel,axis=0)
    
    return np.hstack((pp,np.zeros((pp.shape[0],1)))),  np.vstack((eActs,np.asarray(np.zeros((1,eActs.shape[1])))))
    
def mse(A,B):
    #flatten if not flat
    if A.ndim > 1:
        flatA = A.flatten()
        flatB = B.flatten()
        
        return (flatA-flatB).T@(flatA-flatB)/flatA.size
    else:
        return (A-B).T@(A-B)/A.size

def jpge(A,B,psychParams,pixelDims, imgData):
    jpge.D = flatDCT(pixelDims)
    diffImg = A - B
    if diffImg.ndim is not 1: #flatten image if not already flattened
        diffImg = diffImg.flatten
    W = flatW(psychParams, pixelDims, imgData)
    W = W/np.max(W)

    return np.linalg.norm(W@jpge.D@diffImg)**2 / A.size

def SSIM(X, Y, K1=.01, K2=.03, alpha=1, beta=1, gamma=1, L=1 ):
    # Given two images A & B of the same size, calculate & Return Their Structural Similarity Index
    # Parameters: A,B: two MN x 1 flattened images
    #             K1,K2: Stability Constants (retried from Wang SSIM Paper)
    #             alpha, beta, gamma: relative powers of luminance, contrast, and structural functions respectivtely
    #             L: dynamic range of pixel intensities
    
    if X.ndim is not 1:
        X = X.flatten
        Y = Y.flatten
    
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    C3 = C2/2 #by default from Wang Paper
    numPixels = X.shape[0]
    meanX = np.mean(X)
    meanY = np.mean(Y)
    lum = (2*meanX*meanY + C1) / (meanX**2 + meanY**2 + C1)

    stdX = np.std(X)
    stdY = np.std(Y)

    con = ( 2*stdX*stdY + C2) / (stdX**2 + stdY**2 + C2)
    stdXY = (1 / (numPixels-1)) * np.sum( np.multiply((X-meanX),(Y - meanY)) ) 

    srt   = (stdXY + C3) / (stdX*stdY + C3)
    ssim = lum**alpha * con**beta * srt**gamma
    return ssim

def getElecAngs(smps,stixelSize, eyeDiam, pixelDims):
    # Given a set of psychophysical parameters,the reconstructing electrode array
    # smps: stimulus monitor pixel size: the size of a single monitor pixel in lab setup on the retina (microns)
    # stixelSize:  the stixel size,which is the square root of the number of monitor pixels grouped together 
    #      to form a single STA pixel (one STA pixel is stixelSize x stixelSize monitor pixels)
    # eyeDiam: the Emmetropia diameter of the eye in milimeters
        
    
    retArea  = ( # Retinal area in milimeters
        pixelDims[0]*smps*stixelSize/1000,
        pixelDims[1]*smps*stixelSize/1000
    )
    
    elecVisAng = ( # Visual Angle Spanned by the Electrode Reconstruction 
        np.rad2deg(np.arctan(retArea[0]/eyeDiam)),
        np.rad2deg(np.arctan(retArea[1]/eyeDiam))
    )
    
    return elecVisAng

def preProcessImage(img,psychParams,simParams):
    # Given psychophysically determined viewing angles for the visual
    # scene, the image, and the dimensions of the stimulus reconstruction in 
    # pixels, tile the image into a set of subimages, where each subimage
    # covers precisely elecVisAng[0] x elecVisAng[1] degrees of the visual
    # scene. Resample these tiled images to have the same dimensions as the 
    # stimulus pixel (pixelDims) for reconstruction.
    # elecVisAng[0]/objVisAngle[0] = selection/  img.shape[0]
    
    def tileImage(img,pixelDims):
        # Given an mxn image and pixelDims, tile the image by splitting it into 
        # numImgs subimages obtained by taking pieces of size pixelDims from the original image, stacking,
        # and then returning the images, as well as the x & y locations of the top left corner of each image
        
        def fitToDims(img,pixelDims):
            # Given an mxn image, fit the image to the given dimension by padding it with zeros. 
            # This imamge assumes m<= pixelDIms[0] and/or n <= pxielDims[1]
            fitImg = np.zeros(pixelDims)
            fitImg[0:img.shape[0],0:img.shape[1]] = img
            return fitImg       
        
        print('Tiling Image ...')
        x = 0
        y = 0 # initial location is top left of image
        subImgs = np.zeros((pixelDims[0]*pixelDims[1],0))
        xs = np.asarray([])
        ys = np.asarray([])

        while y <= img.shape[1]-pixelDims[1]:
            # sweep horizontally. if x >= img.shape set x to 0 and update y
            if x >= img.shape[0]-pixelDims[0]: 
                x = 0
                y += int(pixelDims[0])

            selection = fitToDims(img[x:x+pixelDims[0],y:y+pixelDims[1]],pixelDims)
            selection = np.reshape(selection,(pixelDims[0]*pixelDims[1],1),order='F')
            if not np.all(selection==0):
                subImgs = np.concatenate((subImgs,selection),1)
                xs = np.append(xs,[x])
                ys = np.append(ys,[y])
                x += int(pixelDims[0])

        print('Tiled Image')        
        return subImgs, xs, ys

    def csfFilterImg(img, psychParams):
        # given an image and psychophysical parameters object,
        # filter the image according to the contrast sensitivity function#
        # and return the filtered image.
        
        W = csf(psychParams,img.shape)
        filtImg = np.multiply(W,dct2(img))
        filtImg = idct2(filtImg)
        return filtImg/np.max(np.abs(filtImg))  - .5 # renormalize after filter
        

    filtImg   = csfFilterImg(img, psychParams)
    pixelDims = simParams["pixelDims"]
    
    selecDims = getSelectionDims(psychParams,img)

    imgSet, xs, ys        = tileImage(img,selecDims)
    filtImgSet,xs, ys      = tileImage(filtImg, selecDims)

    numImgs = filtImgSet.shape[1]
    resImgSet = np.zeros((pixelDims[0]*pixelDims[1],numImgs))
    resFltSet = np.zeros((pixelDims[0]*pixelDims[1],numImgs))

    # go through each image, resample it and store it in resImgSet
    for i in np.arange(numImgs):
        resImgSet[:,i],zoomF = resample(imgSet[:,i],selecDims,pixelDims)
        resFltSet[:,i],zoomF = resample(filtImgSet[:,i],selecDims,pixelDims)
    imgData = ImageData()
    imgData.numImgs = numImgs
    imgData.imgSet = resImgSet
    imgData.filtImgSet = resFltSet
    imgData.xs = xs
    imgData.ys = ys
    imgData.zoomFac = zoomF
    imgData.origImg    = img
    imgData.filtImg    = filtImg
    imgData.sDims      = selecDims
    
    return imgData

def getSelectionDims(psychParams,img):
    XO = psychParams['XO']
    elecXO = psychParams['elecXO']
    elecYO = psychParams['elecYO']
    selectionSize = int(np.ceil(elecXO/XO * img.shape[1]))

    # select the equivalent of elecVisangx elecVisAng pixels from the image
    selecDims = (selectionSize,selectionSize)
    return selecDims

def actSolver(img,imgData,simParams,psychParams,mode,electrode):
    # Reconstruct an image according to the error metric specified by "mode"
    # Input: img : the image to be reconstructed, dims = psychParams["pixelDims"]
    #        simParams : a simulation parameters dictionary 
    #        psychParams: a psychophysical parameters dictionary
    #        mode : a string specifying the particular error metric being used
    #        electrode : a boolean specifying whether to reconstruct according ot optimal cell 
                # activities or using th electrode stimulation dictionary 
    #Subfunctions:
    def varTerm(simParams,Phi, x):
    # Return the cost function associate with the variance component of the reconstruction
    # error. Only used in the case that electrode is true
    # Inputs: 
    #     simParams: the simulatin parameters dictionary object
    #     electrode: boolean indicating whether performing optimal cellular or electrode dictionary recons
    #     x : the cvx variable representing the activity vector object that is being solved for

        P = simParams["P"]
        A = simParams["A"]
        V = np.zeros(P.shape)
        for j in np.arange(P.shape[1]):
            V[:,j] = np.multiply(P[:,j],(1-P[:,j]))
        varMtx = np.multiply(Phi,Phi)@V
        return  cp.sum(varMtx@x)
    
    def reconsSSM(img, simParams, electrode, epsilon = 10**-2):
        # use bisection search to solve for an optimal-SSIM reconstruction

        def findFeasible(y,alpha,simParams, electrode ):
            # Return a feasible solution to the SSIM optimization problem
            # Using cvxpy solves the constrained feasability problem that is a transformation of the SSIM
            # optimization problem.

            def cvxineq(a,y,x,Phi):
                # a convex inequality to evaluate feasability
                return (1-a)*cp.sum_squares(y-Phi@x)-2*a*(Phi@x).T@y

            A = simParams["A"]
            P = simParams["P"]

            if electrode:
                x = cp.Variable(P.shape[1])
                cost = varTerm(simParams, A , x)
                Phi = A@P
            else:
                x = cp.Variable(A.shape[1])
                cost = 1
                Phi = A

            T = simParams["numStims"]
            N = simParams["maxAct"]
            if T == -1:
                constraints = [x <= N, x >= 0, cvxineq(alpha,y,x,Phi) <= 0]
            else:
                constraints = [x <= N, x >= 0, cvxineq(alpha,y,x,Phi) <= 0, cp.sum(x) <= T]

            prob= cp.Problem(cp.Minimize(cost),constraints)
            try:
                prob.solve(solver=cp.GUROBI)
            except: 
                prob.solve(solver=cp.SCS)

            if x.value is not None:
                return True, x.value
            else:
                return False, x.value

        A = simParams["A"]
        P = simParams["P"]
        if electrode:
            actLength = P.shape[1]
        else:
            actLength = A.shape[1]


        # image preprocessing 
        y = img 


        # bisection initialization
        l = 0 # lower bound
        u = 2 # upper bound
        e = epsilon  # accuracy
        x = np.zeros(actLength) # solution
        xCurr = np.zeros(actLength) # temporary solution

        # bisection search
        while u - l >= e:
            alpha = (l+u)/2
            # find feasible x   let u = alpha
            isFeasible, xCurr = findFeasible(y, alpha, simParams, electrode)

            if isFeasible:
                u = alpha
            elif alpha == 1:
                print('SSIM reconstruction cannot be solved.')
                if electrode:
                    return 0*A@P@x, 0*x
                else:
                    return 0*A@x, 0*x
            else:
                l = alpha

            if xCurr is not None: # only overwrite x is new value is generated
                x = copy.deepcopy(xCurr)            
        x = np.rint(x)
        if electrode:
            return A@P@x, x
        else:
            return A@x, x
    
    
    A = simParams["A"]
    P = simParams["P"]
    T = simParams["numStims"]
    N = simParams["maxAct"]
    pixelDims = simParams["pixelDims"]

    y = img

    if electrode:
        x = cp.Variable(P.shape[1])
    else:
        x = cp.Variable(A.shape[1])

    if mode == "mse": 
        if electrode:
            cost = cp.sum_squares(y-A@P@x) + varTerm(simParams,A,x)
        else:
            cost = cp.sum_squares(y-A@x)
    
    elif mode == "wms":
        W = flatW(psychParams,simParams["pixelDims"],imgData)
        D = flatDCT(pixelDims)
        if electrode:
            cost = cp.sum_squares(W@D@(y-A@P@x)) + varTerm(simParams, W@D@A, x)
        else:
            try:
                cost = cp.sum_squares(W@D@(y-A@x))
            except:
                print(W.shape)
                print(D.shape)
                print(y.shape)
                print(x.shape)
                print(A.shape)
            
    elif mode == "ssm": 
        # custom SSIM bisection search solver
        return reconsSSM(img, simParams, electrode)
        
    # Solve cost function and return x's value and the reconstructed image
    if T == -1:
        prob= cp.Problem(cp.Minimize(cost),[x<=N,x >= 0])
    else:
        prob = cp.Problem(cp.Minimize(cost),[x<=N, x >= 0, cp.sum(x) <= T])
        
    try:
        prob.solve(solver=cp.GUROBI)
    except:
        prob.solve(solver=cp.SCS)
    
    if electrode:
        return A@P@x.value, x.value
    else:
        try:
            return A@x.value, x.value
        except:
                print(W.shape)
                print(D.shape)
                print(y.shape)
                print(x.shape)
                print(A.shape)
                return
    
def numStimSweep(imgData,simParams,psychParams,electrode):
    # Given a set of images, reconstruct each image using all metric and sweep over the number of allowable stimulations.
    # run a metric comparison simulation over a specified number of stimulation times
    
    Tres = 16
    Ts   = np.logspace(0,5,Tres)
    
    mseImgSets = []
    wmsImgSets = []
    ssmImgSets = []
    
    mseActSets = []
    wmsActSets = []
    ssmActSets = []
    
    mseRecSets = []
    wmsRecSets = []
    ssmRecSets = []
    
    for Tidx, T in enumerate(Ts):
        print("T: %i;  %i/%i"%(T, Tidx+1, Ts.size))
        simParams["numStims"] = T
        (
      mseImgs, wmsImgs, ssmImgs,
      mseActs, wmsActs, ssmActs,
      mseRecons, wmsRecons, ssmRecons
    )  =  metricCompar(imgData,simParams,psychParams, electrode)
        
        mseImgSets.append(mseImgs)
        wmsImgSets.append(wmsImgs)
        ssmImgSets.append(ssmImgs)

        mseActSets.append(mseActs)
        wmsActSets.append(wmsActs)
        ssmActSets.append(ssmActs)

        mseRecSets.append(mseRecons)
        wmsRecSets.append(wmsRecons)
        ssmRecSets.append(ssmRecons)
        
        ssData = StimSweepData()
        ssData.Ts        = Ts
        ssData.mseImgSet = np.asarray(mseImgSets)
        ssData.wmsImgSet = np.asarray(wmsImgSets)
        ssData.ssmImgSet = np.asarray(ssmImgSets)
        
        ssData.mseActSet = np.asarray(mseActSets)
        ssData.wmsActSet = np.asarray(wmsActSets)
        ssData.ssmActSet = np.asarray(ssmActSets)
        
        ssData.mseRecSet = np.asarray(mseRecSets)
        ssData.wmsRecSet = np.asarray(wmsRecSets)
        ssData.ssmRecSet = np.asarray(ssmRecSets)
        
    return ssData

def resample(img,currDims,desiredDims):
    # given a (currDims[0]*currDims[1] x 1 ) image vector, resample the image
    # to fit to desired dims and return this image flatted into a 
    #(desiredDims[0],desiredDims[1] x 1) image vector
    currImg = np.reshape(img,currDims,order='F')

    # desiredDims[0] = zoomFac * currDims[0]
    zoomFac =  desiredDims[0]/currDims[0]
    zImg = ndimage.zoom(currImg,zoomFac)
    return np.reshape(zImg,(desiredDims[0]*desiredDims[1],),order='F'),zoomFac

## Visulization Functions

def dataPCAAnalysis(eLocs,eMap,imgs,mseActs,wmsActs, ssmActs):
    # given a set of images, electrode locations, and their dictionary reconstructions,
    # calculate correlations (if any) of electrode activity across the set of images 
    numImages = imgs.shape[1]

#     mseCurr = np.zeros((numImages,))
#     jpgCurr = np.zeros((numImages,))
#     ssmCurr = np.zeros((numImages,))
#     for i in np.arange(numImages):
#         mseCurr[i] = np.dot(mseActs[:,imgNum],eMap[:,1])
#         jpgCurr[i] = np.dot(jpgActs[:,imgNum],eMap[:,1])
#         ssmCurr[i] = np.dot(ssmActs[:,imgNum],eMap[:,1])
    
#     mseActs = np.vstack((mseActs,mseCurr))
#     jpgActs = np.vstack((jpgActs,jpgCurr))
#     ssmActs = np.vstack((ssmActs,ssmCurr))

#     print('Average Current for MSE Images: %i nC' % np.mean(mseCurr))
#     print('Average Current for CSF Images: %i nC' % np.mean(jpgCurr))
#     print('Average Current for SSIM Images: %i nC' % np.mean(ssmCurr))

    data = np.hstack((mseActs,wmsActs,ssmActs))
    covdata = np.cov(data) # covariance matrix
    wdata,vdata = np.linalg.eig(covdata) # eigen decomposition of covariance matrix

    # project each activity vector onto the 3 respective components
    (data1,data2,data3) = projectPC(data,vdata[:,0],vdata[:,1],vdata[:,2])

    # generate set of random data restricted to be positive within the range o
    dataMax = np.max(data)
    randData = np.random.randint(0,dataMax,size=data.shape)
    (rand1,rand2,rand3) = projectPC(randData,vdata[:,0],vdata[:,1],vdata[:,2])

    markerSize = 1

    # 3D Scatter Plot of Image Data Projected onto principal Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[0:numImages-1],data2[0:numImages-1],data3[0:numImages-1],label='MSE',s=markerSize)
    ax.scatter(data1[numImages:2*numImages-1],data2[numImages:2*numImages-1],data3[numImages:2*numImages-1],label='wMSE',s=markerSize)
    ax.scatter(data1[2*numImages:],data2[2*numImages:],data3[2*numImages:],c='red',label='SSIM',s=markerSize)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.legend(loc='upper right')
    plt.show()
    
    xLims = 2
    yLims = 1

    # 2D Plot Projected Onto Principal Axes
    plt.figure()
    plt.scatter(data1[0:numImages-1],data2[0:numImages-1],s=markerSize,label='MSE')
    plt.scatter(data1[numImages:2*numImages-1],data2[numImages:2*numImages-1],s=markerSize,label='JPG')
    plt.scatter(data1[2*numImages:],data2[2*numImages:],c='red',label='SSIM',s=markerSize)
    plt.title('PC1 & PC2')
    plt.legend()
    plt.xlim([-xLims,xLims])
    plt.ylim([-yLims,yLims])
    plt.savefig('PC1PC2Electrode.jpg',bbox_inches='tight')
    plt.show()


    plt.figure()
    plt.scatter(data1[0:numImages-1],data3[0:numImages-1],s=markerSize,label='MSE')
    plt.scatter(data1[numImages:2*numImages-1],data3[numImages:2*numImages-1],s=markerSize,label='JPG')
    plt.scatter(data1[2*numImages:],data3[2*numImages:],c='red',label='SSIM',s=markerSize)
    plt.title('PC1 & PC3')
    plt.legend()
    plt.xlim([-xLims,xLims])
    plt.ylim([-yLims,yLims])
    plt.savefig('PC1PC3Electrode.jpg',bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.scatter(data2[0:numImages-1],data3[0:numImages-1],s=markerSize,label='MSE')
    plt.scatter(data2[numImages:2*numImages-1],data3[numImages:2*numImages-1],s=markerSize,label='JPG')
    plt.scatter(data2[2*numImages:],data3[2*numImages:],c='red',label='SSIM',s=markerSize)
    plt.xlim([-xLims,xLims])
    plt.ylim([-yLims,yLims])
    plt.title('PC2 & PC3')
    plt.legend()
    plt.savefig('PC2PC3Electrode.jpg',bbox_inches='tight')
    plt.show()


## also plot centroids in pca space
#     dataVecs = np.vstack((data1,data2,data3))
#     mseCentroid = np.sum(dataVecs[:,0:numImages-1],1)/numImages
#     sfeCentroid = np.sum(dataVecs[:,numImages:2*numImages-1],1)/numImages
#     jpgCentroid = np.sum(dataVecs[:,2*numImages:],1)/numImages

#     mseCentroidAct = np.real(mseCentroid[0]*vdata[:,0] + mseCentroid[1]*vdata[:,1] + mseCentroid[2]*vdata[:,2])
#     sfeCentroidAct = np.real(sfeCentroid[0]*vdata[:,0] + sfeCentroid[1]*vdata[:,1] + sfeCentroid[2]*vdata[:,2])
#     jpgCentroidAct = np.real(jpgCentroid[0]*vdata[:,0] + jpgCentroid[1]*vdata[:,1] + jpgCentroid[2]*vdata[:,2])


#     print(vdata.shape)
#     mseCentImg = np.reshape(np.expand_dims(A@P@mseCentroidAct,axis=1),(80,40),order='F')
#     sfeCentImg = np.reshape(np.expand_dims(A@P@sfeCentroidAct,axis=1),(80,40),order='F')
#     jpgCentImg = np.reshape(np.expand_dims(A@P@jpgCentroidAct,axis=1),(80,40),order='F')

#     print(mseCentImg)
#     maxval = .0001
#     minval = -maxval

#     plt.figure(figsize=(10,10))
#     plt.subplot(131)
#     plt.title('MSE PC Centroid')
#     plt.imshow(mseCentImg,cmap='bone',vmin=minval,vmax=maxval)
#     plt.axis('off')
#     plt.subplot(132)
#     plt.title('SFE PC Centroid')
#     plt.imshow(sfeCentImg,cmap='bone',vmin=minval,vmax=maxval)
#     plt.axis('off')
#     plt.subplot(133)
#     plt.title('JPG PC Centroid')
#     plt.imshow(jpgCentImg,cmap='bone',vmin=minval,vmax=maxval)
#     plt.axis('off')
#     plt.savefig('centComparCell.jpg',bbox_inches='tight')
#     plt.show()


#     plt.figure(figsize=(10,10))
#     plt.imshow(np.abs(mseCentImg-jpgCentImg)/maxval,cmap='bone',vmin=0,vmax=1)
#     plt.axis('off')
#     plt.title('|MSE - JPG|/max(MSE) PC Centroid')
#     plt.colorbar()
#     plt.savefig('mseJpgCentComparCell.jpg',bbox_inches='tight')
#     plt.show()


    return wdata,vdata

def projectPCA(data,pc1,pc2,pc3):
    #given a dataDim x numPts matrix of data, and 3 dataDim principal component vectors,
    #return a numPts vector containing the scalar projection of the data onto the vector at each numpt

    dataDim = data.shape[0]
    numPts = data.shape[1]
    proj1 = np.zeros((numPts,))
    proj2 = np.zeros((numPts,))
    proj3 = np.zeros((numPts,))

    for i in np.arange(numPts):
        dataNorm = np.sum(np.multiply(data[:,i],data[:,i]))
        proj1[i] = np.dot(data[:,i],pc1)/(np.linalg.norm(pc1)*np.linalg.norm(data[:,i]))
        proj2[i] = np.dot(data[:,i],pc2)/(np.linalg.norm(pc2)*np.linalg.norm(data[:,i]))
        proj3[i] = np.dot(data[:,i],pc3)/(np.linalg.norm(pc3)*np.linalg.norm(data[:,i]))
    return (proj1,proj2,proj3)

def dispElecAct(elecActs,simParams,color='blue'):
    # Given a vector of electrode activities, a vector of (x,y) electrode locations, and a 2xnumElectrode (contained in simparams)
    # matrix of electrode numbers for each element, sum the total current passing through each electrode, 
    # and display it in a scatter plot
    eLocs = simParams["eLocs"]
    eMap  = simParams["eMap"]
    
    totalCurr = getTotalCurr(elecActs,simParams)
    
    plt.scatter(eLocs[:,0],eLocs[:,1],alpha=.5,s=totalCurr,c=color)
    plt.scatter(eLocs[:,0],eLocs[:,1],alpha=.5,s=1,c='black')
    plt.title('Total Electrode Current: %i nC' %np.sum(totalCurr))
    plt.axis('equal')
    plt.xlabel('Horizontal Position (um)')
    plt.ylabel('Vertical Position (um)')
    


def angBT(vecA,vecB):
    # return the cosine of the angle between to vectors:
    ang = np.arccos(np.dot(vecA,vecB)/(np.linalg.norm(vecA)*np.linalg.norm(vecB)))
    return np.rad2deg(ang)

def rebuildImg(img,imgSet,xs,ys,pixelDims,psychParams,zeroMean=False): 
    # input params:
    # img: an mxn original image matching the desired image dimensions
    # imgSet a (numPixels x numImgs) matrix of flattened subimages
    # xs a numImgs vector of x positions for the upper left loc of each subimage
    # ys a numImgs vector of y positions for the upper left loc of each subimage
    # returns: a reconstructed image having the same dimensions of the original image (img),
    #      built from the set of subimages
    
    # initialize image
    recons    = np.zeros(img.shape)
    xs = xs.astype(int)
    ys = ys.astype(int)
    
    #calc selection dims
    selecDims = getSelectionDims(psychParams,img)
    # iterate through each (x,y) coordinate a
    for i in np.arange(xs.shape[0]): 
        # if dims not correct, resample to selectiondims
        if (pixelDims[0] != selecDims[0] or pixelDims[1] != selecDims[1]):
            # resample image
            resampledImg = resample(imgSet[:,i],pixelDims,selecDims)[0]
            
        reconsImg = np.reshape(resampledImg,selecDims,order='F')
        

        # only add to image if exactly zero pixel
        x = xs[i]
        y = ys[i]
        
        
        selection = recons[x:x+selecDims[0],y:y+selecDims[1]]
        reconsSel = reconsImg[0:selection.shape[0],0:selection.shape[1]]
        
        # If zeromean, make the subimage zero mean, then add the average intensity for the whole image
        if zeroMean:
            reconsSel -= np.mean(reconsSel) 
        recons[x:x+selection.shape[0],y:y+selection.shape[1]] += reconsSel
    
    return recons  

def actAnglePlot(mseActs,otherActs,otherMetricLabel):
    # given two actLength x numImgs matrices of activity over a numImgs set of images,
    # calculate the angle between each pair of activities over all image and plot this
    # angle on a polar graph, where the radius of each data point is the sum of the 
    # activity of the other activity minus the sum of the MSE activity.
    
    numImgs =  mseActs.shape[1]
    angles = np.zeros((numImgs,))
    radii  = np.zeros((numImgs,))
    
    for imgNum in np.arange(numImgs):
        angles[imgNum] = angBT(mseActs[:,imgNum],otherActs[:,imgNum])
        radii[imgNum]  =(np.sum(otherActs[:,imgNum])-np.sum(mseActs[:,imgNum])) 
    
    pos = radii >= 0
    neg = radii <  0 
    
    plt.polar(angles[pos],radii[pos],'ro',c='red')
    plt.polar(angles[neg],np.abs(radii[neg]),'ro',c='blue')
    plt.title('Angle Between MSE & '+otherMetricLabel+' Activity')
    plt.show()
    return

def plotStimCompar(imgData, ssData, metric, psychParams, pixelDims):
    # plot the error of given image sets versus number of stimulations according to  a specified metric, average over
    #  all images for each number of stimulations data point
    # Input: imgData: imgData object created by preProcessImage function
    #        metric: metric according to which activities 1 and 2 will be plotted/calculated: "mse", "wms", or "ssm"
    #        psychParams: a psychophysical parameters object 
    #        ssData: Stimulation Sweep Data output from numStimSweep Function
    #        pixelDims:  the dimensions of the unflattened image (e.g. (20, 20) )
    
    def getErr(refImg, img, metric,psychParams,pixelDims, imgData):
        # given two images, calculate the error according to the given metric
        # metric = "mse", "jpg", or "ssm"
        if metric.upper() == "MSE":
            return mse(refImg, img)/mse(refImg, np.zeros(refImg.shape))
        elif metric.upper() == "WMS" :
            return jpge(refImg, img, psychParams, pixelDims, imgData) / jpge(refImg, np.zeros(refImg.shape), psychParams, pixelDims, imgData)
        elif metric.upper() == "SSIM":
            return ((1-SSIM(refImg, img))/2) / ((1-SSIM(refImg,np.zeros(refImg.shape)/2)))
        else: 
            raise Exception("Bad Metric Passed to getErr")
    
    def getErrVecs(refImg, Ts, img1, img2, img3, metric, psychParams, pixelDims, imgData):
        # given single images generated over numStimPoints number of stimulations, return a numStimPointsx1 vector 
        # of error of each image according to the given metric
        # img1, img2, img3:  3 numPixels x numStimPoints matrices of activity(error is calculated over pixels)
        
        # Initialization
        errs1 = np.zeros((Ts.size,)) # Error Vector for set 1
        errs2 = np.zeros((Ts.size,)) # Error Vector for set 2
        errs3 = np.zeros((Ts.size,)) # Error Vector for set 3


        # caclulate the error of the first set of images, computing a numStimPoints vector of errors:
        for StimIdx in np.arange(Ts.size):
            errs1[StimIdx] = getErr(refImg,img1[:,StimIdx], metric, psychParams, pixelDims, imgData)
            errs2[StimIdx] = getErr(refImg,img2[:,StimIdx], metric, psychParams, pixelDims, imgData)
            errs3[StimIdx] = getErr(refImg,img3[:,StimIdx], metric, psychParams, pixelDims, imgData)
        
        return errs1, errs2, errs3
    
    # Initialization
    numImgs = imgData.numImgs
    Ts      = ssData.Ts
    errMat1 = np.zeros((numImgs,Ts.size))  # holds err vs stimulation vectors for numImgs, from imgSet 1
    errMat2 = np.zeros((numImgs,Ts.size))  # same for imgSet 2
    errMat3 = np.zeros((numImgs,Ts.size))  # same for imgSet 3

    
    
    # For each image i, generate the error vectors for imgs1, imgs2 and store in errMat1, errMat2
    for i in np.arange(numImgs):
        mseImgs = ssData.mseImgSet[:,:,i]
        wmsImgs = ssData.wmsImgSet[:,:,i]
        ssmImgs = ssData.ssmImgSet[:,:,i]
        refImg = imgData.imgSet[:,i]
        errMat1[i,:], errMat2[i,:], errMat3[i,:] = getErrVecs(refImg, Ts, mseImgs.T, wmsImgs.T, ssmImgs.T, metric, psychParams, pixelDims, imgData)

    avgs1 = np.mean(errMat1,0)
    stds1 = np.std(errMat1,0) / np.sqrt(numImgs)
    avgs2 = np.mean(errMat2,0)
    stds2 = np.std(errMat2,0) / np.sqrt(numImgs)
    avgs3 = np.mean(errMat3,0)
    stds3 = np.std(errMat3,0) / np.sqrt(numImgs)
    
    plt.semilogx(Ts,avgs1)
    plt.semilogx(Ts,avgs2)
    plt.errorbar(Ts,avgs1,yerr=stds1,label="MSE")
    plt.errorbar(Ts,avgs2,yerr=stds2,label="wMSE")
    plt.errorbar(Ts,avgs3,yerr=stds3,label="SSIM")

    plt.xlabel('Number of Allowable Stimulations')
    plt.ylabel("Error")
    string = "Relative " + metric + " Error vs. Number of Stimulations"
    plt.title(string)
    plt.ylim([0,1])
    plt.legend()

def actHist(activity,metric,alpha):
    # given a actLength x numImgs set of activity over numImgs images,
    # generate and plot a histogram of sum of activity for each image over
    # the image set. 
    actSum = np.sum(activity,0)
    bins = np.linspace(0, 3000, 100)
    
    if metric.upper() == "MSE":
        label = metric + " Activity, Avg Spikes = %i" %np.mean(actSum)
        color = 'red'
    elif metric.upper() == "WMSE":
        label = metric + "Activity, Avg Spikes = %i" %np.mean(actSum)
        color = 'green'
    else:
        label = metric + "Activity, Avg Spikes = %i" %np.mean(actSum)
        color = 'blue'
    plt.hist(actSum,bins=bins,label=label,alpha=alpha,color=color)
    plt.axvline(x=np.mean(actSum),color=color)
    plt.title('Histogram of Total Number of Spikes Across Image Set',fontSize=18)
    plt.xlabel('Total Number of Spikes',fontsize=18)
    plt.ylabel('Number of Images',fontsize=18)
    plt.legend()
    
def contrast(img, mode='rms'):
    #return the contrast value of a given image. Different modes correspond to different definitions of contrast:
    # Possible modes are: Weber, Michelson, and RMS. 
    
    
    lMax = np.max(img)
    lMin = np.min(img)
        
    if mode.upper() == 'WEB':
        return (lMax-lMin)/lMin
    
    if mode.upper() == 'MICH':
        return (lMax-lMin)/(lMax+lMin)
    
    if mode.upper() == 'RMS':
        flatImg = img.flatten()
        return np.std(flatImg)
    
def relContrast(img, refImg, mode):
    # returns the relative contrast of an original image and its reference.
    # relative contrast is the ratio of the image contrast with that of reference
    return  contrast(img,mode) / contrast(refImg, mode)

def getImgSetContrast(imgSet, refSet, mode):
    # given a numPixel x numImgs set of images, and a same dimension set of reference images,
    # determine the  relative contrast of that set averaged over all the images. 
    # relative contrast is the ratio between img and reference contrasts
    # return the average and standard error of the mean of the relative contrasts
    numImgs = imgSet.shape[1]
    relCons  = np.zeros((numImgs,))
    for i in np.arange(numImgs):
        #relCons[i] = relContrast(imgSet[:,i], refSet[:,i], mode)
        relCons[i] = contrast(imgSet[:,i])
    return np.mean(relCons), np.std(relCons)/np.sqrt(numImgs)

def stimSweepContrastCompar(imgData, ssData, mode):
    # Plot the average relative contrast between 3 sets sweeping over all number of allowed timesteps
    Ts = ssData.Ts
    
    mseRelCons  = np.zeros((Ts.size,))
    mseStds     = np.zeros(mseRelCons.shape)
    wmsRelCons  = np.zeros((Ts.size,))
    wmsStds     = np.zeros(mseRelCons.shape)
    ssmRelCons  = np.zeros((Ts.size,))
    ssmStds     = np.zeros(mseRelCons.shape)

    for i, T in enumerate(Ts):
        mseRelCons[i], mseStds[i] = getImgSetContrast(ssData.mseImgSet[i,:,:], imgData.imgSet,mode)        
        wmsRelCons[i], wmsStds[i] = getImgSetContrast(ssData.wmsImgSet[i,:,:], imgData.imgSet,mode)        
        ssmRelCons[i], ssmStds[i] = getImgSetContrast(ssData.ssmImgSet[i,:,:], imgData.imgSet,mode)        
 
    
    plt.errorbar(Ts,mseRelCons,yerr=(mseStds),label='MSE Contrast')
    plt.errorbar(Ts,wmsRelCons,yerr=(wmsStds),label='wMSE Contrast')
    plt.errorbar(Ts,ssmRelCons,yerr=(ssmStds),label='SSIM Contrast')
    
    plt.xlabel('Number of Allowable Spikes')
    plt.ylabel('Average RMS Contrast')
    plt.xscale("log")
    plt.legend()

    