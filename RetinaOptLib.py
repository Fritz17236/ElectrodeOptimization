# Import Statements

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.io
from scipy import ndimage
import cvxpy as cp
import copy
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
import datetime
import pickle



#  Class Declarations 
class StimSweepData:
    def __init__(self):
        self.Ts        = None 
        self.img_set_mse = None 
        self.img_set_wms = None
        self.img_set_ssm = None
         
        self.act_set_mse = None
        self.act_set_wms = None
        self.act_set_ssm = None
         
        self.rec_set_mse = None
        self.rec_set_wms = None
        self.rec_set_ssm = None
        
class ImageData:
    def __init__(self):
        self.num_imgs    = None
        self.img_set     = None
        self.filtImgSet = None
        self.xs         = None
        self.ys         = None
        self.zoomFac    = None
        self.orig_img    = None
        self.selec_dims  = None
        
class SimData:
    def __init__(self):
        self.acts_mse = None
        self.acts_wms = None
        self.acts_ssm = None

        self.imgs_mse = None
        self.imgs_wms = None
        self.imgs_ssm = None

        self.recons_mse = None
        self.recons_wms = None
        self.recons_ssm = None
        
class RetinaData:
    def __init__(self):
        self.A = None
        self.P = None
        self.e_locs = None
        self.e_map = None
        self.num_pixels = None
        self.dict_len = None

def metric_compar(img_data,sim_params,psych_params, electrode):
    # Compare Error Metrics Side-by-Side for the same set of images    
    img    = img_data.orig_img
    img_set = img_data.img_set
    xs     = img_data.xs
    ys     = img_data.ys
    
    if electrode:
        print('Solving for Electrode Activities...')
    else:
        print('Solving for Cellular Activities...')    
    
    print('MSE Activity Reconsruction:')
    imgs_mse, acts_mse = recons_img_set(img_set,img_data, sim_params, psych_params, "mse", electrode)
    print('wMSE Activity Reconstruction')
    imgs_wms, acts_wms = recons_img_set(img_set,img_data, sim_params, psych_params, "wms", electrode)
    print('SSIM Activity Reconstruction')
    imgs_ssm, acts_ssm = recons_img_set(img_set,img_data, sim_params, psych_params, "ssm", electrode)
    
    print('Activities Solved. Rebuilding Images ...')
    pixel_dims = sim_params["pixel_dims"]
    
    recons_mse = rebuild_img(img,imgs_mse,xs,ys,pixel_dims,psych_params)
    recons_wms = rebuild_img(img,imgs_wms,xs,ys,pixel_dims,psych_params)
    recons_ssm = rebuild_img(img,imgs_ssm,xs,ys,pixel_dims,psych_params)

    print('Images rebuilt.')
    print('Simulation Complete')

    sim_data = SimData()
    sim_data.acts_mse = acts_mse
    sim_data.acts_wms = acts_wms
    sim_data.acts_ssm = acts_ssm

    sim_data.imgs_mse = imgs_mse
    sim_data.imgs_wms = imgs_wms
    sim_data.imgs_ssm = imgs_ssm

    sim_data.recons_mse = recons_mse
    sim_data.recons_wms = recons_wms
    sim_data.recons_ssm = recons_ssm
    
    return sim_data
    
def recons_img_set(img_set, img_data, sim_params, psych_params, metric, electrode):
    # Given a set of images (img_set) as a 2d Matrix, and a metric, reconstruct
    # the image set according to the given image in parallel according to the available cpu cores    
    if electrode:
        activity_length = sim_params["P"].shape[1]
    else:
        activity_length = sim_params["A"].shape[1]
    
    num_pixels = img_set.shape[0]
    num_imgs   = img_set.shape[1]
    
    # convert img_set to list for parallelization
    img_list = []
    for i in np.arange(num_imgs):
        img_list.append(img_set[:,i])
     
    num_cores = mp.cpu_count()-1 # leave at least one open
    
    # run reconstructions in parallel
    results = np.asarray(Parallel(n_jobs=num_cores)(delayed(act_solver)(i,img_data,sim_params,psych_params,metric,electrode) for i in tqdm(img_list)))

    #convert results back to 2 variables separating activity and the reconstructed image
    imgs = np.zeros((num_pixels,num_imgs))
    acts = np.zeros((activity_length,num_imgs))
    for i in np.arange(num_imgs):
        imgs[:,i] = results[i,0]
        acts[:,i] = results[i,1]
    return imgs, acts   

def load_raw_img(file_name):
    # given a filename for an rgb image, load and preprocess the image by doing the following:
    # convert to grayscale --> zero-mean --> normalize to +/-5 .5 intensity
    img = plt.imread(file_name)
    img = np.sum(img,2)/3
    
    # Normalize to +/- 1 intensity range and zero mean
    img -= np.mean(img)
    img = img / (2*np.max((np.abs(img))) ) 

    return img

def save_data(data,file_name):
    # save the data structure passed to the filename given
    session_title = file_name+"_"+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")+'.dat'
    with open(session_title, 'wb') as file_handle:
        pickle.dump(data, file_handle)
        
def load_data(file_name):
    return pickle.load(open(file_name,'rb'))

def load_raw_data(file_name):
    
    # loads raw data from matlab .mat file
    data = scipy.io.loadmat(file_name)
    A   = (data['stas'].T)
    # Select 20x20 slice of Pixels from A to fix border issues
    P,e_map = prune_dict(data['dictionary'].T,data['ea'])

    e_locs = data['elec_loc']  # electrode locations
    dictLength = P.shape[1]
    num_pixels = A.shape[0]

    # Trim A to a 20 x 20 Square
    Aslice = np.zeros((400,A.shape[1]))
    for col in np.arange(8,28):
        Aslice[(col-8)*20:(col-7)*20,:] = A[30+80*col:50+80*col,:]

    A,P = prune_decoder(Aslice,P)
    
    ret_data = RetinaData()
    ret_data.A = A
    ret_data.P = P
    ret_data.e_locs = e_locs
    ret_data.e_map  = e_map
    ret_data.num_pixels = num_pixels
    ret_data.dict_len = dictLength
    ret_data.num_cells = A.shape[1]
    
    return ret_data
    
def dct2(a):
    # 2D Discrete Cosine Transform and Its Inverse
    l_dim = a.shape[0]
    r_dim = a.shape[1]
    # build the matrix
    n, k = np.ogrid[1:2*l_dim+1:2, :l_dim]
    m, l = np.ogrid[1:2*r_dim+1:2, :r_dim]
    Dl = 2 * np.cos(np.pi/(2*l_dim) * n * k)
    Dr = 2 * np.cos(np.pi/(2*r_dim) * m * l)
    return (Dl.T @ a @ Dr)

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def gen_stixel( height, width, s ):
# % genStiheightelImg: Generate a zero-mean white-noise stixelated image of specified
# % dimension.
# %   This function generates an image of size specified bwidth (height,
# %   width), and divides the image into s height s squares
# %   each stiheightel having the same Gaussian Generated white noise value. 
# %   The Gaussian values range from [-0.5, 0.5]. 


    height_stixel = np.floor(height/s).astype(int)  #% full number of stixels
    width_stixel = np.floor(width/s).astype(int)
    rem_width = width - s*width_stixel #% remainder that specifies padding
    rem_height = height - s*height_stixel

    #% Depending whether there is remainder after full stixels, determine
    #% if we need to pad. Otherwise, set pad variables to 0
    if ( rem_width != 0): 
        wpad = 1
    else: 
        wpad = 0

    if (rem_height != 0):
        hpad = 1
    else: 
        hpad = 0


    # pad the image to fit to remainder size
    img = np.zeros((height+rem_height,width+rem_width)) # %initialize image

    #% Fill in the full stixel 
    for i in np.arange(height_stixel+hpad+1):   # For each stixel block
        for j in np.arange(width_stixel+wpad+1):
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

def flat_DCT(pixel_dims):
    # build and return a flattened dct matrix specifically for (80,40) images flattened with fortran ordering
    # Build 80 x 40 2D DCT-II Matrix
    num_pixels = pixel_dims[0]*pixel_dims[1]
    D1 = np.zeros((num_pixels,num_pixels))
    D2 = np.zeros((num_pixels,num_pixels))
    # build a flattened form of a  1d DCT matrix 
    l_dim = pixel_dims[0]
    r_dim = pixel_dims[1]
    n, k = np.ogrid[1:2*l_dim+1:2, :l_dim]
    m, l = np.ogrid[1:2*r_dim+1:2, :r_dim]
    Dl = 2 * np.cos(np.pi/(2*l_dim) * n * k)
    Dr = 2 * np.cos(np.pi/(2*r_dim) * m * l)

#     imRows = 80
#     imCols = 40
    # build D1
    for i in np.arange(l_dim):
        for j in np.arange(r_dim):
            D1[j*l_dim + i,j*l_dim:(j+1)*l_dim] = Dl.T[i,:]


    # build D2
    for i in np.arange(r_dim):
        for k in np.arange(l_dim):
            for j in np.arange(r_dim):
                D2[k+j*pixel_dims[0],i*pixel_dims[0]+k] = Dr[i,j]
    D = D2@D1
    return D

def flatW(psych_params,pixel_dims,img_data): 
    # build and return a flattned W matrix for images (img) flattned with fortran ordering
    # (1/2) * N / D where D is horizontal degrees, N is number of blocks 
    XO = psych_params["XO"]
    N  =  int(img_data.orig_img.shape[0]/img_data.selec_dims[0]) # number of selection blocks (number of samples of DC terms of each subImage)
    offset = (1/2) * (N / XO)
    Wp = csf(psych_params,pixel_dims,offset=offset) #offset frequency b
    flatW = np.reshape(Wp,(pixel_dims[0]*pixel_dims[1],),order='F')
    W = np.diag(flatW)
    return W

def csf(psych_params,pixel_dims,offset=0):
    # given a peak sensitivity frequency pf, and a psychophysically determined pixels-per-degree of viusal field ppd,
    # and and image, return a mask that has the same shape as the image and applies a weighting to each pixel in the image
    # according to the contrast sensitivity function 
    def getNg(psych_params):
        e = psych_params["e"]
        Ng0 = psych_params["Ng0"]
        eg = psych_params["eg"]
        term1 = .85 / (1 + (e/.45)**2)
        term2 = .15 / (1 + (3/eg)**2)
        return Ng0*term1*term2
    
    def Mopt(f,psych_params):
        #given a spatial frequency f and psychophysical parameters,
        # return the frequnecy filetered by the optical transfer function
        # of the retina
        sigma00 = .30           # Non-retinal optical linespread constant (arcmin)
        sigmaRet = 1 / np.sqrt(7.2*np.sqrt(3)*getNg(psych_params))
        sigma_0 = np.sqrt(sigma00**2 + sigmaRet**2) # (arcmin) std deviation of linespread (function of eccentricity)
        Cab = .08    # (arcmin / mm ) dimensionality constant
        d = psych_params["d"] # pupil size in mm
        sigma = np.sqrt(sigma_0**2 + (Cab*d)**2)
        return np.exp(-2*(np.pi**2)*((sigma/60)**2)*(f**2))
        
    def intTerm(f,psych_params):
        # given spatial frequency f and psychophysical paratmeters,
        # calculate the visual-angle integration term of the CSF
        e = psych_params["e"]
        Xmax = 12   # (degrees) maximum visual integration area  
        term1 = .85 / (1 + (e/4)**2)
        term2 = .15 / (1 + (e/12)**2)
        Xmax=Xmax*(term1+term2)**-.5
        Ymax = Xmax
        Nmax = 15  # (cycles) maximum number of cycles function of eccentriicty
        XO = psych_params["XO"]
        YO = psych_params["YO"]
        
        term1 = (.5*XO)**2 + 4*e**2
        term2 = (.5*XO)**2 + e**2
        NmaxFac = term1/term2
        
        return 1/(XO*YO) + 1/(Xmax*Ymax) + NmaxFac*(f/Nmax)**2
    
    def illumTerm(psych_params):
        #given spatial frequency f and psychophysical parameters,
        # calculate the  illumance term of the CSF
        n = .03  #quantum efficiency term (function of eccentricity)
        e = psych_params["e"]
        term1 = .4 / (1 + (e/7)**2)
        term2 = .48 / (1 + (e/20)**2) 
        n = n*(term1 + term2 +.12)
        p = 1.24 # photon conversion factor (function of incident light)
        d = psych_params["d"]
        L = psych_params["L"]
        E = np.pi/4 * d**2 * L * (1 - (d/9.7)**2 + (d/12.4)**4)
        return 1/(n*p*E)
        
    def inhibTerm(f,psych_params):
        # given spatial frequency f and psychophysical parameters,
        # calculate the lateral inhibition term of the CSF
        Ng0 = psych_params["Ng0"]
        e = psych_params["e"]
        u0 = 7  #(cycles/deg) stop frequency of lateral inhibition
        term1 = .85 / (1 + (e/4)**2)
        term2 = .13 / (1 + (e/20)**2)
        u0 = u0 * (getNg(psych_params)/Ng0)**.5 * (term1 + term2 + .02)**-.5
        return 1 - np.exp(-(f/u0)**2)
    
    k  = psych_params["k"]
    X0 = psych_params["elec_XO"]
    T  = psych_params["T"]
    sfRes = 1/pixel_dims[0]
    Ng = getNg(psych_params)
    Ng0 = psych_params["Ng0"]
    ph0= 3*10**-8*Ng0/Ng  # neural noise term (sec / deg^2)
    fxx,fyy = np.meshgrid(np.arange(pixel_dims[1]),np.arange(pixel_dims[0]))
    ppd = pixel_dims[0]/X0
    fs = (sfRes * ppd *((fxx)**2+(fyy)**2)**.5  ) + offset
    
    num   = Mopt(fs,psych_params) / k
    
    if not psych_params["binocular"]:
        num = num /  np.sqrt(2)
    
    denom = np.sqrt( 
        (2/T)
        *intTerm(fs,psych_params)
        *(illumTerm(psych_params) + ph0 / inhibTerm(fs,psych_params)) 
    )
    W = np.divide(num,denom)
    return W

def prune_decoder(A,P):
    # remove the columns of A corresponding to the cells which don't change the image
    # reconstruction
    # also delete the corresponding rows of P
    # if a column of A has a norm of 0 it must be all 0, so delete the column. 
    delList = []
    for i in np.arange(A.shape[1]):
        if (np.linalg.norm(A[:,i])) <= 10**-6:
            delList.append(i)
            
    return np.delete(A,delList,axis=1),np.delete(P,delList,axis=0)

def prune_dict(P,eActs,threshold=.05):
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
    
    return pp, eActs
    # only do if greedy: return np.hstack((pp,np.zeros((pp.shape[0],1)))),  np.vstack((eActs,np.asarray(np.zeros((1,eActs.shape[1])))))
    
def mse(A,B):
    #flatten if not flat
    if A.ndim > 1:
        flatA = A.flatten()
        flatB = B.flatten()
        
        return (flatA-flatB).T@(flatA-flatB)/flatA.size
    else:
        return (A-B).T@(A-B)/A.size

def jpge(A,B,psych_params,pixel_dims, img_data):
    try:
        D = flat_DCT(pixel_dims)
        diffImg = A - B
        if diffImg.ndim is not 1: #flatten image if not already flattened
            diffImg = diffImg.flatten
        W = flatW(psych_params, pixel_dims, img_data)
        W = W/np.max(W)
        return np.linalg.norm(W@D@diffImg)**2 / A.size
    except:
        W = csf(psych_params, pixel_dims)
        W = W/np.max(W)
        return np.linalg.norm(np.multiply(W,dct2(diffImg))) 

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
    num_pixels = X.shape[0]
    meanX = np.mean(X)
    meanY = np.mean(Y)
    lum = (2*meanX*meanY + C1) / (meanX**2 + meanY**2 + C1)

    stdX = np.std(X)
    stdY = np.std(Y)

    con = ( 2*stdX*stdY + C2) / (stdX**2 + stdY**2 + C2)
    stdXY = (1 / (num_pixels-1)) * np.sum( np.multiply((X-meanX),(Y - meanY)) ) 

    srt   = (stdXY + C3) / (stdX*stdY + C3)
    ssim = lum**alpha * con**beta * srt**gamma
    return ssim

def get_elec_angs(smps,stixelSize, eyeDiam, pixel_dims):
    # Given a set of psychophysical parameters,the reconstructing electrode array
    # smps: stimulus monitor pixel size: the size of a single monitor pixel in lab setup on the retina (microns)
    # stixelSize:  the stixel size,which is the square root of the number of monitor pixels grouped together 
    #      to form a single STA pixel (one STA pixel is stixelSize x stixelSize monitor pixels)
    # eyeDiam: the Emmetropia diameter of the eye in milimeters
        
    
    retArea  = ( # Retinal area in milimeters
        pixel_dims[0]*smps*stixelSize/1000,
        pixel_dims[1]*smps*stixelSize/1000
    )
    
    elecVisAng = ( # Visual Angle Spanned by the Electrode Reconstruction 
        np.rad2deg(np.arctan(retArea[0]/eyeDiam)),
        np.rad2deg(np.arctan(retArea[1]/eyeDiam))
    )
    
    return elecVisAng

def preprocess_img(img,psych_params,sim_params):
    # Given psychophysically determined viewing angles for the visual
    # scene, the image, and the dimensions of the stimulus reconstruction in 
    # pixels, tile the image into a set of subimages, where each subimage
    # covers precisely elecVisAng[0] x elecVisAng[1] degrees of the visual
    # scene. Resample these tiled images to have the same dimensions as the 
    # stimulus pixel (pixel_dims) for reconstruction.
    # elecVisAng[0]/objVisAngle[0] = selection/  img.shape[0]
    
    def tileImage(img,pixel_dims):
        # Given an mxn image and pixel_dims, tile the image by splitting it into 
        # num_imgs subimages obtained by taking pieces of size pixel_dims from the original image, stacking,
        # and then returning the images, as well as the x & y locations of the top left corner of each image
        
        def fitToDims(img,pixel_dims):
            # Given an mxn image, fit the image to the given dimension by padding it with zeros. 
            # This imamge assumes m<= pixelDIms[0] and/or n <= pxiel_dims[1]
            fitImg = np.zeros(pixel_dims)
            fitImg[0:img.shape[0],0:img.shape[1]] = img
            return fitImg       
        
        print('Tiling Image ...')
        x = 0
        y = 0 # initial location is top left of image
        subImgs = np.zeros((pixel_dims[0]*pixel_dims[1],0))
        xs = np.asarray([])
        ys = np.asarray([])

        while y <= img.shape[1]-pixel_dims[1]:
            # sweep horizontally. if x >= img.shape set x to 0 and update y
            if x >= img.shape[0]-pixel_dims[0]: 
                x = 0
                y += int(pixel_dims[0])

            selection = fitToDims(img[x:x+pixel_dims[0],y:y+pixel_dims[1]],pixel_dims)
            selection = np.reshape(selection,(pixel_dims[0]*pixel_dims[1],1),order='F')
            if not np.all(selection==0):
                subImgs = np.concatenate((subImgs,selection),1)
                xs = np.append(xs,[x])
                ys = np.append(ys,[y])
                x += int(pixel_dims[0])

        print('Tiled Image')        
        return subImgs, xs, ys

    pixel_dims = sim_params["pixel_dims"]
    selec_dims = get_selection_dims(psych_params,img)

    img_set, xs, ys        = tileImage(img,selec_dims)

    num_imgs = img_set.shape[1]
    resImgSet = np.zeros((pixel_dims[0]*pixel_dims[1],num_imgs))

    # go through each image, resample it and store it in resImgSet
    for i in np.arange(num_imgs):
        resImgSet[:,i],zoomF = resample(img_set[:,i],selec_dims,pixel_dims)
    img_data = ImageData()
    img_data.num_imgs = num_imgs
    img_data.img_set = resImgSet
    img_data.xs = xs
    img_data.ys = ys
    img_data.zoomFac = zoomF
    img_data.orig_img    = img
    img_data.selec_dims      = selec_dims
    img_data.resampled_img = rebuild_img(img, resImgSet, xs, ys, pixel_dims, psych_params)
    
    return img_data

def get_selection_dims(psych_params,img):
    XO = psych_params['XO']
    elec_XO = psych_params['elec_XO']
    selectionSize = int(np.ceil(elec_XO/XO * img.shape[1]))

    # select the equivalent of elecVisangx elecVisAng pixels from the image
    selec_dims = (selectionSize,selectionSize)
    return selec_dims

def act_solver(img,img_data,sim_params,psych_params,mode,electrode):
    # Reconstruct an image according to the error metric specified by "mode"
    # Input: img : the image to be reconstructed, dims = psych_params["pixel_dims"]
    #        sim_params : a simulation parameters dictionary 
    #        psych_params: a psychophysical parameters dictionary
    #        mode : a string specifying the particular error metric being used
    #        electrode : a boolean specifying whether to reconstruct according ot optimal cell 
                # activities or using th electrode stimulation dictionary 
    #Subfunctions:
    def var_term(sim_params,Phi, x):
    # Return the cost function associate with the variance component of the reconstruction
    # error. Only used in the case that electrode is true
    # Inputs: 
    #     sim_params: the simulatin parameters dictionary object
    #     electrode: boolean indicating whether performing optimal cellular or electrode dictionary recons
    #     x : the cvx variable representing the activity vector object that is being solved for
        var_mtx = variance_matrix(sim_params, Phi)
        return  cp.sum(var_mtx@x)
    def variance_matrix(sim_params, Phi):
        # get the variance matrix that maps dictionary selections to 
        # total variance of the reconstructed image according to the given
        # metric, Phi
        P = sim_params["P"]
        V = np.zeros(P.shape)
        for j in np.arange(P.shape[1]):
            V[:,j] = np.multiply(P[:,j],(1-P[:,j]))
        return np.multiply(Phi,Phi)@V
    
    def recons_ssm(img, sim_params, electrode, epsilon = 10**-2):
        # use bisection search to solve for an optimal-SSIM reconstruction

        def find_feasible(y,alpha,sim_params, electrode ):
            # Return a feasible solution to the SSIM optimization problem
            # Using cvxpy solves the constrained feasability problem that is a transformation of the SSIM
            # optimization problem.

            def cvxineq(a,y,x,Phi):
                # a convex inequality to evaluate feasability
                return (1-a)*cp.sum_squares(y-Phi@x)-2*a*(Phi@x).T@y

            A = sim_params["A"]
            P = sim_params["P"]

            if electrode:
                x = cp.Variable(P.shape[1])
                Phi = A@P
                cost = var_term(sim_params,Phi, x)
            else:
                x = cp.Variable(A.shape[1])
                cost = 1
                Phi = A

            T = sim_params["num_stims"]
            N = sim_params["max_act"]
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

        A = sim_params["A"]
        P = sim_params["P"]
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
            isFeasible, xCurr = find_feasible(y, alpha, sim_params, electrode)

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
    
    A = sim_params["A"]
    P = sim_params["P"]
    T = sim_params["num_stims"]
    N = sim_params["max_act"]
    pixel_dims = sim_params["pixel_dims"]

    y = img

    if electrode:
        x = cp.Variable(P.shape[1])
    else:
        x = cp.Variable(A.shape[1])

    if mode == "mse": 
        if electrode:
            cost = cp.sum_squares(y-A@P@x) + var_term(sim_params,A,x)
        else:
            cost = cp.sum_squares(y-A@x)
    
    elif mode == "wms":
        W = flatW(psych_params,sim_params["pixel_dims"],img_data)
        D = flat_DCT(pixel_dims)
        if electrode:
            cost = cp.sum_squares(W@D@(y-A@P@x)) + var_term(sim_params, W@D@A, x)
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
        return recons_ssm(img, sim_params, electrode)
        
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
    
def num_stim_sweep(img_data,sim_params,psych_params,electrode):
    # Given a set of images, reconstruct each image using all metric and sweep over the number of allowable stimulations.
    # run a metric comparison simulation over a specified number of stimulation times
    
    Tres = 16
    Ts   = np.logspace(0,5,Tres)
    
    img_set_mses = []
    img_set_wmss = []
    img_set_ssms = []
    
    act_set_mses = []
    act_set_wmss = []
    act_set_ssms = []
    
    rec_set_mses = []
    rec_set_wmss = []
    rec_set_ssms = []
    
    for Tidx, T in enumerate(Ts):
        print("T: %i;  %i/%i"%(T, Tidx+1, Ts.size))
        sim_params["num_stims"] = T
        
        sim_data  =  metric_compar(img_data,sim_params,psych_params, electrode)
        
        img_set_mses.append(sim_data.imgs_mse)
        img_set_wmss.append(sim_data.imgs_wms)
        img_set_ssms.append(sim_data.imgs_ssm)

        act_set_mses.append(sim_data.acts_mse)
        act_set_wmss.append(sim_data.acts_wms)
        act_set_ssms.append(sim_data.acts_ssm)

        rec_set_mses.append(sim_data.recons_mse)
        rec_set_wmss.append(sim_data.recons_wms)
        rec_set_ssms.append(sim_data.recons_ssm)
        
        ss_data = StimSweepData()
        ss_data.Ts        = Ts
        ss_data.img_set_mse = np.asarray(img_set_mses)
        ss_data.img_set_wms = np.asarray(img_set_wmss)
        ss_data.img_set_ssm = np.asarray(img_set_ssms)
        
        ss_data.act_set_mse = np.asarray(act_set_mses)
        ss_data.act_set_wms = np.asarray(act_set_wmss)
        ss_data.act_set_ssm = np.asarray(act_set_ssms)
        
        ss_data.rec_set_mse = np.asarray(rec_set_mses)
        ss_data.rec_set_wms = np.asarray(rec_set_wmss)
        ss_data.rec_set_ssm = np.asarray(rec_set_ssms)
        
    return ss_data

def resample(img,curr_dims,desiredDims):
    # given a (curr_dims[0]*curr_dims[1] x 1 ) image vector, resample the image
    # to fit to desired dims and return this image flatted into a 
    #(desiredDims[0],desiredDims[1] x 1) image vector
    currImg = np.reshape(img,curr_dims,order='F')

    # desiredDims[0] = zoomFac * curr_dims[0]
    zoomFac =  desiredDims[0]/curr_dims[0]
    zImg = ndimage.zoom(currImg,zoomFac)
    return np.reshape(zImg,(desiredDims[0]*desiredDims[1],),order='F'),zoomFac

## Visulization Functions

def pca_analysis(imgs,acts_mse,acts_wms, acts_ssm):
    # given a set of images, electrode locations, and their dictionary reconstructions,
    # calculate correlations (if any) of electrode activity across the set of images 
    num_imgs = imgs.shape[1]

    data = np.hstack((acts_mse,acts_wms,acts_ssm))
    covdata = np.cov(data) # covariance matrix
    wdata,vdata = np.linalg.eig(covdata) # eigen decomposition of covariance matrix

    # project each activity vector onto the 3 respective components
    (data1,data2,data3) = project_PCA(data,vdata[:,0],vdata[:,1],vdata[:,2])

    markerSize = 1

    # 3D Scatter Plot of Image Data Projected onto principal Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[0:num_imgs-1],data2[0:num_imgs-1],data3[0:num_imgs-1],label='MSE',s=markerSize)
    ax.scatter(data1[num_imgs:2*num_imgs-1],data2[num_imgs:2*num_imgs-1],data3[num_imgs:2*num_imgs-1],label='wMSE',s=markerSize)
    ax.scatter(data1[2*num_imgs:],data2[2*num_imgs:],data3[2*num_imgs:],c='red',label='SSIM',s=markerSize)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.legend(loc='upper right')
    plt.show()
    
    xLims = 2
    yLims = 1

    # 2D Plot Projected Onto Principal Axes
    plt.figure()
    plt.scatter(data1[0:num_imgs-1],data2[0:num_imgs-1],s=markerSize,label='MSE')
    plt.scatter(data1[num_imgs:2*num_imgs-1],data2[num_imgs:2*num_imgs-1],s=markerSize,label='JPG')
    plt.scatter(data1[2*num_imgs:],data2[2*num_imgs:],c='red',label='SSIM',s=markerSize)
    plt.title('PC1 & PC2')
    plt.legend()
    plt.xlim([-xLims,xLims])
    plt.ylim([-yLims,yLims])
    plt.savefig('PC1PC2Electrode.jpg',bbox_inches='tight')
    plt.show()


    plt.figure()
    plt.scatter(data1[0:num_imgs-1],data3[0:num_imgs-1],s=markerSize,label='MSE')
    plt.scatter(data1[num_imgs:2*num_imgs-1],data3[num_imgs:2*num_imgs-1],s=markerSize,label='JPG')
    plt.scatter(data1[2*num_imgs:],data3[2*num_imgs:],c='red',label='SSIM',s=markerSize)
    plt.title('PC1 & PC3')
    plt.legend()
    plt.xlim([-xLims,xLims])
    plt.ylim([-yLims,yLims])
    plt.savefig('PC1PC3Electrode.jpg',bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.scatter(data2[0:num_imgs-1],data3[0:num_imgs-1],s=markerSize,label='MSE')
    plt.scatter(data2[num_imgs:2*num_imgs-1],data3[num_imgs:2*num_imgs-1],s=markerSize,label='JPG')
    plt.scatter(data2[2*num_imgs:],data3[2*num_imgs:],c='red',label='SSIM',s=markerSize)
    plt.xlim([-xLims,xLims])
    plt.ylim([-yLims,yLims])
    plt.title('PC2 & PC3')
    plt.legend()
    plt.savefig('PC2PC3Electrode.jpg',bbox_inches='tight')
    plt.show()

    return wdata,vdata

def project_PCA(data,pc1,pc2,pc3):
    #given a data_dim x num_pts matrix of data, and 3 data_dim principal component vectors,
    #return a num_pts vector containing the scalar projection of the data onto the vector at each numpt

    num_pts = data.shape[1]
    proj1 = np.zeros((num_pts,))
    proj2 = np.zeros((num_pts,))
    proj3 = np.zeros((num_pts,))

    for i in np.arange(num_pts):
        proj1[i] = np.dot(data[:,i],pc1)/(np.linalg.norm(pc1)*np.linalg.norm(data[:,i]))
        proj2[i] = np.dot(data[:,i],pc2)/(np.linalg.norm(pc2)*np.linalg.norm(data[:,i]))
        proj3[i] = np.dot(data[:,i],pc3)/(np.linalg.norm(pc3)*np.linalg.norm(data[:,i]))
    return (proj1,proj2,proj3)

def disp_elec_act(elecActs,sim_params,color='blue'):
    # Given a vector of electrode activities, a vector of (x,y) electrode locations, and a 2xnumElectrode (contained in simparams)
    # matrix of electrode numbers for each element, sum the total current passing through each electrode, 
    # and display it in a scatter plot
    e_locs = sim_params["e_locs"]
    
    totalCurr = get_total_curr(elecActs,sim_params)
    
    plt.scatter(e_locs[:,0],e_locs[:,1],alpha=.5,s=totalCurr,c=color)
    plt.scatter(e_locs[:,0],e_locs[:,1],alpha=.5,s=1,c='black')
    plt.title('Total Electrode Current: %i nC' %np.sum(totalCurr))
    plt.axis('equal')
    plt.xlabel('Horizontal Position (um)')
    plt.ylabel('Vertical Position (um)')

def get_total_curr(acts, sim_params):
    # given a vector of activities, get the total current from that 
    # set of activities that is passed through the electrode array
    return acts.T@sim_params["e_map"][:,0]

def angle_between(vecA,vecB):
    # return the cosine of the angle between to vectors:
    ang = np.arccos(np.dot(vecA,vecB)/(np.linalg.norm(vecA)*np.linalg.norm(vecB)))
    return ang

def rebuild_img(img,img_set,xs,ys,pixel_dims,psych_params,zeroMean=False): 
    # input params:
    # img: an mxn original image matching the desired image dimensions
    # img_set a (num_pixels x num_imgs) matrix of flattened subimages
    # xs a num_imgs vector of x positions for the upper left loc of each subimage
    # ys a num_imgs vector of y positions for the upper left loc of each subimage
    # returns: a reconstructed image having the same dimensions of the original image (img),
    #      built from the set of subimages
    
    # initialize image
    recons    = np.zeros(img.shape)
    xs = xs.astype(int)
    ys = ys.astype(int)
    
    #calc selection dims
    selec_dims = get_selection_dims(psych_params,img)
    # iterate through each (x,y) coordinate a
    for i in np.arange(xs.shape[0]): 
        # if dims not correct, resample to selectiondims
        if (pixel_dims[0] != selec_dims[0] or pixel_dims[1] != selec_dims[1]):
            # resample image
            resampled_img = resample(img_set[:,i],pixel_dims,selec_dims)[0]
            
        reconsImg = np.reshape(resampled_img,selec_dims,order='F')
        

        # only add to image if exactly zero pixel
        x = xs[i]
        y = ys[i]
        
        
        selection = recons[x:x+selec_dims[0],y:y+selec_dims[1]]
        reconsSel = reconsImg[0:selection.shape[0],0:selection.shape[1]]
        
        # If zeromean, make the subimage zero mean, then add the average intensity for the whole image
        if zeroMean:
            reconsSel -= np.mean(reconsSel) 
        recons[x:x+selection.shape[0],y:y+selection.shape[1]] += reconsSel
    
    return recons  

def act_angle_plot(acts_mse, other_acts, other_label, r_lim, a=.5):
    # given two actLength x num_imgs matrices of activity over a num_imgs set of images,
    # calculate the angle between each pair of activities over all image and plot this
    # angle on a polar graph, where the radius of each data point is the sum of the 
    # activity of the other activity minus the sum of the MSE activity.
    
    num_imgs =  acts_mse.shape[1]
    angles = np.zeros((num_imgs,))
    radii  = np.zeros((num_imgs,))
    
    for imgNum in np.arange(num_imgs):
        angles[imgNum] = angle_between(acts_mse[:,imgNum],other_acts[:,imgNum])
        radii[imgNum]  =(np.sum(other_acts[:,imgNum])-np.sum(acts_mse[:,imgNum])) 
    
    pos = radii >= 0
    neg = radii <  0 

    avg_ang = np.mean(angles)
    str_rads = np.linspace(0,r_lim,angles.size)

    
    
    plt.polar(avg_ang*np.ones(angles.shape),str_rads,'-',c='black',
              linewidth=5,
              alpha=.5)
    plt.polar(angles[pos],radii[pos],'ro',c='red',alpha=a)
    plt.polar(angles[neg],np.abs(radii[neg]),'ro',c='blue',alpha=a)
    ax = plt.gca()
    ax.set_rlim(0,r_lim)
    plt.title('Angle Between MSE & '+other_label+
              ' Activity \n Avg Angle: %i deg'%np.rad2deg(avg_ang))
    return

def plot_stim_comparison(img_data, ss_data, metric, psych_params, pixel_dims):
    # plot the error of given image sets versus number of stimulations according to  a specified metric, average over
    #  all images for each number of stimulations data point
    # Input: img_data: img_data object created by preprocess_img function
    #        metric: metric according to which activities 1 and 2 will be plotted/calculated: "mse", "wms", or "ssm"
    #        psych_params: a psychophysical parameters object 
    #        ss_data: Stimulation Sweep Data output from num_stim_sweep Function
    #        pixel_dims:  the dimensions of the unflattened image (e.g. (20, 20) )
    
    def get_error(ref_img, img, metric,psych_params,pixel_dims, img_data):
        # given two images, calculate the error according to the given metric
        # metric = "mse", "jpg", or "ssm"
        if metric.upper() == "MSE":
            return mse(ref_img, img)/mse(ref_img, np.zeros(ref_img.shape))
        elif metric.upper() == "WMS" :
            return jpge(ref_img, img, psych_params, pixel_dims, img_data) / jpge(ref_img, np.zeros(ref_img.shape), psych_params, pixel_dims, img_data)
        elif metric.upper() == "SSIM":
            return ((1-SSIM(ref_img, img))/2) / ((1-SSIM(ref_img,np.zeros(ref_img.shape)/2)))
        else: 
            raise Exception("Bad Metric Passed to get_error")
    
    def get_error_vecs(ref_img, Ts, img1, img2, img3, metric, psych_params, pixel_dims, img_data):
        # given single images generated over numStimPoints number of stimulations, return a numStimPointsx1 vector 
        # of error of each image according to the given metric
        # img1, img2, img3:  3 num_pixels x numStimPoints matrices of activity(error is calculated over pixels)
        
        # Initialization
        errs1 = np.zeros((Ts.size,)) # Error Vector for set 1
        errs2 = np.zeros((Ts.size,)) # Error Vector for set 2
        errs3 = np.zeros((Ts.size,)) # Error Vector for set 3


        # caclulate the error of the first set of images, computing a numStimPoints vector of errors:
        for stim_idx in np.arange(Ts.size):
            errs1[stim_idx] = get_error(ref_img,img1[:,stim_idx], metric, psych_params, pixel_dims, img_data)
            errs2[stim_idx] = get_error(ref_img,img2[:,stim_idx], metric, psych_params, pixel_dims, img_data)
            errs3[stim_idx] = get_error(ref_img,img3[:,stim_idx], metric, psych_params, pixel_dims, img_data)
        
        return errs1, errs2, errs3
    
    # Initialization
    num_imgs = img_data.num_imgs
    Ts      = ss_data.Ts
    err_mat_1 = np.zeros((num_imgs,Ts.size))  # holds err vs stimulation vectors for num_imgs, from img_set 1
    err_mat_2 = np.zeros((num_imgs,Ts.size))  # same for img_set 2
    err_mat_3 = np.zeros((num_imgs,Ts.size))  # same for img_set 3

    
    
    # For each image i, generate the error vectors for imgs1, imgs2 and store in err_mat_1, err_mat_2
    for i in np.arange(num_imgs):
        imgs_mse = ss_data.img_set_mse[:,:,i]
        imgs_wms = ss_data.img_set_wms[:,:,i]
        imgs_ssm = ss_data.img_set_ssm[:,:,i]
        ref_img = img_data.img_set[:,i]
        err_mat_1[i,:], err_mat_2[i,:], err_mat_3[i,:] = get_error_vecs(ref_img, Ts, imgs_mse.T, imgs_wms.T, imgs_ssm.T, metric, psych_params, pixel_dims, img_data)

    avgs_1 = np.mean(err_mat_1,0)
    stds_1 = np.std(err_mat_1,0) / np.sqrt(num_imgs)
    avgs_2 = np.mean(err_mat_2,0)
    stds_2 = np.std(err_mat_2,0) / np.sqrt(num_imgs)
    avgs_3 = np.mean(err_mat_3,0)
    stds_3 = np.std(err_mat_3,0) / np.sqrt(num_imgs)
    
    plt.semilogx(Ts,avgs_1)
    plt.semilogx(Ts,avgs_2)
    plt.errorbar(Ts,avgs_1,yerr=stds_1,label="MSE")
    plt.errorbar(Ts,avgs_2,yerr=stds_2,label="wMSE")
    plt.errorbar(Ts,avgs_3,yerr=stds_3,label="SSIM")

    plt.xlabel('Number of Allowable Stimulations')
    plt.ylabel("Error")
    string = "Relative " + metric + " Error vs. Number of Stimulations"
    plt.title(string)
    plt.ylim([0,1])
    plt.legend()

def act_hist(activity,metric,alpha):
    # given a actLength x num_imgs set of activity over num_imgs images,
    # generate and plot a histogram of sum of activity for each image over
    # the image set. 
    act_sum = np.sum(activity,0)
    bins = np.linspace(0, 3000, 100)
    
    if metric.upper() == "MSE":
        label = metric + " Activity, Avg Spikes = %i" %np.mean(act_sum)
        color = 'red'
    elif metric.upper() == "WMSE":
        label = metric + "Activity, Avg Spikes = %i" %np.mean(act_sum)
        color = 'green'
    else:
        label = metric + "Activity, Avg Spikes = %i" %np.mean(act_sum)
        color = 'blue'
    plt.hist(act_sum,bins=bins,label=label,alpha=alpha,color=color)
    plt.axvline(x=np.mean(act_sum),color=color)
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
    
def relative_contrast(img, ref_img, mode):
    # returns the relative contrast of an original image and its reference.
    # relative contrast is the ratio of the image contrast with that of reference
    return  contrast(img,mode) / contrast(ref_img, mode)

def get_img_set_contrast(img_set, mode):
    # given a numPixel x num_imgs set of images, and a same dimension set of reference images,
    # determine the  relative contrast of that set averaged over all the images. 
    # relative contrast is the ratio between img and reference contrasts
    # return the average and standard error of the mean of the relative contrasts
    num_imgs = img_set.shape[1]
    rel_cons  = np.zeros((num_imgs,))
    for i in np.arange(num_imgs):
        rel_cons[i] = contrast(img_set[:,i],mode)
    return np.mean(rel_cons), np.std(rel_cons)/np.sqrt(num_imgs)

def stim_sweep_contrast_compar(img_data, ss_data, mode):
    # Plot the average relative contrast between 3 sets sweeping over all number of allowed timesteps
    Ts = ss_data.Ts
    
    mse_rel_cons  = np.zeros((Ts.size,))
    mse_stds     = np.zeros(mse_rel_cons.shape)
    wms_rel_cons  = np.zeros((Ts.size,))
    wms_stds     = np.zeros(mse_rel_cons.shape)
    ssm_rel_cons  = np.zeros((Ts.size,))
    ssm_stds     = np.zeros(mse_rel_cons.shape)

    for i in np.arange(Ts.size):
        mse_rel_cons[i], mse_stds[i] = get_img_set_contrast(ss_data.img_set_mse[i,:,:], img_data.img_set,mode)        
        wms_rel_cons[i], wms_stds[i] = get_img_set_contrast(ss_data.img_set_wms[i,:,:], img_data.img_set,mode)        
        ssm_rel_cons[i], ssm_stds[i] = get_img_set_contrast(ss_data.img_set_ssm[i,:,:], img_data.img_set,mode)        
 
    
    plt.errorbar(Ts,mse_rel_cons,yerr=(mse_stds),label='MSE Contrast')
    plt.errorbar(Ts,wms_rel_cons,yerr=(wms_stds),label='wMSE Contrast')
    plt.errorbar(Ts,ssm_rel_cons,yerr=(ssm_stds),label='SSIM Contrast')
    
    plt.xlabel('Number of Allowable Spikes')
    plt.ylabel('Average RMS Contrast')
    plt.xscale("log")
    plt.legend()

    

# greedy reconstruction (real time)

# displayed image is generated by a linear mapping of cellular activities
    # each cell contains activity for (int_time) miliseconds
    # activity is stored as a queue
    # at each time step, if a cell j is stimulated, the activity at that time step becomes 1 with probability p_j
    # contains a cell_num vector refractory_cells that specify which cells cannot fire because they are in refractory period 

# greedy dictionary selector 
    # at each time step, greedy receives an image, and has access to cell_activity and refractory_cells
    # greedy temporarily deletes refractory cells from its dictionary (remove column of a , row of 
    # greedy chooses the dictionary element to stimulate according to a given metric
    # greedy passes stimulation vector to cell object 
    
   
    
    







