# Import Statements

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.io
from scipy import ndimage
import cvxpy as cp
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import datetime
import pickle
import cv2
import time

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
        self.num_stixels = None
        self.dict_len = None

def metric_compar(img_data, params):
    # Compare Error Metrics Side-by-Side for the same set of images    
    img    = img_data.orig_img
    xs     = img_data.xs
    ys     = img_data.ys
    
    if params.electrode:
        print('Solving for Electrode Activities...')
    else:
        print('Solving for Cellular Activities...')    
    
    print('MSE Activity Reconsruction:')
    imgs_mse, acts_mse = recons_img_set(img_data, params, "mse")
    print('wMSE Activity Reconstruction')
    imgs_wms, acts_wms = recons_img_set(img_data, params, "wms")
    print('SSIM Activity Reconstruction')
    imgs_ssm, acts_ssm = recons_img_set(img_data, params, "ssm")
    
    print('Activities Solved. Rebuilding Images ...')
    
    recons_mse = rebuild_img(img,imgs_mse,xs,ys,params)
    recons_wms = rebuild_img(img,imgs_wms,xs,ys,params)
    recons_ssm = rebuild_img(img,imgs_ssm,xs,ys,params)

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
    
def recons_img_set(img_data, params, metric):
    # Given a set of images (img_set) as a 2d Matrix, and a metric, reconstruct
    # the image set according to the given image in parallel according to the available cpu cores    
    if params.electrode:
        activity_length = params.P.shape[1]
    else:
        activity_length = params.A.shape[1]
    
    num_imgs   = img_data.num_imgs
    # convert img_set to list for parallelization
    img_list = []
    for i in np.arange(num_imgs):
        img_list.append(img_data.img_set[:,i])
     
    # run reconstructions in parallel
    results = np.asarray(Parallel(n_jobs=-2)(delayed(act_solver)(i,img_data, params,metric) for i in tqdm(img_list)))
    #convert results back to 2 variables separating activity and the reconstructed image
    imgs = np.zeros((params.num_stixels,num_imgs))
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

    return img.T

def save_data(data,file_name):
    # save the data structure passed to the filename given
    session_title = file_name+"_"+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")+'.dat'
    with open(session_title, 'wb') as file_handle:
        pickle.dump(data, file_handle)
        
def load_data(file_name):
    return pickle.load(open(file_name,'rb'))

def load_raw_data(file_name, full_stixel_dims, position, width, height):
    # loads raw data from matlab .mat file
    
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
        # Given a dictionary and a threshold value, remove any dictionary elements whose maximum value is 
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
    
    def normalize_cell_stas(A,norm='l2'):
        # given a num_pixel x num_cell decoder matrix,
        # normalize each cell (each column) to unity
        # norm (either inf_norm or l2)
        
        # first record the original max value
        if norm == 'l2':
            max_val = np.max(np.linalg.norm(A,axis=0))
        elif norm == 'inf':
            max_val = np.max(np.abs(A))
        
        
        #go through each column
        for i in np.arange(A.shape[1]):
            if norm == 'l2':
                A[:,i] = A[:,i] / np.linalg.norm(A[:,i])
                
            elif norm == 'inf':
                A[:,i] = A[:,i] / np.max(np.abs(A[:,i]))
        
        return  max_val * A
            
    def trim_decoder(A, image_dims, coords, width, height):
        # Trim the (image_dims) image that A maps to to a width x height
        # rectangle with the top left situated at coords
        # square
        A_slice = np.zeros((width*height,A.shape[1]))
        
        col_size = image_dims[0]
     
        for col in np.arange(coords[1],coords[1]+height):
            A_slice[width*(col-coords[1]):width*(col-coords[1]+1),:] = A[coords[0]+col_size*col:coords[0]+width+col_size*col,:] 
    
        return A_slice
    
    data = scipy.io.loadmat(file_name)
    
    ## preprocess decoder matrix: trim to desired dimensions, prune 0 entries and normalize STAs.
    A   = (data['stas'].T)
    A = trim_decoder(A, full_stixel_dims, position, width, height)
    
    P,e_map = prune_dict(data['dictionary'].T,data['ea'])
    e_locs = data['elec_loc']  # electrode locations
    
    A,P = prune_decoder(A,P)
    dict_len = P.shape[1]
    num_stixels = A.shape[0]
    A = normalize_cell_stas(A,'inf')
    
    # save data in RetinaData Object
    ret_data = RetinaData()
    ret_data.A = A
    ret_data.P = P
    ret_data.e_locs = e_locs
    ret_data.e_map  = e_map
    ret_data.num_stixels = num_stixels
    ret_data.dict_len = dict_len
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

def flat_DCT(dims):
    # build and return a flattened dct matrix specifically for (dim0,dim1) images flattened with fortran ordering
    # Build 80 x 40 2D DCT-II Matrix
    num_pixels = dims[0]*dims[1]
    D1 = np.zeros((num_pixels,num_pixels))
    D2 = np.zeros((num_pixels,num_pixels))
    # build a flattened form of a  1d DCT matrix 
    l_dim = dims[0]
    r_dim = dims[1]
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
                D2[k+j*dims[0],i*dims[0]+k] = Dr[i,j]
    D = D2@D1
    return D

def flatW(params,img_data): 
    # build and return a flattned W matrix for images (img) flattned with fortran ordering
    # (1/2) * N / D where D is horizontal degrees, N is number of blocks 
    vis_ang_horz = params.vis_ang_horz
    N  =  int(img_data.orig_img.shape[0]/img_data.selec_dims[0]) # number of selection blocks (number of samples of DC terms of each subImage)
    offset = (1/2) * (N / vis_ang_horz)
    Wp = csf(params,offset=offset) #offset frequency b
    flatW = np.reshape(Wp,(params.stixel_dims[0]*params.stixel_dims[1],),order='F')
    W = np.diag(flatW)
    return W

def csf(params,offset=0):
    # given a peak sensitivity frequency pf, and a psychophysically determined pixels-per-degree of viusal field ppd,
    # and and image, return a mask that has the same shape as the image and applies a weighting to each pixel in the image
    # according to the contrast sensitivity function 
    def getNg(params):
        e = params.e
        Ng0 = params.Ng0
        eg = params.eg
        term1 = .85 / (1 + (e/.45)**2)
        term2 = .15 / (1 + (3/eg)**2)
        return Ng0*term1*term2
    
    def Mopt(f,params):
        #given a spatial frequency f and psychophysical parameters,
        # return the frequnecy filetered by the optical transfer function
        # of the retina
        sigma00 = .30           # Non-retinal optical linespread constant (arcmin)
        sigmaRet = 1 / np.sqrt(7.2*np.sqrt(3)*getNg(params))
        sigma_0 = np.sqrt(sigma00**2 + sigmaRet**2) # (arcmin) std deviation of linespread (function of eccentricity)
        Cab = .08    # (arcmin / mm ) dimensionality constant
        d = params.pupil_diam # pupil size in mm
        sigma = np.sqrt(sigma_0**2 + (Cab*d)**2)
        return np.exp(-2*(np.pi**2)*((sigma/60)**2)*(f**2))
        
    def intTerm(f,params):
        # given spatial frequency f and psychophysical paratmeters,
        # calculate the visual-angle integration term of the CSF
        e = params.e
        Xmax = 12   # (degrees) maximum visual integration area  
        term1 = .85 / (1 + (e/4)**2)
        term2 = .15 / (1 + (e/12)**2)
        Xmax=Xmax*(term1+term2)**-.5
        Ymax = Xmax
        Nmax = 15  # (cycles) maximum number of cycles function of eccentriicty
        vis_ang_horz = params.vis_ang_horz
        vis_ang_vert = params.vis_ang_vert
        
        term1 = (.5*vis_ang_horz)**2 + 4*e**2
        term2 = (.5*vis_ang_horz)**2 + e**2
        NmaxFac = term1/term2
        
        return 1/(vis_ang_horz*vis_ang_vert) + 1/(Xmax*Ymax) + NmaxFac*(f/Nmax)**2
    
    def illumTerm(params):
        #given spatial frequency f and psychophysical parameters,
        # calculate the  illumance term of the CSF
        n = .03  #quantum efficiency term (function of eccentricity)
        e = params.e
        term1 = .4 / (1 + (e/7)**2)
        term2 = .48 / (1 + (e/20)**2) 
        n = n*(term1 + term2 +.12)
        p = 1.24 # photon conversion factor (function of incident light)
        d = params.pupil_diam
        L = params.L
        E = np.pi/4 * d**2 * L * (1 - (d/9.7)**2 + (d/12.4)**4)
        return 1/(n*p*E)
        
    def inhibTerm(f,params):
        # given spatial frequency f and psychophysical parameters,
        # calculate the lateral inhibition term of the CSF
        Ng0 = params.Ng0
        e = params.e
        u0 = 7  #(cycles/deg) stop frequency of lateral inhibition
        term1 = .85 / (1 + (e/4)**2)
        term2 = .13 / (1 + (e/20)**2)
        u0 = u0 * (getNg(params)/Ng0)**.5 * (term1 + term2 + .02)**-.5
        return 1 - np.exp(-(f/u0)**2)
    
    k  = params.k
    X0 = params.elec_ang_horz
    T  = params.T
    sf_res = 1/params.stixel_dims[0]
    Ng = getNg(params)
    Ng0 = params.Ng0
    ph0= 3*10**-8*Ng0/Ng  # neural noise term (sec / deg^2)
    fxx,fyy = np.meshgrid(np.arange(params.stixel_dims[1]),np.arange(params.stixel_dims[0]))
    ppd = params.stixel_dims[0]/X0
    fs = (sf_res * ppd *((fxx)**2+(fyy)**2)**.5  ) + offset
    
    num   = Mopt(fs, params) / k
    
    if not params.binocular:
        num = num /  np.sqrt(2)
    
    denom = np.sqrt( 
        (2/T)
        *intTerm(fs,params)
        *(illumTerm(params) + ph0 / inhibTerm(fs,params)) 
    )
    W = np.divide(num,denom)
    return W

def mse(A,B):
    #flatten if not flat
    if A.ndim > 1:
        flatA = A.flatten()
        flatB = B.flatten()
        
        return (flatA-flatB).T@(flatA-flatB)/flatA.size
    else:
        return (A-B).T@(A-B)/A.size

def jpge(A,B,params, img_data):
    try:
        D = flat_DCT(params.stixel_dims)
        diffImg = A - B
        if diffImg.ndim is not 1: #flatten image if not already flattened
            diffImg = diffImg.flatten
        W = flatW(params, img_data)
        W = W/np.max(W)
        return np.linalg.norm(W@D@diffImg)**2 / A.size
    except:
        W = csf(params)
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

def get_elec_angs(stixel_dims, smps, stixel_size, eye_diam):
    # Given a set of psychophysical parameters,the reconstructing electrode array
    # smps: stimulus monitor pixel size: the size of a single monitor pixel in lab setup on the retina (microns)
    # stixelSize:  the stixel size,which is the square root of the number of monitor pixels grouped together 
    #      to form a single STA pixel (one STA pixel is stixelSize x stixelSize monitor pixels)
    # eyeDiam: the Emmetropia diameter of the eye in milimeters
        
    retArea  = ( # Retinal area in milimeters
        stixel_dims[0]*smps*stixel_size/1000,
        stixel_dims[1]*smps*stixel_size/1000
    )
    
    elecVisAng = ( # Visual Angle Spanned by the Electrode Reconstruction 
        np.rad2deg(np.arctan(retArea[0]/eye_diam)),
        np.rad2deg(np.arctan(retArea[1]/eye_diam))
    )
    
    return elecVisAng

def preprocess_img(img,params):
    # Given psychophysically determined viewing angles for the visual
    # scene, the image, and the dimensions of the stimulus reconstruction in 
    # pixels, tile the image into a set of subimages, where each subimage
    # covers precisely elecVisAng[0] x elecVisAng[1] degrees of the visual
    # scene. Resample these tiled images to have the same dimensions as the 
    # stimulus pixel (stixel_dims) for reconstruction.
    # elecVisAng[0]/objVisAngle[0] = selection/  img.shape[0]
    
    def tileImage(img,stixel_dims):
        # Given an mxn image and stixel_dims, tile the image by splitting it into 
        # num_imgs subimages obtained by taking pieces of size stixel_dims from the original image, stacking,
        # and then returning the images, as well as the x & y locations of the top left corner of each image
        
        def fitToDims(img,stixel_dims):
            # Given an mxn image, fit the image to the given dimension by padding it with zeros. 
            # This imamge assumes m<= pixelDIms[0] and/or n <= pxiel_dims[1]
            fitImg = np.zeros(stixel_dims)
            fitImg[0:img.shape[0],0:img.shape[1]] = img
            return fitImg       
        
        print('Tiling Image ...')
        x = 0
        y = 0 # initial location is top left of image
        subImgs = np.zeros((stixel_dims[0]*stixel_dims[1],0))
        xs = np.asarray([])
        ys = np.asarray([])

        while y <= img.shape[1]-stixel_dims[1]:
            # sweep horizontally. if x >= img.shape set x to 0 and update y
            if x >= img.shape[0]-stixel_dims[0]: 
                x = 0
                y += int(stixel_dims[0])

            selection = fitToDims(img[x:x+stixel_dims[0],y:y+stixel_dims[1]],stixel_dims)
            selection = np.reshape(selection,(stixel_dims[0]*stixel_dims[1],1),order='F')
            if not np.all(selection==0):
                subImgs = np.concatenate((subImgs,selection),1)
                xs = np.append(xs,[x])
                ys = np.append(ys,[y])
                x += int(stixel_dims[0])

        print('Tiled Image')        
        return subImgs, xs, ys

    
    selec_dims = get_selection_dims(params, img.shape)

    img_set, xs, ys        = tileImage(img,selec_dims)

    num_imgs = img_set.shape[1]
    img_set_res = np.zeros((params.stixel_dims[0]*params.stixel_dims[1],num_imgs))

    # go through each image, resample it and store it in img_set_res
    for i in np.arange(num_imgs):
        img_set_res[:,i] = resample(img_set[:,i],selec_dims,params.stixel_dims)
    img_data = ImageData()
    img_data.num_imgs = num_imgs
    img_data.img_set = img_set_res
    img_data.xs = xs
    img_data.ys = ys
    img_data.orig_img    = img
    img_data.selec_dims      = selec_dims
    img_data.resampled_img = rebuild_img(img, img_set_res, xs, ys, params)
    
    return img_data

def get_selection_dims(params,img_dims):
    vis_ang_horz = params.vis_ang_horz
    
    elec_ang_horz = params.elec_ang_horz
    elec_ang_vert = params.elec_ang_vert
    
    selection_size_x = int(np.ceil(elec_ang_horz/vis_ang_horz * img_dims[0]))
    selection_size_y = int(np.ceil(elec_ang_vert/vis_ang_horz * img_dims[0]))

    # select the equivalent of elecVisangx elecVisAng pixels from the image
    selec_dims = (selection_size_x,selection_size_y)
    return selec_dims

def act_solver(img,img_data,params,mode):
    # Reconstruct an image according to the error metric specified by "mode"
    # Input: img : the image to be reconstructed, dims = psych_params["stixel_dims"]
    #        sim_params : a simulation parameters dictionary 
    #        psych_params: a psychophysical parameters dictionary
    #        mode : a string specifying the particular error metric being used
    #        electrode : a boolean specifying whether to reconstruct according ot optimal cell 
                # activities or using th electrode stimulation dictionary 
    #Subfunctions:
    def var_term(params,phi, x):
    # Return the cost function associate with the variance component of the reconstruction
    # error. Only used in the case that electrode is true
    # Inputs: 
    #     sim_params: the simulatin parameters dictionary object
    #     electrode: boolean indicating whether performing optimal cellular or electrode dictionary recons
    #     x : the cvx variable representing the activity vector object that is being solved for
        var_mtx = variance_matrix(params, phi)
        return  cp.sum(var_mtx@x)
    def variance_matrix(params, phi):
        # get the variance matrix that maps dictionary selections to 
        # total variance of the reconstructed image according to the given
        # metric, phi
        P = params.P
        V = np.zeros(P.shape)
        for j in np.arange(P.shape[1]):
            V[:,j] = np.multiply(P[:,j],(1-P[:,j]))
        return np.multiply(phi,phi)@V
    
    def recons_ssm(img, params, epsilon = 10**-2):
        # use bisection search to solve for an optimal-SSIM reconstruction

        def find_feasible(y,alpha, params ):
            # Return a feasible solution to the SSIM optimization problem
            # Using cvxpy solves the constrained feasability problem that is a transformation of the SSIM
            # optimization problem.

            def cvxineq(a,y,x,phi):
                # a convex inequality to evaluate feasability
                return (1-a)*cp.sum_squares(y-phi@x)-2*a*(phi@x).T@y

            A = params.A
            P = params.P

            if params.electrode:
                x = cp.Variable(P.shape[1])
                phi = A@P
                cost = var_term(params ,phi, x)
            else:
                x = cp.Variable(A.shape[1])
                cost = 1
                phi = A

            num_stims = params.num_stims
            N = params.max_act
            if num_stims == -1:
                constraints = [x <= N, x >= 0, cvxineq(alpha,y,x,phi) <= 0]
            else:
                constraints = [x <= N, x >= 0, cvxineq(alpha,y,x,phi) <= 0, cp.sum(x) <= num_stims]

            prob= cp.Problem(cp.Minimize(cost),constraints)
            try:
                prob.solve(solver=cp.GUROBI)
            except: 
                prob.solve(solver=cp.SCS)

            if x.value is not None:
                return True, x.value
            else:
                return False, x.value

        
        A = params.A
        P = params.P
        if params.electrode:
            act_length = P.shape[1]
        else:
            act_length = A.shape[1]


        # image preprocessing 
        y = img


        # bisection initialization
        l = 0 # lower bound
        u = 2 # upper bound
        e = epsilon  # accuracy
        x = np.zeros(act_length) # solution
        xCurr = np.zeros(act_length) # temporary solution

        # bisection search
        while u - l >= e:
            alpha = (l+u)/2
            # find feasible x   let u = alpha
            isFeasible, xCurr = find_feasible(y, alpha, params)

            if isFeasible:
                u = alpha
            elif alpha == 1:
                print('SSIM reconstruction cannot be solved.')
                if params.electrode:
                    return 0*A@P@x, 0*x
                else:
                    return 0*A@x, 0*x
            else:
                l = alpha

            if xCurr is not None: # only overwrite x is new value is generated
                x = copy.deepcopy(xCurr)            
        x = np.rint(x)
        if params.electrode:
            return A@P@x, x
        else:
            return A@x, x
    
    A = params.A
    P = params.P
    T = params.num_stims
    N = params.max_act

    y = img

    if params.electrode:
        x = cp.Variable(P.shape[1])
    else:
        x = cp.Variable(A.shape[1])

    if mode == "mse": 
        if params.electrode:
            cost = cp.sum_squares(y-A@P@x) + var_term(params,A,x)
        else:
            cost = cp.sum_squares(y-A@x)
    
    elif mode == "wms":
        W = flatW(params,img_data)
        D = flat_DCT(params.stixel_dims)
        if params.electrode:
            cost = cp.sum_squares(W@D@(y-A@P@x)) + var_term(params, W@D@A, x)
        else:
            cost = cp.sum_squares(W@D@(y-A@x))
            
    elif mode == "ssm": 
        # custom SSIM bisection search solver
        return recons_ssm(img, params)
        
    # Solve cost function and return x's value and the reconstructed image
    if T == -1:
        prob= cp.Problem(cp.Minimize(cost),[x<=N,x >= 0])
    else:
        prob = cp.Problem(cp.Minimize(cost),[x<=N, x >= 0, cp.sum(x) <= T])
        
    try:
        prob.solve(solver=cp.GUROBI)
    except:
        prob.solve(solver=cp.SCS)
    
    if params.electrode:
        return A@P@x.value, x.value
    else:
        return A@x.value, x.value

def resample(img,curr_dims, desiredDims, return_flat = True):
    
    # given a (curr_dims[0]*curr_dims[1] x 1 ) image vector, resample the image
    # to fit to desired dims and return this image flatted into a 
    #(desiredDims[0],desiredDims[1] x 1) image vector
    
    # if image flat, reshape
    if img.ndim == 1:
        img = np.reshape(img,curr_dims,order='F')
    
    # desiredDims[0] = zoomFac * curr_dims[0]
    zoomFac =  desiredDims[0]/curr_dims[0]
    zImg = ndimage.zoom(img,zoomFac)
    if return_flat:
        return np.reshape(zImg,(desiredDims[0]*desiredDims[1],),order='F')
    else:
        return zImg

def rebuild_img(img,img_set,xs,ys,params,zeroMean=False): 
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
    selec_dims = get_selection_dims(params,img.shape)
    # iterate through each (x,y) coordinate a
    for i in np.arange(xs.shape[0]): 
        # if dims not correct, resample to selectiondims
        if (params.stixel_dims[0] != selec_dims[0] or params.stixel_dims[1] != selec_dims[1]):
            # resample image
            resampled_img = resample(img_set[:,i],params.stixel_dims,selec_dims)
            reconsImg = np.reshape(resampled_img,selec_dims,order='F')
        
        else:
            reconsImg = np.reshape(img_set[:,i],selec_dims,order='F')
        # only add to image if exactly zero pixel
        x = xs[i]
        y = ys[i]
        
        
        selection = recons[x:x+selec_dims[0],y:y+selec_dims[1]]
        reconsSel = reconsImg[0:selection.shape[0],0:selection.shape[1]]
        
        # If zeromean, make the subimage zero mean, then add the average intensity for the whole image
        if zeroMean:
            reconsSel -= np.mean(reconsSel) 
        recons[x:x+selection.shape[0],y:y+selection.shape[1]] += reconsSel
    
    return recons.T  

    
### Device Programming

class ActivityQueue:
    # An activity queue is an int_time length queue data
    # structure that contains a rolling window of spike
    # times for a given cell. 
    
    def __init__(self, int_time):
        self.__int_time = int_time # integration time in miliseconds
        self.__act_queue = np.zeros((int_time,))
        self.__firing_rate = 0     
            
    def get_firing_rate(self):
        # return the firing rate for the cell defined as the sum of spiked in the window divided by the integration time
        return self.__firing_rate / (self.__int_time / 1000)       
            
    def update(self, has_spiked):
        #  dequeue oldest time bin (fifo order),
        # if cell spiked, has_spiked=true so enqueue a 1,
        # otherwise encode a 0
        discard = self.__act_queue[0]
        if has_spiked:
            spike = 1
        else:
            spike = 0
        
        new_act = np.empty_like(self.__act_queue)
        new_act[0:self.__int_time-1] = self.__act_queue[1:]
        new_act[-1] = spike
        
        self.__act_queue = new_act

        self.__firing_rate += spike
        self.__firing_rate -= discard
        
    def __str__(self):
        return str(self.__act_queue)

    def get_spike_raster(self):
        # return the vector 
        return self.__act_queue
        
class RetinalCell:
    STIMULATION_RATE = 1000 # stimulation rate in Hz (time between update calls for cells)
    __INTEGRATION_TIME = 100  # integration time of the cell in miliseconds
    __NUM_TIME_BINS    = int(__INTEGRATION_TIME * STIMULATION_RATE)
    
    __REFRACTORY_PERIOD = 1  # Refractory Period of Cells in miliseconds
    __REFRACTORY_IDX = int(__REFRACTORY_PERIOD * STIMULATION_RATE)   # How many indices from the end correspond to __REFRACTORY_PERIOD

    
    
    # each cell contains an activity
    # it is either in or not in the refractory period
    def __init__(self, cell_num, cell_type):
        self.cell_type = cell_type
        self.cell_num = cell_num
        self.in_refractory = False
        
        self.__act_queue = ActivityQueue(self.__NUM_TIME_BINS)
        
    def get_firing_rate(self):
        # return the firing rate of the cell (sum of spikes / integration time)
        return self.__act_queue.get_firing_rate()
    
    def update(self,has_spiked):
        #update the cells activity depending whether or not the cell is meant to spike
        # if cell is in refractory period, ignore spikes
        # assume refractory period is 1ms (1 bin), so if cell in refractory period, change cell status    
        
        # first update whether or not the cell is in its refractory period        
     
        #ignore spike if in refractory period    
        if has_spiked and not self.in_refractory:
            self.__act_queue.update(1)                
            
        elif not has_spiked and self.in_refractory:
            self.__act_queue.update(0)
            
        elif not has_spiked and not self.in_refractory:
            self.__act_queue.update(0)
                
        self.__update_refractory_period()

    def get_spike_raster(self):
        # return a vector of time bins representing cell activity 
        return self.__act_queue.get_spike_raster()

    def disp_info(self):
        # self-explanatory: display cell attributes to console
        print('Cell Number: %i'%self.cell_num)
        print('Cell Type: '+self.cell_type)
        print('Firing Rate: %.2f Hz'%self.get_firing_rate())
        print('Spikes: ',self.get_spike_raster())
        print('In Refractory: %s'%self.in_refractory)
        
    def __update_refractory_period(self):
        # look at the previous activity over the cells integration window and determine if the cell 
        # is in its refractory period
        spike_raster = self.get_spike_raster()
        if np.any(spike_raster[-self.__REFRACTORY_IDX:]):
            self.in_refractory = True
        else:
            self.in_refractory = False
        
class Retina:
    # Init (A, P, stixel_dims)
    def __init__(self, params):
        
        
        self.__A = params.A # linear decoder matrix (from activity to image)
        self.__P = params.P # dictionary matrix (probabilities of activation for each dictionary electrode stim pattern)
        self.__num_cells = self.__A.shape[1]   # number of RGCs in the retina
        self.__dict_length = self.__P.shape[1] # Number of dictionary elements
        self.__num_stixels = params.stixel_dims[0]*params.stixel_dims[1]  # number of Stimulus pixels in reconstructed image
        self.__cells = np.zeros((self.__num_cells),dtype=np.object) # array to hold cell objects
        
        self.activities = np.zeros((self.__num_cells,))  # array to hold cellular activities for quick access
        self.stixel_dims = params.stixel_dims # pixel dimensions of the Stimulus reconstruction 
        
        # add cells 
        for i in np.arange(self.__num_cells):
            self.__cells[i] = RetinalCell(i,'parasol')
      
    def __cell_update(self, cell_num, stim_vector):
        # stimulate cell by calling update on the cell, where
        # the argument of update is 1 with probability stim_prob
        
        has_spiked = np.random.binomial(1,stim_vector[cell_num])
        self.__cells[cell_num].update(has_spiked)
        self.activities[cell_num] = self.__cells[cell_num].get_firing_rate()
        
    def __get_stim_vector(self, dict_sel):  
        # get the stimulation vector corresponding to the dictionary selection
        return self.__P[:,dict_sel]
          
    def get_cell_rasters(self):
        # get the rasters of each cell in the retina and return them as a matrix
        rasters = np.zeros((self.__num_cells, RetinalCell.INTEGRATION_TIME))
        for i in np.arange(self.__num_cells):
            rasters[i,:] = self.__cells[i].get_spike_raster()
        return rasters    

    def get_image_vector(self):
        # return the image encoded in the retina by a 
        return self.__A@self.activities
    
    def display_image(self):
        # display the image encoded in the retina
        
        if plt.fignum_exists(0):
            fig = plt.figure(0)
            img = self.get_image_vector()
            plt.imshow(np.reshape(img,self.__stixel_dims,order='F'),cmap='bone')
            plt.pause(.00001)
            fig.canvas.draw()
        else:
            plt.figure(0)
            img = self.get_image_vector()
            plt.imshow(np.reshape(img,self.__stixel_dims,order='F'),
                       cmap='bone',
                       vmax=.5,
                       vmin=-.5)
            
            plt.axis('off')
            plt.ion()
            plt.show()
        
    def stimulate(self, dict_sel):
        # given a dictionary electrode, simulate electrode stimulation and update each cell
        
        # get dictionary element
        stim_vector = self.__get_stim_vector(dict_sel)
        
        # TODO: Fix bug so that we can update all cells in parallel
        #Parallel(n_jobs=4)(delayed(self.__cell_update)(cell_num, stim_vector[cell_num]) for cell_num in np.arange(self.__num_cells))
        
        for i in np.arange(self.__num_cells):
            self.__cell_update(i,stim_vector)

class EyeTracker:

    def __init__(self, simulate_eye_position=True):
        
        if simulate_eye_position:
            self.__count = 0
            self.__gaze_x = 300
            self.__gaze_y = 300
            self.simulate_eye_pos = True
            self.__sim_jitter = 5
            self.__saccade_rate = 3 # average fixation time in seconds
            self.__next_saccade_time = np.random.poisson(self.__saccade_rate)
            self.__start_time = time.time()
            self.__curr_time  = time.time()
    def __get_eye_pos(self):
        pass
    
    def __get_simulated_eye_pos(self):
#         self.__curr_time = time.time()
#         if self.__curr_time-self.__start_time >= self.__next_saccade_time:
#             self.__next_saccade_time = np.random.poisson(self.__saccade_rate)
#             print('saccade time is: ',self.__next_saccade_time)
#             self.__start_time = self.__curr_time
#             self.__gaze_x = np.random.randint(1280) 
#             self.__gaze_y = np.random.randint(720) + 200
        self.__gaze_x += int(np.random.randn()*self.__sim_jitter)
        self.__gaze_y += int(np.random.randn()*self.__sim_jitter)    
 
        return (self.__gaze_x, self.__gaze_y)
        
    def get_gaze_center(self):
        if self.simulate_eye_pos:
            return self.__get_simulated_eye_pos()
        else:
            return self.__get_eye_pos()
            
# class ElectrodeArray:
#     # electrode locations
#     # dictionary element currents
#     # stimulate Retina(dictionary)
#     # get current current
#     pass

class Parameters:
    # parameters class is a data structure holding simulation parameters for easy access by other classes
    def __init__(self, 
                 A, P, 
                 vis_ang_horz, vis_ang_vert,
                 stim_monitor_pixel_size, eye_diam, stixel_size, 
                 stixel_dims, elec_ang_horz, elec_ang_vert,
                 use_electrode,
                 num_stims, max_activity,
                 eccentricity, k, Ng0, eg, L, is_binocular  ):
        ## retinal data
        self.A = A
        self.P = P
        self.vis_ang_horz = vis_ang_horz
        self.vis_ang_vert = vis_ang_vert
        self.smps = stim_monitor_pixel_size
        self.elec_ang_horz = elec_ang_horz
        self.elec_ang_vert = elec_ang_vert
        ## simulation parameters
        self.eye_diam = eye_diam
        self.stixel_size = stixel_size
        self.stixel_dims = stixel_dims
        self.electrode = use_electrode
        self.num_stims = num_stims
        self.max_act = max_activity
        ## psychophysical parameters for CSF 
        self.e = eccentricity
        self.k = k
        self.Ng0 = Ng0
        self.eg = eg
        self.L = L
        self.binocular = is_binocular

    @classmethod
    def empty(cls):
        return Parameters(None,None,None,None,None,None,None,
                          None,None,None,None,None,None,None,
                          None,None,None,None,None)

class GreedyChooser:
    pass
    
class Camera:
    # receives frames in time
    
    def __init__(self, device_id):
        self.__cap = cv2.VideoCapture(device_id,cv2.CAP_DSHOW)
        
        ## for now force resolution to 1080p
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920);
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080);
        
        cam_width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height =int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_resolution = np.asarray([cam_width, cam_height])
    def get_frame(self):
            # Capture frame-by-frame
            ret_val, frame = self.__cap.read()
            if ret_val:
                return np.asarray(frame)
            else:
                raise "Bad Frame Grab From Camera"    
  
    def __del__(self):   

        # When everything done, release the capture
        self.__cap.release()
        cv2.destroyAllWindows()
 
class ImageProcessor:
    # an image processor integrates camera input and eye tracking info
    # to compute the desired image that will be reconstructed in the device
            
    def __init__(self, camera_id, params ):
        self.__cam = Camera(camera_id)
        self.__eye_tracker = EyeTracker()
        
        self.__electrode_angle_horz = params.elec_ang_horz
        self.__electrode_angle_vert = params.elec_ang_vert
        
        
        self.__visual_angle_horz = params.vis_ang_horz  
        self.__visual_angle_vert = params.vis_ang_vert
        
        
        self.__selection_dims = get_selection_dims(params, self.__cam.camera_resolution)
        
        self.__stixel_dims = params.stixel_dims
        
    def preprocess_image(self,img):
        # zero mean and normalize scene to -.5,.5 grayscale,  then use eye position to window
        # image to a desired subimage to be reconstructed, return the preprocessed image
        
        ## grayscale rgb image
        #img = np.sum(img,2)/3
        
        ## normalize image to zero-mean [-.5,.5] pixel values
        #img -= np.mean(img)
        #img = img/(2*np.max(img))
        
        ## window image based on gaze position
        gaze_pos_x, gaze_pos_y = self.__eye_tracker.get_gaze_center()
        
        ## gaze position is center of window, clip edges if window extends bevis_ang_vertnd visual scene (img)
        dx = int(np.ceil(self.__selection_dims[0]/2))
        dy = int(np.ceil(self.__selection_dims[1]/2))
        mask = np.zeros(img.shape)
        mask[gaze_pos_y-dy:gaze_pos_y+dy,gaze_pos_x-dx:gaze_pos_x+dx,:] = 1
        img[np.logical_not(mask)] = 0
        
        
        # resample image to have same dimensions as stixels for reconstruction
        #img = resample(img, self.__selection_dims, self.__stixel_dims, return_flat=False )
        return img

    def get_stim_frame(self):
        return self.preprocess_image(self.__cam.get_frame())

class Device:
    # a device is a simulation of a complete retinal prosthetic, encompassing the following:
    # a camera to caputure the visual scene at some fixed resolution
    # an eye imager to capture eye position used to stabilizinge the image on the retina
    # a greedy processor that receives images and greedily stimulates the retina to reconstruct a given image
    # a retina that contains cells that are interfaced with the electrode array
    
    ## time is measured in stimulations, set by RetinalCell.STIMULATION_RATE
    STIMULATION_RATE = RetinalCell.STIMULATION_RATE
    
    def __init__(self, params ):
        self.__retina = Retina()
        self.__imager = EyeTracker()
        self.__camera = Camera()
    

# greedy reconstruction (real time)


# greedy dictionary selector 
    # at each time step, greedy receives an image, and has access to cell_activity and refractory_cells
    # greedy temporarily deletes refractory cells from its dictionary (remove column of a , row of 
    # greedy chooses the dictionary element to stimulate according to a given metric
    # greedy passes stimulation vector to cell object 
    


## at each time step
## get eye position
    ## if saccade
        ## get visual scene from camera
        ## compute & update target image
        ## reset retina
    ## choose element greedily according to metric
    ## stimulate electrode array for retina
        
   

## compute optimal image

## given preceding image
    

# 
# ret_data = load_raw_data('dict.mat')
# 
# params = Parameters(
#             ret_data.A, ret_data.P, 
#             vis_ang_horz=120, vis_ang_vert=100,
#             stim_monitor_pixel_size=5.5, eye_diam=17, stixel_size=8, 
#             stixel_dims=(100,100),
#             use_electrode = False,
#          )
# 
# img_processor = ImageProcessor(1,params)
# cv2.namedWindow('frame',cv2.WINDOW_FULLSCREEN)
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))
# 
# while True:
#     frame = img_processor.get_stim_frame()
#     if frame.shape[0] is not 0 and frame.shape[1] is not 0:
#         out.write(frame)
#         cv2.imshow('frame',frame)
#         cv2.resizeWindow('frame', 1920,1080)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
#   


### Workspace


# def get_band_rms(image):
#     # split an image into separate bandwidths (cycles per degree of visual field) 
#     # and measure the RMS energy of each band
#         pass
#     
#     def get_band_power(image, start_freq, stop_freq, params):
#         # bandpass filter a given image between start_freq and stop_freq (assuming ideal filter)
#         # return the RMS value of the remaining filter coefficients (energy of the band)
#          pass 


