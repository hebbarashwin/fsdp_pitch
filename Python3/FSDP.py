import numpy as np
import scipy.io as io
import scipy.io.wavfile as wav
import scipy.signal as sig
import os.path
import pickle
import warnings

def classify(modelFile, tester):
    
    with open(modelFile, 'rb') as fid:clf=pickle.load(fid, encoding = 'latin1')
    zz = clf.predict(list(zip(*tester)))
    return(zz)

def VuV_Compute(score):
        """Classifies into voiced and unvoiced frames. It uses an SVM based classifier."""
        nCand = 3
        srcCorpus = 'KEELE'
        ext = 'New'
        path = 'E:/IIScInternshipWork/pythonCodes/'
        score_sort = np.sort(score,axis=1)
        score_sort=score_sort[:,::-1]
        estVuV = np.zeros(len(score_sort), dtype = np.int8)
        indsRemain = list(range(0,len(score_sort)))
        inds = list()
        #print('score', score_sort.shape)
        for i in np.arange(1,nCand+1):
            currCand = i
            #print(currCand, len(indsRemain))
            if currCand < nCand:
                    tempInds= np.arange(currCand,nCand)
                    inds1 = list(np.where(np.sum(score_sort[:,tempInds],axis=1)== -1*len(tempInds))[0])
                    #print('inds1', len(inds1),len(inds))
                    if len(inds)==0:
                        inds = inds1
                    else:
                        tem = inds.extend(inds1)
                    #print('inds', len(inds))
            else:
                    inds = indsRemain
                        
            #print('cand :', currCand)
            #feature= score(1:nCand,inds);
            feature = score[inds,0:nCand]
            io.savemat(path+'testFeature.mat',{'feature':feature})
            #feature = [score_sort[i1,0:nCand] for i1 in inds]
            #print(len(inds),feature.shape)
            file = path+'KEELENew_model'+str(i)+'.pk1'

            if os.path.isfile(file):
                    
                    currEstVuV = classify(file, np.transpose(feature))
                    estVuV[inds] = currEstVuV
                    #print('a',len(indsRemain), len(inds))
                    indsRemain = [x for x in indsRemain if x not in inds]
                    #print('b',len(indsRemain))
                    inds = []
        return(np.transpose(estVuV))

def pitchStrengthAllCandidates(f,L,pc,kernel,J0):
    """Computes pitch strength matrix"""
    
    S = np.zeros((len(pc),len(L[0])))
    NL = np.divide(L,np.matlib.repmat(np.sqrt(np.sum(np.multiply(L,L),0)),int(len(L)),1))
    #print(len(NL),len(NL[0]))
    for j in np.arange(0,len(pc)):
        S[j,:]=np.matmul(np.transpose(kernel[0,j+J0]),NL) 
    return(S)
    
def hz2erbs(hz):
    erbs = 6.44*(np.log2(229+hz)-7.84)
    return(erbs)

def erbs2hz(erbs):
    #twos = np.ones(len(erbs))*2
    hz = np.power(2,(erbs/6.44 + 7.84)) - 229
    return(hz)

def swipep_mod(x,fs,plim,dt,dlog2p,dERBs,woverlap,sTHR, kernel_cell):
    dlog2p = 1/48
    t = np.arange(0,(len(x)/fs),dt)
    log2pc = np.arange(np.log2(plim[0]),np.log2(plim[1]),dlog2p)
    pc = np.power(2,log2pc)
    S = np.zeros((len(pc),len(t)))
    logWs = np.round(np.log2(8*np.divide(fs,plim)))
    temp_log = np.arange(logWs[0],logWs[1]-1,-1)
    ws = np.power((2*np.ones(len(temp_log))),temp_log)
    #print(ws)
    myN = 8
    pO = myN * np.divide(fs,ws) #Optimal pitches
    d = 1+log2pc - np.log2(myN*np.divide(fs,ws[0])) #Window sizes by each candidate
    fERBs = erbs2hz(np.arange(hz2erbs(min(pc)/4),hz2erbs(fs/2),dERBs))
    for i in np.arange(0,len(ws)):
        dn = max(1,round(myN*(1-woverlap)*fs/pO[i]))
        xzp = np.append(np.zeros(int(ws[i]/2)),x)
        xzp = np.append(xzp,np.zeros(int(dn+ws[i]/2)))
        w = sig.hanning(int(ws[i]))
        o = max(0,round(ws[i]-dn)) #window overlap
        f,ti,X = sig.spectrogram(xzp,fs,window=w,nperseg=len(w),noverlap=o,nfft = int(ws[i]),mode='complex')
        ti = ti - ti[0]
        
        if len(ws)==1:
            j = pc
            k = np.array([])
            k=k[0]
        elif i == len(ws)-1:
            j = np.where(d-i>0)
            j=j[0]
            k = np.where(d[j]-i<1)
            k=k[0]
        elif i==0:
            j = np.where(abs(d-i-1)<1)
            j=j[0]
            k = np.where(d[j]-i>1)
            k=k[0]
        else:
            j = np.where(abs(d-i-1)<1)
            j=j[0]
            k = np.arange(0,len(j))
        
        from scipy.interpolate import CubicSpline
        from scipy.interpolate import interp1d
        temp = np.where(fERBs > pc[j[0]]/4)
        fERBs = fERBs[temp[0]]
        #print('fERBs',fERBs)
        
        f1 = interp1d(f,abs(X),kind='cubic',axis=0)
        int1=f1(fERBs)
        L = np.sqrt(np.maximum(np.zeros((len(int1),len(int1[0]))),int1))
        #print(len(L),len(L[0]))
        Si = pitchStrengthAllCandidates(fERBs,L,pc[j],kernel_cell[0,i],j[0])
        row, col = Si.shape
        if col>1:
            f = interp1d(ti,np.matrix.transpose(Si),kind='linear',axis=0)
            Si = f(t)
        else:
            Si[:] = np.nan 
        lamda = d[j[k]]-i-1
        mu = np.ones((j.shape))
        mu[k]=1-abs(lamda)
        mu=np.transpose(mu)
        S[j,:]= S[j,:]+ np.transpose(np.multiply(np.matlib.repmat(mu,len(Si),1),Si))
    return(S,log2pc)

def pitchCandCost(signal,Fs,f0_candidates, kernel_cell):
    """This function estimates the confidence score for each of the pitch candidates in f0_candidates, by
        using the SWIPE algorithm."""
    
    allS,logP= swipep_mod(signal,Fs,[75,500],0.01,[],1/20,0.5,0.2,kernel_cell)
    p = np.power(2,logP)
    allS = allS[:,:-1] #check this
    rows = len(f0_candidates)
    cols = len(f0_candidates[0])
    s_swipe_new= np.zeros((rows,cols))
    
    for i in np.arange(0,rows):
        for j in np.arange(0,cols):
            if (f0_candidates[i,j] != 0):
                ind = np.argmin(abs(p-f0_candidates[i,j]))
                s_swipe_new[i,j]=allS[ind,i]
            else:
                s_swipe_new[i,j]=-1
    return(s_swipe_new)

def twomax(x,lowerbound,upperbound,unitlen,nCand,ii):
    """This function only finds two maximum peaks"""
    
    lenx=len(x)
    lowerbound=int(lowerbound)
    upperbound=int(upperbound)
    nCand=int(nCand)
    mag = np.amax(x[lowerbound:upperbound+1])
    index=np.argmax(x[lowerbound:upperbound+1])
    index=index+lowerbound
    if mag<=0:
        return(mag,index)
    startInd = index
    if nCand%2 ==0:
        temp=np.arange(-nCand/2 -1,nCand/2 +1)
    else:
        temp=np.arange(-(int(nCand/2)),int(nCand/2)+1)
    twos=2*np.ones(len(temp))
    allharmonics = np.power(twos,temp)
    LIMIT=0.0625
    abvRange=np.where(startInd+np.round(np.log2(allharmonics-LIMIT)/unitlen) > min(lenx,upperbound))
    allharmonics = np.delete(allharmonics,abvRange,0)
    if len(abvRange[0])!=0:
        allharmonics = np.append(np.power(2,np.arange(np.log2(min(allharmonics))-1,len(abvRange)+1)),allharmonics)
        allharmonics = np.delete(allharmonics,np.where(startInd+np.round(np.log2(allharmonics+LIMIT)/unitlen) < max(1,lowerbound)),0)
    blwRange = np.where(startInd + np.round(np.log2(allharmonics+LIMIT)/unitlen)<max(1,lowerbound))
    allharmonics = np.delete(allharmonics,blwRange,0)

    if len(blwRange[0])!=0:
        allharmonics = np.append(np.power(2,np.arange(np.log2(max(allharmonics))+1,len(abvRange)+1)),allharmonics)
        allharmonics = np.delete(allharmonics,np.where(startInd+np.round(np.log2(allharmonics+LIMIT)/unitlen) > min(lenx,upperbound)),0)
        
    allharmonics=np.delete(allharmonics,np.where(allharmonics==1))
    for i in np.arange(0,len(allharmonics)):
        harmonics=allharmonics[i]
        startpos = startInd + np.round(np.log2(harmonics-LIMIT)/unitlen)
        if(startpos <= min(lenx,upperbound)):
            if np.isscalar(index):
                index2=index
            else:
                index2=index[0]
            endpos=index2+np.round(np.log2(harmonics+LIMIT)/unitlen)
            endpos1=index+np.round(np.log2(harmonics+LIMIT)/unitlen)
            if (np.all(endpos1> min(lenx, upperbound)-1)):
                endpos = min(lenx,upperbound)-1
            if (startpos<0):
                startpos =0 
            mag1=np.amax(x[int(startpos):int(endpos)+1])
            index1=np.argmax(x[int(startpos):int(endpos)+1])
            index1=index1+startpos
            mag=np.append(mag,mag1)
            index=np.append(index,index1)
    return(mag,index)

def GetLogSpectrum(segment,fftlen,limit,logf,interp_logf):
    """Computes Log Spectrum"""
    Spectra = np.fft.fft(segment,fftlen)
    amplitude = abs(Spectra[:int(fftlen/2)+1])
    amplitude = amplitude[1:limit+2]
    interp_amplitude = np.interp(interp_logf,logf,amplitude)
    interp_amplitude = interp_amplitude - min(interp_amplitude)
    return(interp_amplitude)

def ComputeSHR(log_spectrum,min_bin,startpos,endpos,lowerbound,upperbound,N,shift_units,nCand,ii):
    """Computes subharmonic to harmonic ratio for a short time signal"""
    len_spectrum=len(log_spectrum)
    totallen=shift_units+len_spectrum
    N=int(N)
    lowerbound=int(lowerbound)
    upperbound=int(upperbound)
    shift_units=int(shift_units)
    totallen=int(totallen)
    
    shshift=np.zeros((N,totallen))
    shshift[0,(totallen-len_spectrum):totallen]=log_spectrum
    for i in np.arange(1,N):
        shshift[i,int(startpos[i-1]):int(endpos[i-1])+1]=log_spectrum[:int(endpos[i-1]-startpos[i-1])+1]
        
    shshift=shshift[:,shift_units:totallen+1]
    shsodd = np.sum(shshift[0:N:2,:],axis=0)
    shseven= np.sum(shshift[1:N:2,:],axis=0)
    difference = shseven-shsodd
    mag, index = twomax(difference,lowerbound,upperbound,min_bin,nCand,ii)
    if np.amin(mag)<0:
        peak_index=-1
    else:
        peak_index=1
    return(peak_index,index)

def mySHR_Frame_4f0_new(sigFrame,Fs,nWndw,f0MinMax,nCand,ii):
    minf0= f0MinMax[0]
    maxf0= f0MinMax[1]
    ceiling=1250 # max frequency for analysis.
    interpolation_depth=0.5
    maxlogf=np.log2(maxf0/2)
    minlogf=np.log2(minf0/2) # the search region to compute SHR is as low as 0.5 minf0.
    
    N=(ceiling/minf0)//1 #maximum number harmonics
    m=N%2
    N=N-m
    N=N*4
    
    fftlen=1;
    while (fftlen < nWndw * (1 +interpolation_depth)):
        fftlen = fftlen* 2
    frequency = (Fs/fftlen)*np.arange(1,fftlen/2 +1)
    limit1 = np.where(frequency>=ceiling)
    limit = limit1[0][0]
    frequency = frequency[0:int(limit)+1]
    logf = np.log2(frequency)
    min_bin = logf[-1]-logf[-2]
    shift=np.log2(N)
    shift_units=round(shift/min_bin)
    i=np.arange(2,N+1)
    startpos=np.zeros(len(i))
    for m in np.arange(0,len(i)):
        k = np.log2(i[m])/min_bin
        startpos[m]=shift_units-round(k)
    interp_logf = np.arange(logf[0],logf[-1],min_bin)
    interp_len=len(interp_logf)
    totallen=interp_len+shift_units
    endpos=np.zeros(len(i))
    endpos=startpos+interp_len-1
    twos=2*np.ones(len(interp_logf))
    newfre = np.power(twos,interp_logf)
    upperbound1 = np.where(interp_logf>=maxlogf)
    upperbound = upperbound1[0][0]
    lowerbound1 = np.where(interp_logf>=minlogf)
    lowerbound = lowerbound1[0][0]
    
    ff0_candidates=np.zeros(nCand)
    log_spectrum=np.zeros(len(interp_logf))
    log_spectrum = GetLogSpectrum(sigFrame,fftlen,limit,logf,interp_logf)
    #log spectrum is correct
    peak_index,all_peak_indices=ComputeSHR(log_spectrum,min_bin,startpos,endpos,lowerbound,upperbound,N,shift_units,nCand,ii)
    all_peak_indices = np.unique(all_peak_indices)
    r=np.zeros(len(all_peak_indices))
    for m in np.arange(0,len(all_peak_indices)):
        r[m]=newfre[int(all_peak_indices[m])]*2
    
    if(len(all_peak_indices)<nCand):
        ff0_candidates[-len(all_peak_indices):] = np.sort(r)
    else:
        ff0_candidates = np.sort(r)
    return(ff0_candidates)
    
def buffer(x,n,p):
    if p >= n:
        raise ValueError('p ({}) must be less than n ({}).'.format(p,n))
    cols = int(np.ceil(len(x)/float(n-p)))
    print(n,cols)
    # Create empty buffer array
    b = np.zeros((n, cols))
    # Fill buffer by column handling for initial condition and overlap
    j = 0
    for i in range(cols):
        # set first values of row to last p values
        if i != 0 and p != 0:
            b[:p, i] = b[-p:, i-1]
        # If initial condition, set p elements in buffer array to zero
        else:
            b[:p, i] = 0
        # Get stop index positions for x
        k = j + n - p
        # Get stop index position for b, matching number sliced from x
        n_end = p+len(x[j:k])
        # Assign values to buffer array from x
        b[p:n_end,i] = x[j:k]
        # Update start index location for next iteration of x
        j = k
    b1=np.matrix.transpose(b)
    return b1

def mySHR_new(signal,Fs,f0MinMax,wndwduration,wndwShift,nCand):
    """ Finding pitch candidates at every time instant  
        This function estimates 'nCand' pitch candidates at every time segment, which are
        'wndwShift' seconds apart.

        The spectrum is computed using a Hamming window of size wndwduration. The
        logarithmic spectrum is found. Using this, we find the SHR (subharmonic
        to harmonic ratio). nCand pitch candidates are computed using the SHR
        values."""
    
    wndwduration= wndwduration*1000
    wndwShift= wndwShift*1000
    nWndw=round(wndwduration*(Fs/1000))
    nShift=round(wndwShift*(Fs/1000))
    signal= signal- (np.mean(signal))
    signal = signal/(max(abs(signal)))
    signal=np.append(np.zeros(1),signal)
    
    frames_temp = buffer(signal,nWndw,nWndw - nShift)
    frames_temp=frames_temp[3:-1,:]
    win = sig.hamming(nWndw)
    frames_temp = np.multiply(frames_temp,win)
    sigFrames=frames_temp
    nf=len(sigFrames)
    f0 = np.zeros((nf,nCand))
    for i in np.arange(0,nf):
        segment=sigFrames[i,:]
        cur_cand=mySHR_Frame_4f0_new(segment,Fs,nWndw,f0MinMax,nCand,i)
        f0[i,:]= cur_cand
        if i>0:
            if(sum(cur_cand)==0):
                f0[i,:]=f0[i-1,:]
    z=np.zeros((3,nCand))
    f0 = np.insert(z,2,f0,axis=0)
    
    return(f0)

def FrameSelection(score,estVuV,cls1_cls2Data,cls1_Lngth):
    """This function selects the frames into DP and non-DP frames, using the
        K-nearest neighbours algorithm"""
    r=1
    score_sort = np.sort(score,axis=1)
    score_sort=score_sort[:,::-1]
    for m in np.arange(0,len(cls1_cls2Data)):
        if np.all(cls1_cls2Data[m]==-1):
            cls1_cls2Data[m]=[-100,-100,-100]
    for m in np.arange(0,len(score_sort)):
        if np.all(score_sort[m]==-1):
            score_sort[m]=[-100,-100,-100]
    for m in np.arange(0,len(score_sort)):
        if estVuV[m]==0:
            score_sort[m]=[0,0,0]
    inds=np.where(estVuV==1)
    score_voiced = score_sort[np.where(estVuV==1)]
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=r).fit(cls1_cls2Data)
    distances,idx=nbrs.kneighbors(score_voiced)
    t1 = np.where(idx>=cls1_Lngth)
    t2 = np.where(idx<cls1_Lngth)
    nonDPindices=np.zeros(len(t1[0]))
    DPindices=np.zeros(len(t2[0]))
    for m in np.arange(0,len(t1[0])):
        nonDPindices[m] = inds[0][t1[0][m]]
    for m in np.arange(0,len(t2[0])):
        DPindices[m]=inds[0][t2[0][m]]
    
    return(nonDPindices,DPindices)

def DP_Pitch_Estimation(f0_candidates,score,nonDPindices,DPindices):
    """This function outputs the pitch, given the pitch candidates, the
        confidence scores, and the frames selected for DP """
    
    rows=len(f0_candidates)
    cols=len(f0_candidates[0])
    pitch = np.zeros((1,rows))
    indsmax=np.argmax(score,axis=1)
    f0_candidates_dp = np.zeros((rows,cols))
    for m in np.arange(0,len(nonDPindices)):
        f0_candidates_dp[int(nonDPindices[m])] = f0_candidates[int(nonDPindices[m]),indsmax[int(nonDPindices[m])]]
        #print(f0_candidates_dp[int(nonDPindices[m]),:])
    for m in np.arange(0,len(DPindices)):
        f0_candidates_dp[int(DPindices[m]),:]=f0_candidates[int(DPindices[m]),:]
        #print(f0_candidates_dp[int(DPindices[m]),:])
    
    VuV = np.sign(abs(np.sum(f0_candidates_dp,axis=1)))
    boundary = abs(VuV-np.append(VuV[1:,],np.zeros(1)))
    boundary_inds = np.where(boundary==1)
    
    #for m in np.arange(0,len(f0_candidates_dp)):
        #print(f0_candidates_dp[m,:])
    for i2 in np.arange(0,len(boundary_inds[0]),2):
        inds_temp = np.arange(boundary_inds[0][i2]+1,boundary_inds[0][i2+1]+1)
        
        if len(inds_temp)>1:
            x = f0_candidates_dp[inds_temp,:]
            rows1=len(x)
            cols1=len(x[0])
            c=np.zeros((rows1,cols1))
            b=np.zeros((rows1,cols1))
            out=np.zeros((1,rows1))
            temp2=np.zeros((1,cols1))
            
            for j1 in np.arange(1,rows1):
                for j2 in np.arange(0,cols1):
                    for j3 in np.arange(0,cols1):
                        temp2[0][j3]=c[j1-1,j3]+np.square(x[j1,j2]-x[j1-1,j3])
                    c[j1,j2]=np.amin(temp2[0])
                    b[j1,j2]=np.argmin(temp2[0])

            indd = np.argmin(c[-1,:])
            for j in np.arange(len(out[0])-1,-1,-1):
                out[0][j]=x[j][int(indd)]
                indd=b[j][int(indd)]
            pitch[0][inds_temp]=np.matrix.transpose(out[0])
        else:
            pitch[0][inds_temp]=f0_candidates_dp[inds_temp,indsmax[inds_temp]]
    
    uvInds = np.where(VuV==0)
    for m in np.arange(0,len(uvInds[0])):
        pitch[0][uvInds[0][m]]=f0_candidates[uvInds[0][m],indsmax[uvInds[0][m]]]
    pitch = np.matrix.transpose(pitch)
    
    return(pitch)

def FSDP_Compute(signal, Fs, cls1_cls2Data, cls1_Lngth):
    """PITCH ESTIMATION USING FSDP APPROACH

    This function estimates the pitch using a frame selective dynamic
    programming approach, which uses the characteristics of the subharmonic
    to harmonic ratio (SHR) and sawtooth wave inspired pitch estimator
    (SWIPE) methods of estimating pitch. The frames are classified into two
    classes; for the first class, a confidence score maximisation is used to
    estimate pitch, while a dynamic programming approach is used for the
    other class.

    REFERENCE : Chiranjeevi Yarra, Om D Deshmukh, Prasanta Kumar Ghosh (2018),
    "A frame selective dynamic programming approach for noise robust pitch
    estimation" """

    warnings.filterwarnings("ignore");
    """Resampling of signal to 16KHz"""
    if Fs != 16000:
        signal1 = sig.resample(signal1, round(len(signal1)*(16000/Fs)))
        signal = signal1
        Fs = 16000
    """Pitch candidate selection"""
    nCand=3 # Number of pitch candidates
    f0_candidates=mySHR_new(signal,Fs,[50,500],0.04,0.01,nCand)
    """Confidence score computation"""
    temp = io.loadmat('E:\IIScInternshipWork\kernels1.mat')
    kernel_cell = temp['kernel_cell']
    score= pitchCandCost(signal,Fs,f0_candidates,kernel_cell)
    """VuV Classification"""
    estVuV1 = VuV_Compute(score)
    estVuV1[np.where(estVuV1-np.append(estVuV1[1:],np.zeros(1)))]=0
    estVuV1[np.where(estVuV1-np.append(np.zeros(1),estVuV1[:-1]))]=0    
    """Frame selection strategy"""
    nonDPindices, DPindices = FrameSelection(score,estVuV1,cls1_cls2Data,cls1_Lngth)
    """Dynamic programming based pitch estimation"""
    pitch = DP_Pitch_Estimation(f0_candidates,score,nonDPindices,DPindices)
    
    return(pitch)

