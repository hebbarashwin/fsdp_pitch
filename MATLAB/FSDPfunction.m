function FSDPfunction

allCorpus='KEELE';
allSnr = 0; %%%Clean signal
srcCorpus = 'KEELE';
ext= 'New';
DPcostChoice = 'euc';
wavDir= 'E:\IIScInternshipWork\wav\';

method = 'KNN';
nCand =3;
load (['optKNN',ext,'_',num2str(nCand),'_',srcCorpus],'optData','DPsize');
cls1_cls2Data= optData;
%cls1_Lngth=DPsize;
cls_label = [ones(DPsize,1);zeros(length(cls1_cls2Data)-DPsize,1)];

gpe20=zeros(2,1);
snr=[];
corpus=allCorpus;
if(isempty(snr))
    spurtsDir='spurts';
else
    spurtsDir= strcat('spurts_snr',num2str(snr));
end

%KEELE
temp= load('keeleSpurtNames'); tempFieldName= fieldnames(temp); estSpurtName= getfield(temp,tempFieldName{1});
estVuVDir= ['E:\IIScInternshipWork\',spurtsDir,'\'];
pchDir= ['E:\IIScInternshipWork\pch\'];
gpe_20= []; rmse= []; gpe_20woDP= []; rmse_woDP= [];

if isempty(snr)
    load(['pitchCand',ext,'_',num2str(nCand),'_',corpus]);
else
    load(['pitchCand',ext,'_',num2str(nCand),'_',corpus,'_snr',num2str(snr)]);
end
for i= 1:length(estSpurtName)
    pch= load([pchDir,estSpurtName{i},'.pch']); if size(pch,1)<size(pch,2); pch= pch'; end
    if (strcmp(corpus,'KEELE'))
        pch(pch<=0)= Inf; pch= 20000*ones(length(pch),1)./pch; % unComment this line for KEELE database
    end
    load([estVuVDir,estSpurtName{i}]);
%     estVuV(estVuV-[0,estVuV(1:end-1)]~=0)=0;
%     estVuV(estVuV-[estVuV(2:end),0]~=0)=0;
    groundVuV(groundVuV-[0,groundVuV(1:end-1)]~=0)=0;
    groundVuV(groundVuV-[groundVuV(2:end),0]~=0)=0;
    inds= find(groundVuV==1);
    
    score = score_all{i};
    [signal,Fs]= audioread([wavDir,estSpurtName{i},'.wav']);
    %[signal,Fs]= audioread('E:\IIScInternshipWork\Recording.wav');
    pitch = FSDP_Compute(signal(:,1),Fs,cls1_cls2Data,cls_label);
     %%GPE
    %abs(pitch(inds)-pch(inds)) - 0.2*pch(inds)
    %find(abs(pitch(inds)-pch(inds))>0.2*pch(inds))
    gpe_20 = [gpe_20;[length(find(abs(pitch(inds)-pch(inds))>0.2*pch(inds))),length(inds)],length(pch)];
end
 
allgpe20 = gpe_20;
nComp=1;
temp = sum(gpe_20)/sum(gpe_20(:,(nComp+1)));
gpe20(1,1)=temp(1);

gpe20_allsrcCorpus{1}=gpe20*100;
disp(gpe20*100);
end

function [pitch]=FSDP_Compute(signal,Fs,cls1_cls2Data,cls_label)

% PITCH ESTIMATION USING FSDP APPROACH
% 
% This function estimates the pitch using a frame selective dynamic
% programming approach, which uses the characteristics of the subharmonic
% to harmonic ratio (SHR) and sawtooth wave inspired pitch estimator
% (SWIPE) methods of estimating pitch. The frames are classified into two
% classes; for the first class, a confidence score maximisation is used to
% estimate pitch, while a dynamic programming approach is used for the
% other class.
% 
% REFERENCE : Chiranjeevi Yarra, Om D Deshmukh, Prasanta Kumar Ghosh (2018),
% "A frame selective dynamic programming approach for noise robust pitch
% estimation"

if isempty(cls1_cls2Data) || isempty(cls_label), disp('Please include the Class1_Class2_Data');return; end
if size(signal,1) == 1
    signal = signal';
end
%%Resampling to 16KHz
if Fs ~= 16000
    signal = resample(signal, 16000, Fs);
    Fs = 16000;
end
%%PitchCandidateSelection
nCand = 3; %Number of pitch candidates
f0_candidates= mySHR_new(signal,Fs,[50,500],0.04,0.01,nCand);
%%ComputingConfidenceScores
score= pitchCandCost(signal,Fs,f0_candidates);
%[score_sort,~]=sort(score,2,'descend');
%%VuV Classification
estVuV = VuV_Compute(score');
estVuV(estVuV-[0,estVuV(1:end-1)]~=0)=0;
estVuV(estVuV-[estVuV(2:end),0]~=0)=0;
%%FrameSelectionStrategy
[nonDPframeInds, DPframeInds] = FrameSelection(score,estVuV,cls1_cls2Data,cls_label);
%%DPbasedPitchEstimation
pitch = DP_Pitch_estimation(f0_candidates,score,nonDPframeInds,DPframeInds);

end

function [f0_candidates]= mySHR_new(signal,Fs,f0MinMax,wndwduration,wndwShift,nCand)

%  Finding pitch candidates at every time instant  
% f0_candidates = mySHR_new(signal,Fs,f0MinMax,wndwduration,wndwShift,nCand)
% estimates 'nCand' pitch candidates at every time segment, which are
% 'wndwShift' seconds apart.
% 
% The spectrum is computed using a Hamming window of size wndwduration. The
% logarithmic spectrum is found. Using this, we find the SHR (subharmonic
% to harmonic ratio). nCand pitch candidates are computed using the SHR
% values.

% fid=fopen('E:\IIScInternshipWork\signal.txt','w');
% fprintf(fid, '%f, ', signal(1:10000));
% fclose(fid);true

wndwduration= wndwduration*1000;
wndwShift= wndwShift*1000;
nWndw=round(wndwduration*(Fs/1000));
nShift=round(wndwShift*(Fs/1000));
%signal=signal-mean(signal);
%signal=signal/max(abs(signal));
frames_temp= buffer([0;signal],nWndw,(nWndw-nShift))';
frames_temp= frames_temp(4:end-1,:);

win= window(@hamming,nWndw)';
frames_temp = frames_temp .* win(ones(size(frames_temp,1),1),:);
sigFrames= frames_temp;
[nf,~]=size(sigFrames);
clear Y;
% nCand= 2;
f0_candidates=zeros(nf,nCand);


for i=1:nf
    segment=sigFrames(i,:);
    
    [cur_cand]= mySHR_Frame_4f0_new(segment,Fs,nWndw,f0MinMax,nCand); % This is for obtaining candidates in roll over basis when the candidates are exceeded the boundaries.
    f0_candidates(i,:)= cur_cand;
    if i>1
        if(sum(cur_cand)==0)
            i;
            f0_candidates(i,:)= f0_candidates(i-1,:);
            cur_cand;
        end
    end
end
f0_candidates= [zeros(2,nCand);f0_candidates;zeros(1,nCand)];
end


function [f0_candidates]= mySHR_Frame_4f0_new(sigFrame,Fs,nWndw,f0MinMax,nCand)
minf0= f0MinMax(1);
maxf0= f0MinMax(2);
ceiling=1250; % max frequency for analysis.
interpolation_depth=0.5;
maxlogf=log2(maxf0/2);
minlogf=log2(minf0/2); % the search region to compute SHR is as low as 0.5 minf0.
N=floor(ceiling/minf0); % maximum number harmonics
m=mod(N,2);
N=N-m;
N=N*4;
%------------------ determine FFT length ---------------------
fftlen=1;
while (fftlen < nWndw * (1 +interpolation_depth))
    fftlen =fftlen* 2;
end
% fftlen= 8192;
frequency=Fs*(1:fftlen/2)/fftlen; % we ignore frequency 0 here since we need to do log transformation later and won't use it anyway.
limit=find(frequency>=ceiling);
limit=limit(1); % only the first is useful
frequency=frequency(1:limit);
logf=log2(frequency);

clear frequency;
min_bin=logf(end)-logf(end-1); % the minimum distance between two points after interpolation
shift=log2(N); % shift distance
shift_units=round(shift/min_bin); %the number of unit on the log x-axis
i=(2:N);
% ------------- the followings are universal for all the frames ---------------%%
startpos=shift_units+1-round(log2(i)/min_bin);  % find out all the start position of each shift
index= startpos<1; % find out those positions that are less than 1
startpos(index)=1; % set them to 1 since the array index starts from 1 in matlab
interp_logf=logf(1):min_bin:logf(end);
interp_len=length(interp_logf);% new length of the amplitude spectrum after interpolation
totallen=shift_units+interp_len;
endpos=startpos+interp_len-1; %% note that : totallen=shift_units+interp_len;
index= endpos>totallen;
endpos(index)=totallen; % make sure all the end positions not greater than the totoal length of the shift spectrum

newfre=2.^(interp_logf); % the linear Hz scale derived from the interpolated log scale
upperbound=find(interp_logf>=maxlogf); % find out the index of upper bound of search region on the log frequency scale.
upperbound=upperbound(1);% only the first element is useful
lowerbound=find(interp_logf>=minlogf); % find out the index of lower bound of search region on the log frequency scale.
lowerbound=lowerbound(1);

% nCand= 3;
f0_candidates= zeros(1,nCand);
[log_spectrum]=GetLogSpectrum(sigFrame,fftlen,limit,logf,interp_logf);
[peak_index,all_peak_indices]=ComputeSHR(log_spectrum,min_bin,startpos,endpos,lowerbound,upperbound,N,shift_units,nCand);
all_peak_indices= unique(all_peak_indices);
% if (peak_index~=-1) % -1 indicates a possibly unvoiced frame, if CHECK_VOICING, set f0 to 0, otherwise uses previous value

if (length(all_peak_indices)<nCand)
    f0_candidates((end-length(all_peak_indices)+1):end)= sort(newfre(all_peak_indices)*2);
else
    f0_candidates= sort(newfre(all_peak_indices)*2);
end

end

function [interp_amplitude]=GetLogSpectrum(segment,fftlen,limit,logf,interp_logf)
%--------------------------To find Log spectrum----------------------
%  fid=fopen('E:\IIScInternshipWork\MyFile1.txt','w');
%  fprintf(fid, '%f, ', segment');
%  fclose(fid);true
Spectra=fft(segment,fftlen);
amplitude = abs(Spectra(1:fftlen/2+1)); % fftlen is always even here. Note: change fftlen/2 to fftlen/2+1. bug fixed due to Herbert Griebel
amplitude=amplitude(2:limit+1); % ignore the zero frequency component
interp_amplitude=interp1(logf,amplitude,interp_logf,'linear');
interp_amplitude=interp_amplitude-min(interp_amplitude);
end

function [peak_index,index]=ComputeSHR(log_spectrum,min_bin,startpos,endpos,lowerbound,upperbound,N,shift_units,nCand)
% computeshr: compute subharmonic-to-harmonic ratio for a short-term signal
len_spectrum=length(log_spectrum);
totallen=shift_units+len_spectrum;
shshift=zeros(N,totallen); %initialize the subharmonic shift matrix; each row corresponds to a shift version
shshift(1,(totallen-len_spectrum+1):totallen)=log_spectrum; % place the spectrum at the right end of the first row
% note that here startpos and endpos has N-1 rows, so we start from 2
% the first row in shshift is the original log spectrum
for i=2:N
    shshift(i,startpos(i-1):endpos(i-1))=log_spectrum(1:endpos(i-1)-startpos(i-1)+1); % store each shifted sequence
end
shshift=shshift(:,shift_units+1:totallen); % we don't need the stuff smaller than shift_units
shsodd=sum(shshift(1:2:N-1,:),1);
shseven=sum(shshift(2:2:N,:),1);
difference=shseven-shsodd;
[mag,index]=twomax(difference,lowerbound,upperbound,min_bin,nCand); % only find two maxima
if(min(mag)<=0)
    peak_index=-1;
else
    peak_index= 1;
end
end

%******************    this function only finds two maximum peaks   ************************
function [mag,index]=twomax(x,lowerbound,upperbound,unitlen,nCand)
lenx=length(x);
[mag,index]=max(x(lowerbound:upperbound));%find the maximum value
if (mag<=0)
    return
end
index=index+lowerbound-1;
startInd= index;
if (mod(nCand,2)==0)
    temp= -(nCand/2-1):1:(nCand/2);
    allharmonics=2.^temp;
else
    temp= -(floor(nCand/2)):1:(floor(nCand/2));
    allharmonics=2.^temp;
end
LIMIT=0.0625; % 1/8 octave
abvRange= find(startInd+round(log2(allharmonics-LIMIT)/unitlen) > min(lenx,upperbound));
allharmonics(abvRange)= [];
if ~isempty(abvRange)
    allharmonics= [2.^(log2(min(allharmonics))-1:length(abvRange)),allharmonics];
    allharmonics(startInd+round(log2(allharmonics+LIMIT)/unitlen) < max(1,lowerbound)) = [];
end
blwRange= find(startInd+round(log2(allharmonics+LIMIT)/unitlen) < max(1,lowerbound));
allharmonics(blwRange)= [];
if ~isempty(blwRange)
    allharmonics= [2.^(log2(max(allharmonics))+1:length(blwRange)),allharmonics];
    allharmonics(startInd+round(log2(allharmonics-LIMIT)/unitlen) > min(lenx,upperbound)) = [];
end
allharmonics(allharmonics==1)= [];

    
for i= 1:length(allharmonics)
    harmonics= allharmonics(i);
    startpos=startInd+round(log2(harmonics-LIMIT)/unitlen);
    if (startpos<=min(lenx,upperbound))
        endpos=index+round(log2(harmonics+LIMIT)/unitlen); % for example, 100hz-200hz is one octave, 200hz-250hz is 1/4octave
        if (endpos> min(lenx,upperbound))
            endpos=min(lenx,upperbound);
        end
        if (startpos<1)
            startpos= 1;
        end
        [mag1,index1]=max(x(startpos:endpos));%find the maximum value at right side of last maximum
%         if (mag1>0)
            index1=index1+startpos-1;
            mag=[mag;mag1];
            index=[index;index1];
%         end
    end
end
end

function score= pitchCandCost(signal,Fs,f0_candidates)

% The function, score = pitchCandCost(signal,Fs,f0_candidates) estimates the
% confidence score for each of the pitch candidates in f0_candidates, by
% using the SWIPE algorithm.


[allS,logP]= swipep_mod(signal,Fs,[75 500],0.01,[],1/20,0.5);
%[~,~,~,allS,logP]= swipep_mod(signal,Fs,[75 500],0.01,[],1/20,0.5,0.2);
p= 2.^logP;
allS= allS(:,1:end-1);
[rows,cols]= size(f0_candidates);
s_swipe_new= zeros(rows,cols);

%num = zeros(length(p),1);
for i=1:rows
    for j= 1:cols
        if (f0_candidates(i,j)~=0)
            [~,ind]= min(abs(p-f0_candidates(i,j)));
            %num(ind) = num(ind)+1;
            s_swipe_new(i,j)= allS(ind,i);
        else
            s_swipe_new(i,j)= -1;
        end
    end
    
end

score= s_swipe_new;
end

function [S,log2pc] = swipep_mod(x,fs,plim,dt,dlog2p,dERBs,woverlap)
% SWIPEP Pitch estimation using SWIPE'.
%    P = SWIPEP(X,Fs,[PMIN PMAX],DT,DLOG2P,DERBS) estimates the confidence
%    score of the vector signal X every DT seconds. The sampling frequency
%    of the signal is Fs (in Hertz). The spectrum is computed using a Hann
%    window with an overlap WOVERLAP between 0 and 1. The spectrum is
%    sampled uniformly in the ERB scale with a step size of DERBS ERBs. The
%    pitch is searched within the range [PMIN PMAX] (in Hertz) with samples
%    distributed every DLOG2P units on a base-2 logarithmic scale of Hertz.
%
%%%% Original code for SWIPE was written by Camacho et al. 
%%%% We have modified the above said code.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~ exist( 'plim', 'var' ) || isempty(plim), plim = [30 5000]; end
if ~ exist( 'dt', 'var' ) || isempty(dt), dt = 0.001; end
if ~ exist( 'dlog2p', 'var' ) || isempty(dlog2p), dlog2p = 1/48; end
if ~ exist( 'dERBs', 'var' ) || isempty(dERBs), dERBs = 0.1; end
if ~ exist( 'woverlap', 'var' ) || isempty(woverlap)
    woverlap = 0.5;
elseif woverlap>1 || woverlap<0
    error('Window overlap must be between 0 and 1.')
end
if ~ exist( 'sTHR', 'var' ) || isempty(sTHR), sTHR = -Inf; end
t = [ 0: dt: length(x)/fs ]'; % Times
t=t(1:end-1);
% Define pitch candidates
log2pc = [ log2(plim(1)): dlog2p: log2(plim(2)) ]';
pc = 2 .^ log2pc;
S = zeros( length(pc), length(t) ); % Pitch strength matrix
% Determine P2-WSs
logWs = round( log2( 8*fs ./ plim ) ); 
ws = 2.^[ logWs(1): -1: logWs(2) ]; % P2-WSs
myN= 8;
pO = myN * fs ./ ws; % Optimal pitches for P2-WSs
% Determine window sizes used by each pitch candidate
d = 1 + log2(pc) - log2( myN*fs./ws(1) );
% Create ERB-scale uniformly-spaced frequencies (in Hertz)
fERBs = erbs2hz([ hz2erbs(min(pc)/4): dERBs: hz2erbs(fs/2) ]');

temp=load('E:\IIScInternshipWork\kernels2.mat');
tempFieldName= fieldnames(temp); kernel_cells= getfield(temp,tempFieldName{1});

for i = 1 : length(ws)
    dn = max( 1, round( myN*(1-woverlap) * fs / pO(i) ) ); % Hop size
    % Zero pad signal
    xzp = [ zeros( ws(i)/2, 1 ); x(:); zeros( dn + ws(i)/2, 1 ) ];
    % Compute spectrum
%     w= ones(ws(i),1); % rectangular window
    w = hanning( ws(i) ); % Hann window 
    o = max( 0, round( ws(i) - dn ) ); % Window overlap
    [ X, f, ti ] = specgram( xzp, ws(i), fs, w, o );
    % Select candidates that use this window size
    if length(ws) == 1
        j=[(pc)]'; k = [];
    elseif i == length(ws)
        j=find(d-i>-1); k=find(d(j)-i<0);
    elseif i==1 
        j=find(d-i<1); k=find(d(j)-i>0);
    else
        j=find(abs(d-i)<1); k=1:length(j);
    end

    % Compute loudness at ERBs uniformly-spaced frequencies
    %fERBs = fERBs( find( fERBs > pc(j(1))/4, 1, 'first' ) : end );
    L = sqrt( max( 0, interp1( f, abs(X), fERBs, 'spline', 0) ) );
    % Compute pitch strength
    Si = pitchStrengthAllCandidates(L, pc(j),kernel_cells{i},j(1));
    
    % Interpolate pitch strength at desired times
    if size(Si,2) > 1
        warning off MATLAB:interp1:NaNinY
        Si = interp1( ti, Si', t, 'linear', NaN )';
        warning on MATLAB:interp1:NaNinY
    else
        Si = repmat( NaN, length(Si), length(t) );
    end
    % Add pitch strength to combination
    lambda = d( j(k) ) - i;
    mu = ones( size(j) );
    mu(k) = 1 - abs( lambda );
    S(j,:) = S(j,:) + repmat(mu,1,size(Si,2)) .* Si;
end
end

function S = pitchStrengthAllCandidates(L, pc, kernels, J1)
% Calculate pitch strength matrix
S = zeros( length(pc), size(L,2) );
NL = L ./ repmat(sqrt(sum(L.*L , 1)), size(L,1),1); 

for j = 1 : length(pc)
    S(j,:) = cell2mat(kernels(j+J1-1))' * NL;
end
end

function erbs = hz2erbs(hz)
erbs = 6.44 * ( log2( 229 + hz ) - 7.84 );
end

function hz = erbs2hz(erbs)
hz = ( 2 .^ ( erbs./6.44 + 7.84) ) - 229;
end

function estVuV = VuV_Compute(score)

% This function classifies the frames into voiced and unvoiced frames. It
% uses a SVM based classifier to do this.
% 
% NOTE : Use 'classify.py' if using Python3 , 'classify1.py' if using
% Python2
       
nCand = 3;
srcCorpus = 'KEELE';
ext= 'New';
[score_sort,~]=sort(score,1,'descend');
estVuV= 0*(1:size(score_sort,2));
indsRemain= 1:size(score_sort,2);
inds= [];
for i= 1:nCand
    currCand= i;
    if currCand<nCand
        tempInds= currCand+1:nCand; 
        inds= [inds,find(sum(score_sort(tempInds,:),1)==-1*length(tempInds))];  
    else
        inds= indsRemain;
    end
    %disp(inds);
    feature= score(1:nCand,inds); save('E:\IIScInternshipWork\pythonCodes\testFeature.mat','feature');
    file= dir(['E:\IIScInternshipWork\pythonCodes\',srcCorpus,ext,'_model',num2str(currCand),'.pk1']);

    if ~isempty(file)
        system(['copy E:\IIScInternshipWork\pythonCodes\',srcCorpus,ext,'_model',num2str(currCand),'.pk1 E:\IIScInternshipWork\pythonCodes\model.pk1']);
        system('py E:\IIScInternshipWork\pythonCodes\classify.py');
        currEstVuV= load('E:\IIScInternshipWork\pythonCodes\SVMOutput.txt');
        estVuV(inds)= currEstVuV;
        indsRemain(ismember(indsRemain,inds))= [];
        inds= [];
    end
end
end

function [nonDPframeInds,DPframeInds]=FrameSelection(score,estVuV,cls1_cls2Data,cls_label)

% This function selects the frames into DP and non-DP frames, using the
% K-nearest neighbours algorithm

r=1;
[score_sort,~]=sort(score,2,'descend');
cls1_cls2Data(cls1_cls2Data == -1)= -100;
score_sort(score_sort == -1) = -100;
score_sort(estVuV ==0, :)=0;

score_voiced = score_sort(estVuV==1,:);
inds1 = find(estVuV==1);
IDX=knnsearch(cls1_cls2Data,score_voiced,'K',r);
%temp1=sum((IDX>=cls1_Lngth),2)>floor(r/2);
temp3 = cls_label(IDX)==0;
% nonDPframeInds = inds1(temp1);
% DPframeInds=inds1(~temp1);

nonDPframeInds = inds1(temp3);
DPframeInds = inds1(~temp3);

end

function pitch = DP_Pitch_estimation(f0_candidates,score,nonDPframeInds,DPframeInds)

% This function outputs the pitch, given the pitch candidates, the
% confidence scores, and the frames selected for DP

[rows,cols] = size(f0_candidates);
pitch = zeros(1,rows);

[~,indsMax]=max(score,[],2);
f0_candidates_dp = zeros(size(f0_candidates));
for i1=1:length(nonDPframeInds)
    f0_candidates_dp(nonDPframeInds(i1),:)=repmat(f0_candidates(nonDPframeInds(i1),indsMax(nonDPframeInds(i1))),1,cols);
end
f0_candidates_dp(DPframeInds,:)=f0_candidates(DPframeInds,:);

VuV = sign(abs(sum(f0_candidates_dp,2)));
boudry = abs(VuV-[0;VuV(1:end-1)]);
boudry_inds= find(boudry==1);

for i2 = 1:2:length(boudry_inds)
    inds_temp=boudry_inds(i2):boudry_inds(i2+1)-1;
    if(length(inds_temp)>1)
        x = f0_candidates_dp(inds_temp,:);
        [rows1,cols1]=size(x);
        c=zeros(size(x));
        b=c;
        out = zeros(1,rows1);
        temp2=zeros(1,cols1);
        for j1= 2:rows1
            for j2=1:cols1
                for j3=1:cols1
                    temp2(j3)=c(j1-1,j3)+(x(j1,j2)-x(j1-1,j3)).^2;
                end
                %temp2
                [c(j1,j2),b(j1,j2)]=min(temp2);
            end
        end
        [~,indd]=min(c(end,:));
        for j = length(out):-1:1
            out(j)=x(j,indd);
            indd = b(j,indd);
        end
        pitch(inds_temp)=out';
    else
        pitch(inds_temp) = f0_candidates_dp(inds_temp,indsMax(inds_temp));
    end
end
uvInds = find(VuV==0);
for j=1:length(uvInds)
    pitch(uvInds(j))=f0_candidates(uvInds(j),indsMax(uvInds(j)));
end
pitch = pitch';

end