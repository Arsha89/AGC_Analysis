%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Simulation Study: Adaptive Granger Causality Analysis
% For details refer to the paper in Results section: A Simulated Example

clc
clear all
%close all

Tt = 120; % Total duration (sec)
delta = 0.001; % time bin (sec)
T = Tt/delta; % Number of time bins

%%% Model Specification for Neurons: Dynamic logic-linked GLM Model
%%% Construct a dynamic G-causality map matrix 
% contains the dynamics + signs of G-causal links
Ncells = 8; % Number of neurons/cells (network size)
Cmap = zeros(Ncells,Ncells,T); % Ground truth causality maps (scaling+sign)
ExcInhID = [1 -1 1 -1 1 -1 1]'; % sign vector specifying the excitatory or inhibitory nature of GC links
% This Causal pattern corresponds to the Simulation section of paper
for t = 1:T
    % Dynamic links
    Cmap(:,:,t) = (-1)*eye(Ncells); % Self-Inhibition for all neurons
    Cmap(2:end,1,t) = ExcInhID.*ones(Ncells-1,1)*(1 - min(max(t-T/3,0)*(1/(T/3)),1)); % shrinking links
    CC5 = 1:Ncells; CC5(5) =[];
    Cmap(CC5,5,t) = ExcInhID.*ones(Ncells-1,1)*(min(max(t-T/3,0)*(1/(T/3)),1)); % growing links
    % Static links
    Cmap(7,3,t) = 1; Cmap(6,8,t) = -1; Cmap(2,6,t) = 1;
end

%%% Parameter specification for GLM model
mspkmax = 0.08; muMax = log(mspkmax/(1-mspkmax));
mspkmin = 0.06; muMin = log(mspkmin/(1-mspkmin));
mspkavg = (mspkmax+mspkmin)/2; % desired average spiking rate
muBase = (muMax + muMin)/2; % baseline firing parameter \mu

%%% Model cross-history dependence among neurons
% Modulation parameter vector asscoiated with all the G-causal interactions
thetac = ones(10,1)*[1 0 0 2 0 0 0 0 0 1]; thetac = thetac(:);
Mc = numel(thetac); thetac = thetac/norm(thetac); % normalize thetac vector

%%% Generate spike train sequences for all neurons in the ensemble
%%% according to the dynamic causal structure Cmap
%%% [Uncomment for the generation of a new realization of spike sequences]
% resp = GenerateSpikes(thetac,muBase,Cmap);

%% LOAD Spike Trains from Previous Runs
% Load the spike trains already generated from the Simulation example in
% paper
load('DataSim_OriginalRun.mat')

%% Filtering and Deviance Computation
% Adaptive Estimation and Deviance Computation for FULL and REDUCED Models
% Adaptive Filtering setting [L1PPF1 parameters] 
LL = 1; % Number of iterations per step
cw = 1; % Step size constant
ff = 0.998; % Forgetting factor
W = 20; % window size (filtering)
Nw = floor(T/W); % Number of window segments
Devkt = zeros(Nw,Ncells,Ncells); 
wknft = cell(Ncells,Ncells);

%%% Form Cross-History Covariate Matrix
WH = 10; 
Mhc = 10; % Number of Cross-history parameters
Whc = WH*ones(1,Mhc); % Cross-history Kernel
Xcz = FormHistMatrix(resp,Whc); % resp of size [N x Ncells]
% Self-History setting
Mhcs = 10; % Number of Self-history parameters
Whcs = WH*ones(1,Mhcs); % Self-history Kernel
Mf = (Ncells-1)*Mhc+Mhcs+1; % Dimension of Full GLM model
Mr = (Ncells-2)*Mhc+Mhcs+1; % Dimension of Reduced GLM model
Md = Mf - Mr; % Dimensionality difference 

%%% Two-fold even-odd Cross-validation on gamma
KK = 2; % KK-fold Cross-valid
gammacv = [0.25:0.05:0.65]; % Choice of gamma's 
Whcv = {Whc}; Whscv = {Whcs};
gammaOpt = CrossValidwin(resp,gammacv,Whcv,Whscv,KK,ff,W,cw,LL);
tic
%%% Perform AGC+deviance computation + Sparse Adaptive Filtering 
for ct = 1:Ncells % ct: index of target cell
    %%% Prepare for Filtering/Estimation
    N_indto = num2str(ct);
    robs = resp(:,ct);
    cellC = 1:Ncells; cellC(ct) = [];
    %%% Form Self-History Covariate matrix
    Xcsz = FormHistMatrix(robs,Whcs);
    %%% Full Model
    %%% Form standardized Full Model Covariate Matrix
    Xfn = FormFullDesign(Xcz,Xcsz,ct);
    %%% Filtering Stage
    % Set initials for estimation
    navg = mean(robs);
    w0f = zeros(Mf,1); w0f(1) = log(navg/(1-navg));
    u0f = 0; v0f = zeros(Mf,1); U0f = 1*eye(Mf);
    gammact = gammaOpt(ct);
    gammaEffct = gammact*sqrt(W*log(Mf)*navg*(1-navg)/(1-ff));
    [wknf,Devkf,ukf,vkf,Ukf] = L1PPF1ModDev(Xfn,robs,ff,gammact,LL,cw,W,w0f,u0f,v0f,U0f);
    wknft{ct} = wknf;
    %%% Analyze the GC from Neuron N_indfrom to Neuron N_indto :
    for cf = cellC % cf : index of Electrode/Unit we check causality from to the target
        N_indfrom = num2str(cf);
        disp(['Estimating Causality from cell ' N_indfrom ' to cell ' N_indto ' ...'])
        %%% Reduced Model
        %%% Form standardized Reduced Model Covariate Matrix
        Xrn = Xfn;
        cc = find(cellC == cf);
        Xrn(:,(cc-1)*Mhc+Mhcs+2:cc*Mhc+Mhcs+1) = [];
        %%% Filtering Stage
        % Set initials for estimation
        w0r = zeros(Mr,1); w0r(1) = w0f(1);
        u0r = 0; v0r = zeros(Mr,1); U0r = 1*eye(Mr);
        [wknr,Devkr,ukr,vkr,Ukr] = L1PPF1ModDev(Xrn,robs,ff,gammact,LL,cw,W,w0r,u0r,v0r,U0r);
        %%% Adaptive Granger Causality metric: 2nd order Taylor Approx.
        % Compute Deviance Difference Recursively
        Devkd = (1+ff)*(Devkf - Devkr);
        Devkt(:,ct,cf) = Devkd;
    end
end
toc
tic
%%% Non-central Chi2 Filtering & Smoothing Algorithm
sz0 = 5*10^(-6); % variance of zk serves as SMOOTHING FACTOR
rho = 1; % scaling factor
NN = 20; % Number of newtons iterations
Nem = 1; % Number of EM iterations
[nukSmth,nukSmthU,nukSmthL,nukFilt,nukFiltU,nukFiltL] = NoncentChi2FiltSmooth(Devkt,Md,sz0,rho,NN,Nem);

%%% Statistical Inference Test: FDR Control 
% Benjamini-Hochburg-Yuketieli Test
alpha = 0.1; % FDR rate
[AGCs,AGC] = FDRcontrolBY(Devkt,nukSmth,alpha,Md,wknft,Whc); % J-statistics based on mFDR
toc

%% RESULTS: Plot Figures 
%% Plot the simulated Spike Trains for all neurons (Fig 3.B)
Lwin = 1000; % window size (bins) for plotting spike trains
Twin = [Tt/3,Tt/2,Tt]/delta; % Time instances (endpoints) of windows
Ntw = numel(Twin);

hfB = figure;
for cc = 1:Ncells
    for tt = 1:Ntw
        subplot(Ncells,Ntw,(cc-1)*Ntw+tt),stem(resp(Twin(tt)-Lwin:Twin(tt),cc),'k','Marker','none','Linewidth',1.5);
        axis tight
        set(gca,'xTick',[],'yTick',[]);
        if tt ==1
            ylabel(['Cell' num2str(cc)])
        end
    end
end
set(hfB,'position',[100,700,1700,400])
suptitle(['Simulated spike trains within windows of ' num2str(Lwin*delta) 's selected @ t \in \{' num2str(Twin(1)*delta) ',' num2str(Twin(2)*delta) ',' num2str(Twin(3)*delta) '\} s'])

%% Plot estimates of Non-centrality for Selected Candidate GC Links (Fig 3.C)
ct = [2,4,6,2]; 
cf = [5,1,8,8];
nct = numel(ct);

cols = ['b','r','g','m'];
hfC = figure;
for i = 1:numel(ct)
    plot((1:Nw)*W*delta,nukSmth(:,ct(i),cf(i)),cols(i),'LineWidth',2)
    hold on
    plot((1:Nw)*W*delta,nukSmthU(:,ct(i),cf(i)),cols(i))
    plot((1:Nw)*W*delta,nukSmthL(:,ct(i),cf(i)),cols(i))
    plot((1:Nw)*W*delta,Devkt(:,ct(i),cf(i))-Md,'k')
end
axis tight
ylim([-Md,100])
set(gca,'xtick',[0:Tt/3:Tt]);
set(hfC,'position',[100,500,1700,400])
title(['Estimated non-centrality across time corresponding to four selected GC links'])

%% Plot the J-statistics for Selected Candidate GC Links (Fig 3.D)
hfD = figure;
for i = 1:nct
    subplot(nct,1,i),plot((1:Nw)*W*delta,AGC(:,ct(i),cf(i)),cols(i),'LineWidth',2)
    ylim([0,1])
    set(gca,'xtick',[0:Tt/3:Tt]);
end
set(hfD,'position',[100,300,1700,400])
suptitle(['Four panels of estimated J-statistics corresponding to the selected GC links @ FDR = ' num2str(alpha)])

%% Plot estimated GC maps in the form of matrices Phi_k (Fig 3.E)
Tsnaps = [20,30,40,60,70,80,100,110,120]; % time instances (sec)
Nsnp = numel(Tsnaps);

hfE = figure;
%%% Plot ground truth GC maps for Comparison purposes
cmax = +1; cmin = -cmax;
for i = 1:Nsnp
    Cmapi = Cmap(:,:,Tsnaps(i)/delta); Cmapi = Cmapi - diag(diag(Cmapi));
    subplot(2,Nsnp,i), imagesc(Cmapi)
    caxis([cmin,cmax]);
end

%%% Plot the estimated AGC maps using our proposed method
cmax = max(max(max(abs(AGCs)))); cmin = -cmax;
for k = 1:Nsnp
    Phik = squeeze(AGCs(Tsnaps(k)/(delta*W),:,:));
    subplot(2,Nsnp,Nsnp+k), imagesc(Phik)
    caxis([cmin,cmax]);
end
colormap jet
set(hfE,'position',[100,100,1700,400])
suptitle(['Performance of the AGC method @ FDR = ' num2str(alpha)])

%%
% save('SimDataRuns8.mat','resp','thetac','muBase','mspkavg')
% save('DataSimResultsAdapGCAnals_Runs8.mat','AGCs','GC1','GC2','wknft','Devkt','nukSmth','Md','Mhcs','sz0','alpha','cw','mspkavg','thetac')
