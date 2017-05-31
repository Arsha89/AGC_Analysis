function [CM1,CM2] = GCmethods(resp,Wml,mopt,WH,alpha)

% This function corresponds to two alternative GC inference methods:
% 1) the static GC method of Kim et al. 2011,
% Estimates the static GC maps corresponding to the windows of length Wml
% samples from spike actvities of an ensemble of neruons

% 2) the functional connectivity analysis approach of Okatan et al. 2005, 
% Estimates the functional network likelihood parameters as surrogates of 
% causality [minor modification applied for comparison purposes] 

% Parameters:
% mopt: optimal order of AR for ML estimation [Obtained often via AIC]
% WH: window size of history kernel (size of spike counting windows)
% resp: array of size [T x Ncells] as spike actvities of neuronal ensemble,
% where T is total number of samples/bins, and Ncells is number of neurons
% alpha: FDR rate 

% Outputs: 
% CM1: output of causal mesure (1)  
% CM2: output of causal mesure (2)

% Params for ML estimation 
NN = 5; % number of iterations per step (like LL)
aa = 0;
cstep = 1;

% Prepare Data matrices for MLE estimate for optimal selected model order
[T,Ncells] = size(resp);
Whcml = WH*ones(1,mopt); % history kernel
Lhcml = sum(Whcml);
next = [zeros(Lhcml,Ncells);resp];
Xml = zeros(T,mopt,Ncells);
Lhself1ml = zeros(mopt,1); Lhself2ml = Lhself1ml;
for j = 1:mopt
    Lhself1ml(j) = sum(Whcml(mopt-j+2:mopt));
    Lhself2ml(j) = sum(Whcml(mopt-j+1:mopt));
end
for k = 1:T
    for cc = 1:Ncells
        % Spiking History Components WINDOW-BASED
        for j = 1:mopt
            Xml(k,mopt-j+1,cc) = sum(next(k+Lhself1ml(j):k+Lhself2ml(j)-1,cc));
        end 
    end
end
% Zero-mean History Components
Xzml = Xml;
for cc = 1:Ncells
    Xzml(:,:,cc) = Xml(:,:,cc) - ones(T,1)*mean(Xml(:,:,cc)); 
end
% FULL Model common for ALL cells
Mfml = (Ncells)*mopt+1; 
Xfml = zeros(T,Mfml); Xfml(:,1) = ones(T,1);
for cc = 1:Ncells
    Xfml(:,(cc-1)*mopt+2:cc*mopt+1) = Xzml(:,:,cc);
end
varXfml = var(Xfml); varXfml(1) = 1;
Dfnml = diag(sqrt(varXfml));
% Standardize the covariate matrix
Xfnml = Xfml/Dfnml;

%%% ML Estimation and Deviance Computation for static GC
Dmltotal = zeros(ceil(T/Wml),Ncells,Ncells);
wmltotal = cell(Ncells,1);
Hesstotal = cell(Ncells,1);

for ct = 1:Ncells % ct: index of target cell
    %%% Prepare for Filtering/Estimation
    N_indto = num2str(ct);
    robs = resp(:,ct);
    navg = mean(robs);
    cellC = 1:Ncells; cellC(ct) = [];
    
    %%% MLE estimation for KIM deviance comp
    disp(['Estimating ML Estimate...'])
    w0fml = zeros(Mfml,1); w0fml(1) = log(navg/(1-navg));
    tic
    [wfnML,DevfML,HessML] = MLEwinDev(Xfnml,robs,NN,cstep,aa,Wml,w0fml);
    toc
    whatml = wfnML/Dfnml;
    
    for cf = cellC % cf : index of Electrode/Unit we check causality from to the target
        N_indfrom = num2str(cf) ;
        disp(['Estimating Causality from cell ' N_indfrom ' to cell ' N_indto ' ...'])
        %%% MLE estimation for KIM deviance comp
        Xrnml = Xfnml;
        Xrnml(:,(cf-1)*mopt+2:cf*mopt+1) = [];
        Mrml = size(Xrnml,2);
        w0rml = zeros(Mrml,1); w0rml(1) = log(navg/(1-navg));
        
        tic
        [~,DevrML,~] = MLEwinDev(Xrnml,robs,NN,cstep,aa,Wml,w0rml);
        toc
        Dnml = DevfML - DevrML; 
        Dmltotal(:,ct,cf) = Dnml;
    end
     wmltotal{ct} = whatml;  Hesstotal{ct} = HessML;
end


%%% Statistical Test: FDR control on static GC estimates
Nl = Ncells*(Ncells-1); % Number of links
pth = alpha*(1:Nl)/(Nl*log(Nl));

GCstat = zeros(size(Dmltotal));
% Compute p-values
Pv = chi2cdf(Dmltotal,mopt,'upper');

for t = 1:size(Pv,1)
    Pvt = squeeze(Pv(t,:,:));
    [Pvtsort, Indsrt] = sort(Pvt(:));
    Pvtsortx = Pvtsort(1:Nl);
    cnt = 1;
    while Pvtsortx(cnt) < pth(cnt)
        cnt = cnt+1;
    end
    siglind = Indsrt(1:cnt-1);
    cols = ceil(siglind/Ncells);
    rows = siglind - (cols-1)*Ncells;
    for ee = 1:numel(siglind)
        GCstat(t,rows(ee),cols(ee)) = Dmltotal(t,rows(ee),cols(ee));
    end
end

CM1 = zeros(size(GCstat));
for kk = 1:size(GCstat,1)
    GCk = squeeze(GCstat(kk,:,:));
    [ii,jj] = find(GCk);
    Sgn = zeros(size(GCk));
    for ll = 1:numel(ii)
        whati = wmltotal{ii(ll)}; 
        Sgn(ii(ll),jj(ll)) = sign(sum(whati(kk,(jj(ll)-1)*mopt+2:jj(ll)*mopt+1).*Whcml));
    end
    CM1(kk,:,:) = Sgn.*GCk;
end

% Connectivity Analysis based on modified Okatan approach
Kw = ceil(T/Wml);
CM2 = zeros(Ncells,Ncells,Kw);
for ct = 1:Ncells
    for kk = 1:Kw
        Hi = Hesstotal{ct}(:,:,kk);
        sigij = 1.96*sqrt(diag(inv(Hi))); 
        whatij = wmltotal{ct}(kk,2:end);
        whu = whatij + sigij(2:end)'; 
        whl = whatij - sigij(2:end)';
        CM2(ct,:,kk) = sum(reshape((whu.*whl > 0).*whatij,[mopt,Ncells]));
    end
end
% Zero out the diagonals self effects
for kk = 1:Kw
    CM2(:,:,kk) = CM2(:,:,kk) - diag(diag(CM2(:,:,kk)));
end

