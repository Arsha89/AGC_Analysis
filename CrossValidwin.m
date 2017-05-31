function [gammaOpt] = CrossValidwin(Respx,gammacv,Whcv,Whscv,KK,ffcv,Wcv,ccv,LLcv)

% K-fold window-based Cross-Validation (CV)
% When K=2 is equivalent to two-fold even-odd Cross-validation
% Find the optimal data-driven gamma at certain input setup

% Inputs: 
% Respx: the input response for all cells subject to cross-validation
% gammacv: the vector of gamma's often chosen from (0,1)
% Whcv: cross-history kernel for CV
% Whcsv: self-history kernel for CV
% KK: number of CV folds

% Parameters: 
% ffcv: forgetting factor
% Wcv: window size 
% ccv: step size constant
% LLCv: number of L1PPF1 iterations per step

% Outputs: 
% gammaOpt: a vector of size Ncells containing the optimal regularization
% parameter for all neurons

[Nb, Ncells] = size(Respx);
LLcost = zeros(numel(Whcv),numel(gammacv),KK,Ncells);
for ii = 1:numel(Whcv)
    Whi = Whcv{ii}; Mc = numel(Whi); 
    Xc = FormHistMatrix(Respx,Whi);
    % Prepare Self-Hist Conditions
    Whsi = Whscv{ii}; Mhcs = numel(Whsi); 
    Mf = (Ncells-1)*Mc+Mhcs+1;
    
    for ct = 1:Ncells
        N_indto = num2str(ct);
        disp(['Cross-validation on cell ' N_indto '...'])        
        robs = Respx(:,ct);
        %%% Form SELF-History Covriates
        Xcs = FormHistMatrix(robs,Whsi);
        % Form FULL Model Cov matrix
        Xfn = FormFullDesign(Xc,Xcs,ct); 
        % Set initials for filtering/estimation
        navg = mean(robs);        
        w0f = zeros(Mf,1); w0f(1) = log(navg/(1-navg));
        v0f = zeros(Mf,1); U0f = 1*eye(Mf);
        tic
        for kk = 1:KK
            disp(['CV on fold ' num2str(kk) '...'])
            Indvalid = zeros(Wcv,Nb/(KK*Wcv));
            for j = 1:Nb/(KK*Wcv)
                Indvalid(:,j) = (KK*Wcv*(j-1)+(kk-1)*Wcv+1:KK*Wcv*(j-1)+(kk-1)*Wcv+Wcv)'; 
            end
            indvalid = Indvalid(:); 
            nvalid = robs(indvalid); Svalid = Xfn(indvalid,:);
            ntrain = robs; ntrain(indvalid) = [];
            Strain = Xfn; Strain(indvalid,:) = [];
            for jj = 1:numel(gammacv)
                gammaj = gammacv(jj);
                disp(['CV @ gamma = ' num2str(gammaj) '...'])
                % Training Stage: Filtering
                [wTrain,~,~] = L1PPF1(Strain,ntrain,ffcv^KK,gammaj,LLcv,ccv,Wcv,w0f,v0f,U0f);
                %%% Validation Stage
                for k = 1:numel(nvalid)
                    LLcost(ii,jj,kk,ct) = LLcost(ii,jj,kk,ct)+ nvalid(k)*(Svalid(k,:)*wTrain(ceil(k/Wcv),:)') - log(1+exp(Svalid(k,:)*wTrain(ceil(k/Wcv),:)'));
                end
            end
        end
        toc
    end
end
LLavg = squeeze(mean(LLcost,3));

gammaOpt = zeros(Ncells,1);
for ct = 1:Ncells
    LLct = LLavg(:,ct);
    [indgamma] = find(LLct == max(max(LLct)));
    gammaOpt(ct) = gammacv(max(indgamma));
end
