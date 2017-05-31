function [nukSmth,nukSmthU,nukSmthL,nukFilt,nukFiltU,nukFiltL] = NoncentChi2FiltSmooth(Devk,Md,sz0,rho,NN,Nem)

% Non-central Chi2 Filtering and Smoothing Algorithm
% Established based on the asymptotic distributional inference of the
% deviance difference statistics for two nested GLM models (Theorem 1)
% Main Function: Estimate the non-centrality parameter \nu_k from the 
% observed values of the deviance difference statistics D_k for all times k

% Inputs: 
% Devk: the input matrix of deviance difference samples of size
% [K x Ncells x Ncells] for all possible G-causal links among neurons of an
% ensemble
% Md: the dimensionality difference (of full and reduced GLM models)

% Parameters: 
% sz0: input initial variance of zk, serves as SMOOTHING FACTOR
% rho: scaling factor (often chosen in range of [ff,1] )
% NN: number of Newtons iterations (often 7 to 20 is enough)
% Nem: number of EM iterations 

% Outputs: 
% nukSmth: the output vector of "smoothed" estimates of non-centrality nuk
% nukSmthU,nukSmthL: the Upper & Lower bounds of confidence regions for the
% "nukSmth" estimates
% nukFilt: the output vector of "filtered" estimates of non-centrality nuk
% nukFiltU,nukFiltL: the Upper & Lower bounds of confidence regions for the
% "nukFilt" estimates

% Main variables: 
% nuk: the non-centrality parameter 
% Devk: the deviance difference 
% zk: the state variable 
% s2k: the state variance of zk

[K,Ncells,~] = size(Devk);
nukFilt = zeros(K,Ncells,Ncells);
nukFiltU = zeros(K,Ncells,Ncells); nukFiltL = zeros(K,Ncells,Ncells);
nukSmth = zeros(K,Ncells,Ncells);
nukSmthU = zeros(K,Ncells,Ncells); nukSmthL = zeros(K,Ncells,Ncells);
eta = norminv(0.975); % the 95% confidence intervals

for ct = 1:Ncells
    cellC = 1:Ncells; cellC(ct) = [];
    for cf = cellC
        Dnk = Devk(:,ct,cf);
        zk = zeros(K,1); s2k = zeros(K,1); 
        nuk = zeros(K,1); nukU = zeros(K,1); nukL = zeros(K,1);
        sz = sz0;

        % EM Algorithm on sz parameter
        for ll = 1:Nem
            zk(1) = 0; 
            s2k(1) = 1;
            for k = 1:K
                zl = rho*zk(k);
                for l = 1:NN
                    nul = exp(zl);
                    ul2 = nul*max(Dnk(k),0);
                    fl = sqrt(ul2 + (Md-1)^2/4) - (Md-1)/2;
                    gl = zl - rho*zk(k) + (rho^2*s2k(k)+sz)*(nul - fl)/2;
                    gpl = 1 + (rho^2*s2k(k)+sz)*(nul - ul2/(2*(fl+(Md-1)/2)))/2;
                    zl = zl - gl/gpl;
                end
                zk(k+1) = zl;
                nuk(k) = exp(zk(k+1));
                uzk = sqrt(nuk(k)*max(Dnk(k),0));
                fuk = sqrt(uzk^2 + (Md-1)^2/4) - (Md-1)/2;
                s2k(k+1) = 1/( 1/(rho^2*s2k(k) + sz) + nuk(k)/2 - (uzk^2 - (Md-2)*fuk - fuk^2)/4 );
                nukU(k+1) = exp(zk(k+1)+ eta*sqrt(s2k(k+1)));
                nukL(k+1) = exp(zk(k+1)- eta*sqrt(s2k(k+1)));
            end

            %%% Smoothing the Deviance Samples
            zkK = zk; s2kK = s2k;
            s2kp1K = zeros(K,1);
            for k = K:-1:1
                sk = rho*s2k(k+1)/(rho^2*s2k(k+1)+ sz);
                zkK(k) = zk(k+1)+ sk*(zkK(k+1) - rho*zk(k+1));
                s2kK(k) = s2k(k+1)+ (sk^2)*(s2kK(k+1) - (rho^2*s2k(k+1)+sz));
                s2kp1K(k) = sk*s2kK(k+1);
            end
            
            nukKU = nukU; nukKL = nukL;
            for k = 1:K+1
                nukKU(k) = exp(zkK(k)+ eta*sqrt(s2kK(k)));
                nukKL(k) = exp(zkK(k)- eta*sqrt(s2kK(k)));
            end

            alpha = -1; beta = 0;
            % EM update rule for sz based on smoothed values
            Eterm = rho^2*zkK(1)^2 + (1+rho^2)*sum(zkK(2:K).^2)+ zkK(K+1).^2 + rho^2*s2kK(1)+ (1+rho^2)*sum(s2kK(2:K)) + s2kK(K+1) -2*rho*(sum(s2kp1K)+ sum(zkK(2:K+1).*zkK(1:K)));
            szp =  (Eterm/2+beta)/(K/2+alpha+1);
            sz = szp;
        end
        nukFilt(:,ct,cf) = nuk;
        nukFiltU(:,ct,cf) = nukU(2:end); nukFiltL(:,ct,cf) = nukL(2:end);
        nukSmth(:,ct,cf) = exp(zkK(2:end));
        nukSmthU(:,ct,cf) = nukKU(2:end); nukSmthL(:,ct,cf) = nukKL(2:end);
    end
end
