function [resp] = GenerateSpikes(thetac,mu,Cmap)

% Generate the binary spike trains corresponding to an ensemble of
% interacting neurons, inter-connected according to the dynamic causal map
% structure specified by the input "Cmap"

% Inputs: 
% thetac: a single modulation parameter vector used to model all the 
% G-causal interactions
% mu: scalar baseline firing parameter
% Cmap: a causality map matrix of size [Ncells,Ncells,T] containing the
% scaling and signs (nature) of the underlying causal structure among 
% neurons, where Ncells is the number of neurons, and T is the number of 
% time bins/samples

% Outputs
% resp: response matrix of size [TxNcells], which contains the binary spike
% trains for all neurons as columns

Mc = numel(thetac);
[Ncells,~,T] = size(Cmap);

%%% Generate Spike trains for ensembe of Ncells neurons
next = zeros(T+Mc,Ncells); 
ld = zeros(T,Ncells); % Spiking probability (ld = lambda*delta)
lHist = zeros(T,Ncells);
% n_k as Binary Sequence with Conditionaly independent Bernoulli Statistics
for k = 1:T
    for cc = 1:Ncells        
        idx = find(Cmap(cc,:,k));
        lgc = zeros(numel(idx),1);
        for ee = idx
            lgc(ee) = Cmap(cc,ee,k)*thetac'*next(k+Mc-1:-1:k,ee);
        end
        lHist(k,cc) = sum(lgc); % superposition of self- & cross-history effects
        lc = mu + lHist(k,cc);
        % Logistic link model for CIF
        ld(k,cc) = exp(lc)/(1+exp(lc)); % Spiking probability (lambda.Delta)
        % Spike generation as conditional Bernoulli process 
        next(k+Mc,cc) = (rand(1) < ld(k,cc));
    end
end
resp = next(Mc+1:T+Mc,:); % Spiking responses of the ensemble of neurons
