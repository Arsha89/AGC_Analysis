function Xfn = FormFullDesign(Xcross,Xself,ct) 

% This function construct the full covariate (design) matrix associated
% with the full GLM model for neuron (ct), integrating the effects of
% self-history and all the cross-history from other neurons.  
% Note: the output full matrix is standardized (zero-mean and normalized
% columns)

% Inputs:
% Xcross: the cross-history covariate matrix 
% Xself: the self-history covariate matrix 
% ct: index of target neuron, for which we are forming full matrix

% Outputs: 
% Xfn: the full covariate matrix of size [T x Mf], where Mf is the
% dimension of the full GLM model

[T,Mhc,Ncells] = size(Xcross);
[~,Mhcs] = size(Xself);
Mf = (Ncells-1)*Mhc+Mhcs+1;
cellC = 1:Ncells; cellC(ct) = [];

Xf = zeros(T,Mf);
Xf(:,1) = ones(T,1);
Xf(:,2:Mhcs+1) = Xself;
for cc = 1:Ncells-1
    Xf(:,(cc-1)*Mhc+Mhcs+2:cc*Mhc+Mhcs+1) = Xcross(:,:,cellC(cc));
end

% Standardize the covariate matrix
varXf = var(Xf); varXf(1) = 1;
Dfn = diag(sqrt(varXf));
Xfn = Xf/Dfn;