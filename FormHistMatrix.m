function Xz = FormHistMatrix(x,Whc)

% This function forms the covariate (design) matrix corresponding to the
% history dependencies among neurons. Each row Xz(k,:,c) corresponds to the
% covariate vectors for neuron (c) at time step k, which is formed based on 
% the collection of spike counts of neuron's activity history within the 
% windows of history kernel Whc
% [Note: this function can be used for both the self- & cross-history ]

% Inputs:
% Whc: Cross-history kernel containing the size of spike counting windows
% x: input array of size [T x Ncells] as spike actvities of neuronal 
% ensemble, where T is total number of samples/bins, and Ncells is number 
% of neurons

% Outputs: 
% Xz: the history covariate matrix of size [T x Mhc x Ncells]


[T,Ncells] = size(x);
Mhc = numel(Whc); Lhc = sum(Whc);
next = [zeros(Lhc,Ncells);x];

X = zeros(T,Mhc,Ncells);
Lhself1 = zeros(Mhc,1); Lhself2 = Lhself1;
for j = 1:Mhc
    Lhself1(j) = sum(Whc(Mhc-j+2:Mhc));
    Lhself2(j) = sum(Whc(Mhc-j+1:Mhc));
end

% Spiking History Components WINDOW-BASED
for k = 1:T
    for cc = 1:Ncells
        for j = 1:Mhc
            X(k,Mhc-j+1,cc) = sum(next(k+Lhself1(j):k+Lhself2(j)-1,cc));
        end 
    end
end

% Zero-mean History Components
Xz = X;
for cc = 1:Ncells
    Xz(:,:,cc) = X(:,:,cc) - ones(size(X,1),1)*mean(X(:,:,cc)); % Corrected mean on April 4th for window-based history
end

end
