function [what,Devk,uk,vk,Uk] = L1PPF1ModDev(X,n,ff,gamma,LL,cw,W,w0,u0,v0,U0)

% L1-PPF-1 (L1-regularized Point Process Filter of order 1) Algorithm
% Sparse adaptive filter for neural spike data
% Developed for adaptive estimation of time-varying and sparse modulation
% parameters associated with a neuron from sequence of binary spike
% observations

% This function is a modified version of L1PPF1, in a window-based format
% and computes the adaptive de-biased deviance statistic in a recursive
% fashion meanwhile the adaptive L1PPF1 filtering

% Inputs:
% X: the input covariate matrix corresponding to the model (Full or Reduced)
% n: the observation vector (in form of a binary neural sequence) 
% w0: initial estimate of the parameter vector
% u0,v0,U0: initials for recursive variables effective in computation of
% deviance

% Outputs: 
% what: the array of estimated modulation parameter vectors of size [K x M]
% where the k-th row is associated with the time-varying w_k at time window
% k, and M denotes the length of w_k's, and K is number of window segments
% Devk: the adaptive de-biased deviance vector of size K
% uk,vk,Uk: the final values for the recursive variables 

% Parameters:
% ff: forgetting factor
% gamma : regularization parameter for sparsity
% LL: number of L1PPF1 iterations per step
% cw: step size constant
% W: filtering window segment size

[N,M] = size(X);
K = ceil(N/W);
what = zeros(K,M);
Devk = zeros(K,1);
Bk = Devk;
wl = w0; 
uk = u0; vk = v0; Uk = U0;
Pk = 0.001*eye(M);
Neff = W/(1-ff);
navg = mean(n);
% Effective regularization parameter scaled with a factor obtained from the
% asymptotic results
gammaEff = gamma*sqrt(Neff*log(M)*navg*(1-navg));

for k = 1:K
    Xk = X((k-1)*W+1:k*W,:);
    nk = n((k-1)*W+1:k*W);
    for l = 1:LL
        % Refine the log-rate estimates in shrinkage step
        ykl = Xk*wl;
        lambdakl = exp(ykl)./(1+exp(ykl));
        kappakl = diag(lambdakl.*(1-lambdakl));
        ekl = nk - lambdakl;
        % Recursive update rules for thetak estimation
        ukl = ff*uk + lambdakl'*ykl - sum(log(1+exp(ykl))) - (1/2)*ykl'*kappakl*ykl;
        vkl = ff*vk + Xk'*(ekl + kappakl*ykl);
        Ukl = ff*Uk + Xk'*kappakl*Xk;       
        al = 1/(cw*Neff); % step size
        gl = vkl - Ukl*wl; % recursive gradient
        % Proximal gradient update for L1-norm
        wl = SoftThreshold(wl + al*gl ,gammaEff*al); 
    end
    what(k,:) = wl';
    % Compute inverse Hessian using Woodbury matrix identity
    Yk = Pk*Xk';
    Pk = (Pk - (Yk/(ff*eye(W)/kappakl + Xk*Yk))*Yk')/ff;
    % Compue Deviance 
    Bk(k) =  gl'*Pk*gl; % a quadratic Bias term
    Devk(k) = 2*(ukl + vkl'*wl -(1/2)*wl'*Ukl*wl ) + Bk(k); 
    uk = ukl; vk = vkl; Uk = Ukl;
end

end

