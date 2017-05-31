function [what,vk,Uk] = L1PPF1(X,n,ff,gamma,LL,cw,W,w0,v0,U0)

% Feb 25th 2016: Extend PGPPF2v2 to Window-based format 
% May 23rd 2016: Modify to BIAS Corrected version of Deviance
% Mar 3rd 2017: Modify Bias initialization of Pk and scaling

% gamma : sparsity parameter gamma
% tk : step size at each time step (k)
% primal variable sparse: betal
% dual variable: ul

[N,M] = size(X);
K = ceil(N/W);
what = zeros(K,M);
wl = w0;
%uk = u0; 
vk = v0; Uk = U0;
navg = mean(n);
gammaEff = gamma*sqrt(W*log(M)*navg*(1-navg)/(1-ff));
Lipk = 0;

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
        %ukl = ff*uk + lambdakl'*ykl - sum(log(1+exp(ykl))) - (1/2)*ykl'*kappakl*ykl;
        vkl = ff*vk + Xk'*(ekl + kappakl*ykl);
        Ukl = ff*Uk + Xk'*kappakl*Xk;
        
        Lipkl = ff*Lipk + sum(diag(kappakl*(Xk*Xk')));
        al = (1-ff)/(cw*W);% independent of M
        %al = (1-ff)/(cw*W*M*navg);%M*navg % dependent on M
        %al = 1/(cw*Lipkl); scheme 3 based on Lipschitz constant
        %varnk = mean(diag(kappakl));
        %gammaEffkl = gamma*sqrt(W*log(M)*varnk/(1-ff));
        gl = vkl - Ukl*wl;
        wl = SoftThreshold(wl + al*gl ,gammaEff*al);       
    end
    what(k,:) = wl';
    %uk = ukl; 
    vk = vkl; Uk = Ukl; Lipk = Lipkl;
end

end

