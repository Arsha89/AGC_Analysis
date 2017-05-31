function [what,Dev,Hess] = MLEwinDev(X,n,LL,al,aa,W,w0)

% Sep 25th 2016: Extend PGPPF2v2 to Window-based format 

% W : window/segment size
% LL : # of Newtons iterations
% c : step size coefficient

[N,M] = size(X);
K = ceil(N/W);
what = zeros(K,M);
Dev = zeros(K,1);
wl = w0;
Hess = zeros(M,M,K);

for k = 1:K
    Xk = X((k-1)*W+1:k*W,:);
    nk = n((k-1)*W+1:k*W);
    for l = 1:LL
        % Refine the log-rate estimates in shrinkage step
        yl = Xk*wl;
        lambdal = exp(yl)./(1+exp(yl));
        kappal = diag(lambdal.*(1-lambdal));
        el = nk - lambdal;

        % Recursive update rules for thetak estimation
        Ll = nk'*yl - sum(log(1+exp(yl)));
        gl = Xk'*el; % Gradient
        Hl = aa*eye(M)+Xk'*kappal*Xk; % Hessian
    
        %al = 1/(c*W);
        % Relaxed Newton's iteration
        wl = wl + al*(Hl\gl);
    end
    what(k,:) = wl';
    Dev(k) = 2*Ll;   
    Hess(:,:,k) = Hl; % Estimate of Hessian
end

end

