function [AGCs,AGC] = FDRcontrolBY(Devk,nuk,alpha,Md,wfk,Whc)

% FDR control procedure based on the Benjamini-Yekutieli(BY) rejection rule 
% and test strength characterization using the Youden’s J-statistics measure

% Main function: Address the problem of false discovery in the multiple 
% hypothesis testing framework for detecting the significant G-causal links
% in a neuronal ensemble, and test the significance level of rejected nulls
% (or in other words detected GC links)

% Inputs: 
% Devk: the input matrix of deviance difference samples of size
% [K x Ncells x Ncells] for all possible G-causal links among neurons of an
% ensemble
% nuk: the matrix of non-centrality parameters of same size as Devk for all
% GC links
% alpha: the false discovery rate (FDR)
% Md: the dimensionality difference (of full and reduced GLM models)
% wfk: the cell of size Ncells, contains the estimated parameters of full
% model at all times
% Whc: the cross-history kernel

% Outputs: 
% AGCs: the signed AGC maps of size [K x Ncells x Ncells] based on the 
% J-statistics
% AGC: the AGC maps based on the J-statistics (no signs)


[N,Ncells,~] = size(Devk);
Mhc = numel(Whc);
[~,Mf] = size(wfk{1}); Mhcs = Mf - (Ncells-1)*Mhc - 1;

Nl = Ncells*(Ncells-1); % Number of links/OR multiple tests
pth = alpha*(1:Nl)/(Nl*log(Nl)); 
alpham = alpha*(Nl+1)/(2*Nl*log(Nl)); % mean FDR by BY rule

AGC = zeros(size(Devk)); AGCs = AGC;
% Compute p-values
Pv = chi2cdf(Devk,Md,'upper');

for t = 1:N
    Pvt = squeeze(Pv(t,:,:));
    [Pvtsort, Indsrt] = sort(Pvt(:));
    Pvtsortx = Pvtsort(1:Nl);
    cnt = 0;
    while Pvtsortx(cnt+1) < pth(cnt+1)
        cnt = cnt+1;
        if cnt == Nl
            break
        end
    end
    siglind = Indsrt(1:cnt);
    cols = ceil(siglind/Ncells);
    rows = siglind - (cols-1)*Ncells;
    for ee = 1:numel(siglind)
        AGC(t,rows(ee),cols(ee)) = 1 - alpham - ncx2cdf( chi2inv(1-alpham,Md),Md, nuk(t,rows(ee),cols(ee)) );
        whati = wfk{rows(ee)};
        CellC = 1:Ncells; CellC(rows(ee))=[];
        jji = find(CellC == cols(ee));
        if sign(whati(t,(jji-1)*Mhc+Mhcs+2:jji*Mhc+Mhcs+1)*Whc') ~=0
            AGCs(t,rows(ee),cols(ee)) = sign(whati(t,(jji-1)*Mhc+Mhcs+2:jji*Mhc+Mhcs+1)*Whc')*AGC(t,rows(ee),cols(ee)); % WhOpt{rows(ee)}
        else
            AGCs(t,rows(ee),cols(ee)) = AGC(t,rows(ee),cols(ee)); % No sgn change if over regularized
        end
    end
end

