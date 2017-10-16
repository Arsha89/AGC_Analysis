%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Performance Comparison: Estimate Causal maps using the alternative methods
%%% This code Must be run after running the main mfile "SimGCanalysisFinal.m"
tic
Wml = round(W/(1-ff)); % Equiv. to Neff = W/(1-ff) of adaptive setting 
mopt = numel(thetac)/10; % Optimally selected model order
[GC1,GC2,Dmltotal,wmltotal] = GCmethods(resp,Wml,mopt,WH,alpha);
toc

%%% Plot the estimated GC maps obstained by the two alternative methods [for comparions purposes]
Tsnaps = [20,30,40,60,70,80,100,110,120]; % time instances (sec)
Nsnp = numel(Tsnaps);
hfE2 = figure;
cmax = 1; cmin = -cmax;
for k = 1:Nsnp
    Phik1 = squeeze(GC1(Tsnaps(k)/(delta*Wml),:,:));
    Phik2 = squeeze(GC2(Tsnaps(k)/(delta*Wml),:,:));
    subplot(2,Nsnp,k), imagesc(Phik1), caxis([cmin,cmax]);
    subplot(2,Nsnp,Nsnp+k), imagesc(Phik2), caxis([cmin,cmax]);    
end
colormap jet
set(hfE2,'position',[100,100,1700,400])
suptitle(['Performance comparison of the causal inference methods @ FDR = ' num2str(alpha)])
