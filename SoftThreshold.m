function y = SoftThreshold(x,gamma)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
y = sign(x).*max(abs(x)-gamma,0);
end

