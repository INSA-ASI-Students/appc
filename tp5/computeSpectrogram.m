function [V,Phi] = computeSpectrogram(x,fs);
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
FFTSIZE = 1024;
HOPSIZE = 256;
WINDOWSIZE = 512;

X = myspectrogram(x,FFTSIZE,fs,hann(WINDOWSIZE),-HOPSIZE);
V = abs(X(1:(FFTSIZE/2+1),:));
F = size(V,1);
T = size(V,2);

Phi = angle(X);

end

