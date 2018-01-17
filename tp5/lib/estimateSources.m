function [ xhat1 ] = estimateSource(W,H,Phi)

%
%
%

HOPSIZE = 256; % should match the one in computeSpectrogram
numberofSources=size(W,2);

for i=1:numberofSources
    
    XmagHat = W(:,i)*H(i,:);
    
    % create upper half of frequency before istft
    XmagHat = [XmagHat; conj( XmagHat(end-1:-1:2,:))];
   
    % Multiply with phase
    XHat = XmagHat.*exp(1i*Phi);
    
    xhat1(:,i) = real(invmyspectrogram(XHat,HOPSIZE))';
    


end

