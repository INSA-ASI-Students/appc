function [f,g,h]=logisticreg(Z,y,w)



% 
% expyXw=exp(-y.* (X*w));
% f =  sum(log(1+expyXw));
% c = -y.* expyXw./(1+expyXw); 
% g =  sum( X.*repmat(c,1,size(X,2)),2);


n=length(y);
%Z = sparse(1:n,1:n,y,n,n)*X;
Zw = -Z*w; 
posind = (Zw > 0);
logist(posind,1) = 1 + exp(-Zw(posind));
logist(~posind,1) = 1 + exp(Zw(~posind));

if isempty(find(posind))
    AA=0;
else
    AA=sum(Zw(posind) + log(logist(posind)));
end;

f = sum(log(logist(~posind))) + AA; 


temp = logist;
temp(posind) = 1./logist(posind);
temp(~posind) = (logist(~posind)-1)./logist(~posind);
g =  -Z'*temp;

if nargout==3
h =[]; 
end;