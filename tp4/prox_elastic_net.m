function w = prox_elastic_net(w,lambda,alpha)
  w = (abs(w) - lambda * alpha * sign(w)) / (1 + 2 * lambda * (1 - alpha)); 
end
