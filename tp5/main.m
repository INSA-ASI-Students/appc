% TP 05 - APPC
% Florian Martin
% Thibault Théologien

addpath('../utils');
addpath('./lib');
clean_env();

%% 1. Load the signal that we want to decompose and listen to it
[x fs] = audioread('./Mary.wav');
% sound(x, fs);

%% 2. compute its spectogram.
FFTSIZE = 1024;
[V, phi] = computeSpectrogram(x,fs);

% a) Plot the obtained time-frequency representation
F = size(V, 1);
T = size(V, 2);
imagesc(db(V))
set(gca, 'YDir', 'normal')
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
title('Spectrogram of Mary Had a Little Lamb');
ylabel('Frequency');
xlabel('Time');


% b)
K = 3;
nbiter = 200;
[dim, nbsig ] = size (V);
W = 1 + rand (dim, K);
H = 1 + rand (K, nbsig);
for i =1: nbiter
  pas =1/ norm (W' * W);
  for j = 1:20
    H = H + pas * W' * (V - W * H);
    H = H .* (H > 0);
  end

  pas = 1 / norm (H * H');
  for j = 1:20
    W = W+ pas *(V - W * H) * H';
    W =  W .* (W > 0);
  end
end
