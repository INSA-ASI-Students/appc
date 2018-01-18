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
imagesc(db(V));
set(gca, 'YDir', 'normal');
set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
title('Spectrogram of Mary Had a Little Lamb');
ylabel('Frequency');
xlabel('Time');

% On peut voir les différentes notes représentées au cours du temps
% sur ce spectrogramme


% b) Proximal gradient descent
K = 3;
nbiter = 200;
[dim, nbsig ] = size (V);
W = 1 + rand (dim, K);
H = 1 + rand (K, nbsig);

for i =1: nbiter
  % On cherche à détecter les moments où les différentes notes sont jouées
  % et avec quelle amplitude
  pas =1/ norm (W' * W);
  for j = 1:20
    H = H + pas * W' * (V - W * H);
    H = H .* (H > 0);
  end

  % On cherche à décoréller les signaux des différentes notes
  pas = 1 / norm (H * H');
  for j = 1:20
    W = W + pas *(V - W * H) * H';
    W =  W .* (W > 0);
  end
end

% c) Affichage des vecteurs de base
freq = linspace(0, fs / 2, FFTSIZE / 2 + 1);
time = linspace(0, length(x) / fs, T);
figure();
% On affiche les signaux des trois notes décoréllées
for i = 1:K
    plot((i - 1) * max(max(W)) + (1 - W(:, i)), freq, 'LineWidth', 3);
    hold on;
end

title('Basis Vectors')
ylabel('Frequency ( Hz )')
xlabel('Basis')
set(gca , 'XTickLabelMode', 'manual' , 'XTickLabel', []) ;

% d) Affichage des vecteurs d'activation
figure();
for i = 1:K
    plot(time, (i - 1) * max(max(H)) + (H(i, :)), 'LineWidth', 3);
    hold on;
end
ylabel('Activations');
xlabel('Time ( seconds )');
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);

% e) Qu'arrive t'il si l'on fait varier K ?
% Si K diminue, les signaux seront moins décomposés
% et inversement, si K augmente, certains signaux seront plus
% décomposés. Dans notre fichier audio, nous n'avons que 3 notes
% et il semblerait que nous les obtenions toutes correctement
% lorsque K = 3. Si K est différent de 3, certains signaux ne
% représenteront plus une note dans son intégralité, mais le
% fichier audio peut tout de même être restitué grâce aux signaux 
% d'activation.

%% 3. Reconstruction du signal
[xhat] = estimateSources(W, H, phi);
sound(xhat(: ,1), fs)
sound(xhat(: ,2), fs)
sound(xhat(: ,3), fs)

%% 4. Improve source separation
% a)
lambda = 300;
W = 1 + rand(dim, K);
H = 1 + rand(K, nbsig);
for i = 1:nbiter
    pas = 1 / norm (W' * W);
    for j =1:100
        H = H + pas * W' * (V - W * H);
        H = sign(H) .* max(abs(H) - pas * lambda, 0);
        H = H .* (H > 0);
    end
    
    pas = 1 / norm(H * H');
    for j = 1:100
        W = W + pas * (V - W * H) * H';
        W = W .* (W > 0);
        norm_w = sqrt(sum(W.^2)) ;
        for ii =1: K
            if norm_w(ii) > 10
                W(:, ii) = W(:, ii) / norm_w(ii) * 10;
            end
        end
    end
end

% b)
figure();
for i =1: K
    plot(time, (i - 1) * max(max(H)) + (H(i, :)), 'LineWidth', 3);
    hold on;
end
ylabel('Activations');
xlabel('Time ( seconds )');
set(gca, 'YTickLabelMode' , 'manual', 'YTickLabel', []);

% c)
[xhat] = estimateSources(W, H, phi);
sound(xhat(: ,1), fs)
sound(xhat(: ,2), fs)
sound(xhat(: ,3), fs)