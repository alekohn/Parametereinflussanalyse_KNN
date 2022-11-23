close all;
clear all;
clc;
delete(gcp('nocreate'))

%% Konfiguration
TrainDataFile = 'Trainingsdaten.xlsx';

ErgDataFile1 = 'SoCErg1.xlsx';
VarDataFile1 = 'SoCErg1.mat';
ErgDataFile2 = 'SoCErg2.xlsx';
VarDataFile2 = 'SoCErg2.mat';

%% Versuchsvorbereitung
% Full factorial design
% Hier den Versuchsumfang eingeben und Run drücken. Danach das Skript: Auswertung! 
dFF1 = fullfact([2 2]);  % Ermittlung von [Inputdelay und Feedbackdelay]   
dFF2 = fullfact([5 1]);  % Ermittlung von [Neuronen und Layern]  

%% Einlesen Trainingsdaten
tmp = readmatrix(TrainDataFile);       % Trainingsdaten
tmp = downsample(tmp,1);               % Reduzierung der Trainingsdaten 
I = tmp(:, 1);                   % Strom in Ampere 
U = tmp(:, 2);                   % Spannung in Volt
SoC = tmp(:, 3);                 % Ladezustand in Prozent

clear tmp;

%% Vorbereiten Trainingsdaten
% Konvertiert Daten in die Standardform eines neuronalen Netzwerkzellenarrays
% [y,wasMatrix] = tonndata (x, columnSamples, cellTime) verwendet diese Argumente:
% columnSamples-> True, wenn Original-Samples als Spalten ausgerichtet sind, false, wenn Zeilen
% cellTiTime-> True, wenn Original-Samples Spalten eines Zellen-Arrays sind, false, wenn sie in einer Matrix gespeichert sind          
X = tonndata([I,U],false,false);        
Y = tonndata(SoC,false,false); 

% Anzahl der Versuche 
tmp1 = size(dFF1);
numTests1 = tmp1(1);
tmp2 = size(dFF2);
numTests2 = tmp2(1);

% Trainingsdurchläufe für die Bestimmmung von ID und FD
for n = 1:numTests1
    ID = dFF1(n,1)-1;
    FD = dFF1(n,1);
    N = 1;
    H = 1;
    Trys = 1;         %Anzahl der Trainingsversuche
    
    disp(sprintf('Versuch %d:\tID: %d, FD: %d, N: %d', n, ID, FD, N));
    [trainmax, netc] = Narx_Training(ID, FD, N, H, X, Y, Trys);
    
    ErgTrain1(n) = trainmax;          % bestes Ergebnis aus Trainingsversuchen
    net1{n} = netc;                   % trainiertes NN
    disp(sprintf('\t\tErgebnis: R2: %d' ,trainmax(1)));
end

[minV1, minI1] = max(ErgTrain1(:,1));

disp(sprintf('\nVersuchsreihe 1 abgeschlossen!\nBeste Ergebnisse(Training1):'));
disp(sprintf('Versuch: %d ---> R2: %g mit ID: %d, FD: %d',minI1(1), minV1(1),dFF1(minI1(1),1), dFF1(minI1(1),2)));

writematrix(ErgTrain1', ErgDataFile1);
save(VarDataFile1, 'ErgTrain1', 'net1', 'dFF1', 'minI1');

% Trainingsdurchläufe für die Bestimmmung von N und H
for n = 1:numTests2
    ID = dFF1(minI1(1),1)-1;
    FD = dFF1(minI1(1),2);
    N = dFF2(n,1);
    H = dFF2(n,2);
    Trys = 1;        
    
    disp(sprintf('Versuch %d:\tID: %d, FD: %d, N: %d H: %d', n, ID, FD, N, H));
    [trainmax, netc] = Narx_Training(ID, FD, N, H, X, Y, Trys);
    
    ErgTrain2(n) = trainmax;         
    net2{n} = netc;                   
    disp(sprintf('\t\tErgebnis: R2: %d' ,trainmax(1)));  
end

[minV2, minI2] = max(ErgTrain2(:,1));

disp(sprintf('\nVersuchsreihe 1 abgeschlossen!\nBeste Ergebnisse(Trainingn):'));
disp(sprintf('Versuch: %d ---> R2: %g mit ID: %d, FD: %d, N: %d, H: %d',minI2(1), minV2(1),dFF1(minI2(1),1)-1, dFF1(minI2(1),2), dFF2(minI2(1),1), dFF2(minI2(1),2)));

Ergebnis = [minI2(1) minV2(1) dFF1(minI2(1),1)-1 dFF2(minI2(1),2) dFF2(minI2(1),1) dFF2(minI2(1),2)]; % [Versuch; R2; ID; FD; N; H] 

%% Ergebnisse speichern
writematrix(ErgTrain2, ErgDataFile2);
save(VarDataFile2, 'ErgTrain2', 'net2', 'Ergebnis', 'dFF2', 'minI2');

