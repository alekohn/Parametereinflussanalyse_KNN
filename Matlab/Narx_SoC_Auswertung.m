clear all;
close all;
clc;


%% NARX_Auswertung
load('SoCErg2')
[V, Imin] = min(ErgTrain2(:,1));
[V1, Imax] = max(ErgTrain2(:,1)); 
V2 = median(ErgTrain2(:,1))-min(abs(ErgTrain2(:,1)-median(ErgTrain2(:,1))));
Imit = find(abs(ErgTrain2(:,1) - V2) < 0.00001);
net = {net2{Imax, 1}{1, 1} net2{Imin, 1}{1, 1} net2{Imit, 1}{1, 1}};

%% Trainingsdaten
% Einlesen Trainingsdaten
TrainDataFile = 'Trainingsdaten.xlsx';

tmp = readmatrix(TrainDataFile);
tmp = downsample(tmp,1);
I = tmp(:, 1);
U = tmp(:, 2);
SoC = tmp(:, 3);

clear tmp;

% Vorbereiten Trainingsdaten        
input = tonndata([I,U],false,false);        
target = tonndata(SoC,false,false);

% Anzahl der Neuronen und Hidden-Layer aus Training für die Plots
N = [dFF2(Imax,1) dFF2(Imin,1) dFF2(Imit,1)]' 
H = [dFF2(Imax,2) dFF2(Imin,2) dFF2(Imit,2)]'

% Vergleich zwischen Sollwert und CL-Vorhersage
figure(1)
for n = 1:3
set(gcf,'color','w','OuterPosition',[533 133 1138 697]);    
netF = net{1, n};
[Xcs,Xci,Aci,Tcs] = preparets(netF,input,{},target);
subplot(3,1,n) 
y = netF(Xcs,Xci,Aci);
perf = perform(netF,Tcs,y);
R2 = power(corrcoef(cell2mat(y),cell2mat(Tcs)),2);
plot(cell2mat(y'))
hold on
plot(cell2mat(Tcs),'r');
hold off
xlim([0 length(cell2mat(y))]);
grid on 
grid minor
sgtitle('Vorhersage mit Trainingdaten im Closeloop','FontSize',16)
xlabel('Zeit in s','FontSize',14);
ylabel('SoC in %','FontSize',14);
legend('Prediction','Target');
title([ 'R^2 = ' ,num2str(R2(2,1)), ' (H=',num2str(H(n,1)), ', N=',num2str(N(n,1)),')'],'FontSize',14);
end

% Fehler zwischen Sollwert und CL-Vorhersage
figure(2)
for n=1:3
set(gcf,'color','w','OuterPosition',[533 133 1138 697]);
netF = net{1, n};
[Xcs,Xci,Aci,Tcs] = preparets(netF,input,{},target);
subplot(3,1,n)
y = netF(Xcs,Xci,Aci);
perf = perform(netF,Tcs,y);
R2 = power(corrcoef(cell2mat(y),cell2mat(Tcs)),2);
plot(cell2mat(gsubtract(Tcs,y)));
xlim([0 length(Tcs)])
grid on
grid minor
sgtitle('Fehler zwischen Vorhersage und Sollwert(Training)','FontSize',16)
ylabel('Error in %','FontSize',14);
xlabel('Zeit in s','FontSize',14);
title(['(H=',num2str(H(n,1)), ', N=',num2str(N(n,1)),')'],'FontSize',14);
end

% Trainingsergebnisse als Zahlenwerte
for n = 1:size(dFF2) 
netF = net2{1, n}{1, 1};  
[Xcs,Xci,Aci,Tcs] = preparets(netF,input,{},target);
y = netF(Xcs,Xci,Aci);
perf = perform(netF,Tcs,y);
R2 = power(corrcoef(cell2mat(y),cell2mat(Tcs)),2);        
ErgTrain(n,1) = R2(2,1);      
end
writematrix(ErgTrain, 'ErgTrain.xlsx');

%% Testdaten
% Einlesen Testdaten
TestDataFile = 'Testdaten_WLTP.xlsx';
tmp= readmatrix(TestDataFile);
tmp = downsample(tmp,1);
I1 = tmp(:, 1);
U1 = tmp(:, 2);
SoC1 = tmp(:, 3);

clear tmp

inputtest = tonndata([I1,U1],false,false);
targettest = tonndata(SoC1,false,false);

% Vergleich zwischen Sollwert und CL-Vorhersage
figure(3)
for n = 1:3
set(gcf,'color','w','OuterPosition',[533 133 1138 697]);
netF = net{1, n};
[Xcs,Xci,Aci,Tcs] = preparets(netF,inputtest,{},targettest);
subplot(3,1,n)
y = netF(Xcs,Xci,Aci);
perf = perform(netF,Tcs,y);
R2 = power(corrcoef(cell2mat(y),cell2mat(Tcs)),2);
plot(cell2mat(y'))
hold on
plot(cell2mat(targettest),'r');
hold off
xlim([0 length(cell2mat(y))]);
grid on
grid minor
sgtitle('Vorhersage mit Testdaten im Closeloop','FontSize',16)
ylabel('SoC in %','FontSize',14);
xlabel('Zeit in s','FontSize',14);
legend('Prediction','Target');
title([ 'R^2 = ' ,num2str(R2(2,1)), ' (H=',num2str(H(n,1)), ', N=',num2str(N(n,1)),')'],'FontSize',14);
ErgTest(n,1) = R2(2,1);      
end
writematrix(ErgTest, 'ErgTest.xlsx');


% Fehler zwischen Sollwert und CL-Vorhersage
figure(4)
for n=1:3
set(gcf,'color','w','OuterPosition',[533 133 1138 697]);
netF = net{1, n};
[Xcs,Xci,Aci,Tcs] = preparets(netF,inputtest,{},targettest);
subplot(3,1,n)
y = netF(Xcs,Xci,Aci);
perf = perform(netF,Tcs,y);
R2 = power(corrcoef(cell2mat(y),cell2mat(Tcs)),2);
plot(cell2mat(gsubtract(Tcs,y)));
xlim([0 length(Tcs)])
grid on
grid minor
sgtitle('Fehler zwischen Vorhersage und Sollwert(Test)','FontSize',16)
ylabel('Error in %','FontSize',14);
xlabel('Zeit in s','FontSize',14);
title(['(H=',num2str(H(n,1)), ', N=',num2str(N(n,1)),')'],'FontSize',14);
end

% Testergebnisse als Zahlenwerte
for n = 1:size(dFF2) 
netF = net2{1, n}{1, 1};
[Xcs,Xci,Aci,Tcs] = preparets(netF,inputtest,{},targettest);
y = netF(Xcs,Xci,Aci);
perf = perform(netF,Tcs,y);
R2 = power(corrcoef(cell2mat(y),cell2mat(Tcs)),2);        
ErgTest(n,1) = R2(2,1);      
end
writematrix(ErgTest, 'ErgTest.xlsx');
