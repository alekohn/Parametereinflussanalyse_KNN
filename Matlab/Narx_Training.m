function [Trainmin, netstruct] = Narx_Training(ID, FD, N, H, input, target, Trys)
%% Parameter Netz-Architektur
tic
    
    for n = 1:Trys          %Trainingsversuche
    switch H
        case 1
        hiddenLayerSize = N;                        %Hidden-Layer (hier 1 Layer) "hiddenLayerSize = [NHL1,NHL2](Bei 2 Hidden Layer);"
        net.layers{1}.transferFcn='tansig';         %Aktivierungsfunktion: Hidden-Layer                     
        net.layers{2}.transferFcn='purelin';        %Aktivierungsfunktion: Output-Layer
        case 2
        hiddenLayerSize = [N,N];
        net.layers{1}.transferFcn='tansig';
        net.layers{2}.transferFcn='tansig';
        net.layers{3}.transferFcn='purelin';
        case 3
        hiddenLayerSize = [N,N,N];
        net.layers{1}.transferFcn='tansig';
        net.layers{2}.transferFcn='tansig';
        net.layers{3}.transferFcn='tansig';
        net.layers{4}.transferFcn='purelin';
    end
    
    %% Initalisierung Narx-Netz
    % net = narxnet(ID,FD,hiddenLayerSize,FeedbackMode,trainFcn);          %Erstellung Narx-Netz
    trainFcn = 'trainlm';                                                  %Trainingsalgorithmus(Levenberg-Marquardt)
    net = feedforwardnet(hiddenLayerSize,trainFcn);                        %Initialisierung: FFN
    net.inputWeights{1,1}.delays = ID;                                     %Initialisierung: ID 
    net.inputs{1}.name = 'x';                                              %Initialisierung: Input     
    
    % Feedback Output
    net.outputs{net.numLayers}.name = 'y';                                 %Initialisierung Output(SoC) als Input für OL
    net.outputs{net.numLayers}.feedbackMode = 'open';                      %Rückkopplung wird ausgeschalten 
    net.inputConnect(1,2) = true;                                          %Erzeugung der Verbindungen zwischen den Neuronen
    net.inputWeights{1,2}.delays = FD;                                     %Initalisierung: FD     
    
    % Trainingseinstellungen
    net.divideMode = 'time';                                               %Aufteilung der Daten anhand der Zeitschritte                 
    net.performFcn = 'mse';                                                %Trainingsergebnisse(Mittlerer quatratischer Fehler) 
    net.divideFcn = 'divideblock';                                         %Funktion für die Aufteilung der Trainingsdaten(Divideblock default) 
    net.divideParam.trainRatio = 70/100;                                   %Aufteilung in Trainings-,Validierungs-, und Testdaten
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = 0;                                         %Anzeigen von Trainingsinformationen durch Toolbox                                    
    net.trainParam.showCommandLine = 0;
    net.plotFcns = {'plotperform','plottrainstate','plotresponse', ...     %Verschiedene Trainingsinformation während des Trainings     
    'ploterrcorr', 'plotinerrcorr'};
    
    %% Training Openloop
    [Xs,Xi,Ai,Ts] = preparets(net,input,{},target);                        %Netzwerkvorbereitung
    [net,tr,~,~,Xif,Aif] = train(net,Xs,Ts,Xi,Ai);                         %Training OL  
    [Yo,xf,af] = net(Xs,Xif,Aif);                                          %Ergbnis OL
    perfo = perform(net,Ts,Yo);                                            %Auswertung OL
    
    %% Training: Closeloop 
    [netc,xfi,afi] = closeloop(net,xf,af);                                 %Umwandlung von OL in CL                          
    [Xcs,Xci,Aci,Tcs] = preparets(netc,input,{},target);                   %Übergabe der trainierten Gewichte aus OL für die Initalisierung des Trainings im CL 
    [netc,trc,Ycsf,Ecs,Xcf,Acf] = train(netc,Xcs,Tcs,xfi,afi);             %Retrain im CL mit den erzeugen Gewichten aus OL als Startwerte    
    Yc = netc(Xcs,Xcf,Acf);                                                %Ergbnis CL
    perfc = perform(netc,Tcs,Yc);                                          %Auswertung CL         
      
    tmp = power(corrcoef(cell2mat(Yc),cell2mat(Tcs)),2);                   %Berechnung von R^2
    
    Error = struct('R2train', tmp(2,1), 'Trtrain', tr);
    
    R2(n) = Error.R2train;
    
    nets{n,1} = netc;
    nets{n,2} = Error;
    
    end
    [V, I] = max(R2);
   
    netstruct{1} = nets{I,1};
    netstruct{2} = nets{I,2};
    Trainmin = V;
toc    
end
