%-------------------------------------------------------------------%
%Name : Kevin SECRET-MORLAND                                        %
%Section : Master 1 VIBOT                                           %
%Year : 2018/2019                                                   %
%Title : Kohonen Network for patient 1.0                            %
%-------------------------------------------------------------------%

%This program is trained for classifying patient data and tell us if the
%patient is sick or not.

%It's constitued of 3 programs : The training Network, the calcul of the
%distance, the test and the main.

%This is the Test program.
%This program has two main functionalities: The first one is to know which
%column correspond to sick and good health patient with control data
%The second one is to compute the real patient data we want to know thanks
%to training.

function [DistanceEuclyd] = Test_Network (Wij, Y)

DistanceEuclyd = [];
TempDis1 = [];
TempDis2 = [];

for m = 1 : length(Y(:,1))
    
    for n = 1 : length(Wij)
        
        %We simply perform the calcul of Euclydian Distance D1 and D2
        %without modify Wij because it's not training.
        TempDis1(n) = (Y(m,n) - Wij(n,1)).^2;
        TempDis2(n) = (Y(m,n) - Wij(n,2)).^2;
        
    end
    
    Dis1 = sum(TempDis1);
    Dis2 = sum(TempDis2);
    
    %We stock all the new data in a matrix to display.
    DistanceEuclyd(m,:) = [Dis1,Dis2];
    
end


end