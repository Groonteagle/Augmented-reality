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

%This is the Distance program.
%It compute the distance between D1 and D2 to know which column of Wij will
%be computed of the first input vector.


function [Wij] = mydistance(Wij, X, n, learningrate)

D1temp = [];
D2temp = [];

for j = 1 : length(Wij)

    %Calcul of the Euclidian Distances D1 and D2.
    D1temp(j) = (Wij(j,1) - X(n,j)).^2;

    D2temp(j) = (Wij(j,2) - X(n,j)).^2;

end

D1 = sum(D1temp);
D2 = sum(D2temp);

if D2 < D1
    
    %Update the 2nd column of Wij.            
        for k = 1 : length(Wij(:,1))
            Wij(k,2) = (Wij(k,2)+learningrate.*(X(n,k)-Wij(k,2)));

        end   
        
else
    
    %Update the 1st column of Wij.          
        for k = 1 : length(Wij(:,1))
            Wij(k,1) = (Wij(k,1)+learningrate.*(X(n,k)-Wij(k,1)));

        end

end

end
