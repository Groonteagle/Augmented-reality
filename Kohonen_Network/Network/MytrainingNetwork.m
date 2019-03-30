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

%This is the training program.
%We enter the "weight matrix" which will evolve with the training, and we
%enter the input vectors which are the experimental data.

function [Wij] = MytrainingNetwork (Wij,learningrate,X)

k = 1;
flag = true; %Flag to obtain the best number of iteration.

while flag %While flag is True, compute every iteration you want.
        
    for i = 1 : length(X(:,1))
        %fprintf("Training for the vector %d\n",i);
        
        OldWij = Wij;
        
        %Enter in the Distance progrma to compute the Euclydian distance D1
        %and D2 and compute the new Wij.
        Wij = mydistance(Wij,X,i,learningrate);    
        
    end
    
    %After each iteration, the learning rate is divided by 2.
    learningrate = learningrate/2;
    
    %"k" is the count of iterations.
    k = k+1;
    
    %Because of the decreasing of the learning rate, after 56 iterations,
    %the Weight "Wij" doesn't changes, that says the program finished to learn,
    %So we compare the old Wij with the new one. If the old - the new is equal
    %to 0, the Flag become false, the program stop and display the number of
    %iterations with the learning rate.
    
    if OldWij - Wij == 0
       flag = false;
    end  
     
end

fprintf("After %d iteration \n", k); % Display number of iteration.
fprintf("Learning rate = %d\n\n", learningrate);%Display the last learning rate.

end