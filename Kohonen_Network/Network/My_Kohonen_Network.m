clc
clear

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

%This is the Main program.
%I allow to input the data in the different functions and display the
%results with a easier style.


%First we load the data files, there's the patient file with only sick
%people, the control file with only good people, and the test file with 4
%random patients.

load('patient.txt');
control = load('control.txt');
patient_test = load('test_one.txt');

%We concatenate the control and patient files to learn all the
%possibilities to the program
data = [control ; patient_test];

%The original weight if randomly filled
Wij = rand(length(control),2);
%Wij = round(Wij*100)/100;

learningrate = 0.6;

%We enter in the training function.
Wij = MytrainingNetwork(Wij,learningrate,data);

%Now we perform which column correspond to the good person, and by
%elimination, we can know the sick column. And we compare with the test
%file.
Control_Test = Test_Network(Wij,control);
Patient_Test = Test_Network(Wij,patient_test);

%I had a small code to easly and quickly understand the final results

%First we determine which column correspond to "good". The smallest columns
%has to represent the "good" health.

%We have to perform this code because sometimes the columns can change there position 
%due to the randomly filled "Wij".

if Control_Test(:,1) < Control_Test(:,2)
    controltemp = 0;
else
    controltemp = 1;
end

%Finally, we compare the location of "good" column with the test result.

for i = 1 : length(Patient_Test(:,1))
    
        if Patient_Test(i,1) < Patient_Test(i,2) && controltemp == 0
            fprintf("The patient %d is good\n",i);
        
        elseif Patient_Test(i,1) > Patient_Test(i,2) && controltemp == 0
            fprintf("The patient %d is sick\n",i);
        end
        
        if Patient_Test(i,1) > Patient_Test(i,2) && controltemp == 1
            fprintf("The patient %d is good\n",i);
        
        elseif Patient_Test(i,1) < Patient_Test(i,2) && controltemp == 1
            fprintf("The patient %d is sick\n",i);
        end
        
end
