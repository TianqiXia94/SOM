function som = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% som = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% -- Purpose: Trains a 1D SOM i.e. A SOM where the neurons are arranged
%             in a single line. 
%             
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <neuronCount> number of neurons 
% -- <trainingSteps> number of training steps 
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius used to specify the initial neighbourhood size

% TODO:
% The student will need to complete this function so that it returns
% a matrix 'som' containing the weights of the trained SOM.
% The weight matrix should be arranged as follows, where
% N is the number of features and M is the number of neurons:
%
% Neuron1_Weight1 Neuron1_Weight2 ... Neuron1_WeightN
% Neuron2_Weight1 Neuron2_Weight2 ... Neuron2_WeightN
% ...
% NeuronM_Weight1 NeuronM_Weight2 ... NeuronM_WeightN
%
% It is important that this format is maintained as it is what
% lab_vis(...) expects.
%
% Some points that you need to consider are:
%   - How should you randomise the weight matrix at the start?
%   - How do you decay both the learning rate and radius over time?
%   - How does updating the weights of a neuron effect those nearby?
%   - How do you calculate the distance of two neurons when they are
%     arranged on a single line?
    [count,dimension] = size(trainingData);
    
    % learning rate constant
    t1 = 1000;
    % radius constant
    t2 = 1000/log(startRadius);
    
    dimension;
    % initilise weight to some small, random values
    Neuron = rand(neuronCount,(dimension));
    Neuron
    
    % normalize the weight
    for i = 1:neuronCount
        Neuron(i,:) = Neuron(i,:)/norm(Neuron(i,:));
    end
    
    % initical learning rate
    learningRate = startLearningRate;
    % initical radius
    radius = startRadius;
    %tau = 1000;
    % repeat
    for t = 1:trainingSteps
        % if learn rate too small, break loop
        if learningRate < 0.01
            break
        end
        
        for k = 1:count
            % initical mini Euclidean distance
            miniDistance = 9999999999;
            % find the minimum Euclidean distance
            for m = 1:neuronCount
                distance = norm(trainingData(k,:) - Neuron(m,:));
                if distance < miniDistance
                    miniDistance = distance;
                    winner = Neuron(m,:);
                end
            end
            
            % calculate city-block distance
            CBD = Radius(Neuron,winner);
            [M,N] = size(CBD);
            
            for w = 1:M
                if CBD(w) < radius
                    % calculate kernel 
                    kernel = exp(-1 * (CBD(w).^2)/2 * radius.^2);
                    % calculate neuron
                    Neuron(w,:) = Neuron(w,:) + learningRate * kernel * (trainingData(k,:) - Neuron(w,:));
                    
                end
                
            end
            % decrease learning rate
            learningRate  = learningRate * exp(-1* t / t1);
            % decrease radius
            radius = radius * exp(-1 * t / t2);

        end
    end
    % return final neuron
    som = Neuron
end


function CBD = Radius(Neuron,w)
    % city-block distance
    CBD = pdist2(Neuron,w,'cityblock');
    
end


