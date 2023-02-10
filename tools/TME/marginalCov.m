function [sigma] = marginalCov(dataTensor, marVar)
    T = size(dataTensor,1);
    N = size(dataTensor,2);
    C = size(dataTensor,3);
    M = struct('T', [], 'TN', [], 'TNC', [], 'N', [], 'NC', [], 'TC', []);
    meanT = sumTensor(dataTensor, [2 3])/(C*N);
    meanN = sumTensor(dataTensor, [1 3])/(C*T);
    meanC = sumTensor(dataTensor, [1 2])/(T*N);
    mu.T = meanT(:);
    mu.N = meanN(:);
    mu.C= meanC(:);
    
    %% Mean is caluclated by subtracting each reshaping mean
    dataTensor0 = dataTensor;
    meanT = sumTensor(dataTensor0, [2 3])/(C*N);
    dataTensor0 = bsxfun(@minus, dataTensor0, meanT);
    meanN = sumTensor(dataTensor0, [1 3])/(C*T);
    dataTensor0 = bsxfun(@minus, dataTensor0, meanN);
    meanC = sumTensor(dataTensor0, [1 2])/(T*N);
    dataTensor0 = bsxfun(@minus, dataTensor0, meanC);
    M.TNC = dataTensor-dataTensor0;
    M.TN = repmat(sumTensor(M.TNC, [3])/(C), 1, 1, C);
    M.NC = repmat(sumTensor(M.TNC, [1])/(T), T, 1, 1);
    M.TC = repmat(sumTensor(M.TNC, [2])/(N), 1, N, 1);
    M.T = repmat(sumTensor(M.TNC, [2 3])/(N*C), 1, N, C);
    M.N = repmat(sumTensor(M.TNC, [1 3])/(T*C), T, 1, C);
    meanTensor = M.TNC;
    
    %% subtract the mean tensor and calculate the covariances
    XT = reshape(permute((dataTensor-meanTensor),[3 2 1]), [], T);
    
    XN = reshape(permute((dataTensor-meanTensor),[1 3 2]), [], N);
    
    XC = reshape(permute((dataTensor-meanTensor),[1 2 3]), [], C);
    sigma_T = (XT'*XT);
    sigma_N = (XN'*XN);
    sigma_C = (XC'*XC);

    if strcmp(marVar, 'T')
        sigma = sigma_T;
    elseif strcmp(marVar, 'N')
        sigma = sigma_N;
    elseif strcmp(marVar, 'C')
        sigma = sigma_C;
    else
        error('please specify a correct covariance matrix') 
    end
    
    end