CONDITION = 'SC';

% we will do five-fold cross validation, hence data from five subjects is
% grouped together. These are file indices
fileIdx = {'1to5','6to10','11to15','16to20','21to25','26to30','31to35','36to39'};
% where data lives
inDir = fullfile(pwd);
% where to save results
outDir = fullfile(pwd,'gmm_lda_LOSOCV');
% number of gibbs sampler chains to run
chains = 3;

KK = 6; % number of GMMs
MM = [ 5 5 5 10 10 10 ]; % number of components per gmm
aa = [ 1 1 ]; % Dirichlet prior hyperparameters

% number of gibbs sampling iterations
numiters = 4000;

for f = 1:2
    
    fid = fileIdx{f};
    subs = cellfun(@str2num,strsplit(fid,'to'));
    subs = subs(1):subs(end);
    
    powfName = fullfile(inDir, [CONDITION, '_SleepPowerData_', fileIdx{f},'.mat']);
    labfName = fullfile(inDir, [CONDITION, '_SleepLabelData_', fileIdx{f},'.mat']);
    powdata = importdata(powfName);
    labData = importdata(labfName);
    
    xx_mx = cellfun(@(x) x', powdata, 'UniformOutput', false);
    
    DD = length(xx_mx);
    cvMask = eye(DD);
    
    parfor d = 1:DD
        trainIdx = find(cvMask(d,:) == 0);
        testIdx = find(cvMask(d,:) == 1);
        
        xx_train = xx_mx(trainIdx);
        xx_test = xx_mx(testIdx);
        yy_test = labData{testIdx};
        
        for c = 1:chains
            
            saveDir = fullfile(outDir, fid, ['FOLD_',num2str(d)]);
            
            if ~isdir(saveDir)
                mkdir(saveDir);
            end
            
            [ ~, ~, n, mu, sigma ] = nCRP_Gibbs_Sampler(xx_train, KK, MM, aa, numiters);
            
            gmm_lda = [];
            
            gmm_lda.MODE = MODE;
            gmm_lda.trainSubs = subs(trainIdx);
            gmm_lda.testSub = subs(testIdx);
            gmm_lda.testData = xx_test;
            gmm_lda.testLabels = yy_test;
            
            gmm_lda.input.KK = KK;
            gmm_lda.input.MM = MM;
            gmm_lda.input.aa = aa;
            
            gmm_lda.output.n = n;
            gmm_lda.output.mu = mu;
            gmm_lda.output.sigma = sigma;
            
            mySave(fullfile(saveDir,[CONDITION,'_gmm_lda_chain_',num2str(c),'.mat']), gmm_lda);
        end
    end
end