close all;

CONDITION = 'SC';
MODE = '_';
chain = '3';
ch = '1';

fileIdx = {'1to5','6to10','11to15','16to20','21to25','26to30','31to35','36to39'};
% fileIdx = {'1to5','11to15','21to25'};
datDir = fullfile(['C:\Users\ksp6\Documents\Research\SleepStages\data\30sPwrFeatsAndLabels_', MODE], ['ch_', ch]);
% datDir = fullfile(['C:\Users\ksp6\Documents\Research\SleepStages\data\30sPwrFeatsAndLabels'], ['ch_', ch]);
infDir = fullfile(pwd, 'ncrp_fixed_LOSOCV', MODE, ['ch_',ch]);

% figure('Position',[1980 30 566 940]);

compSys = computer;

DO_PLOT = false;

if DO_PLOT
    switch compSys
        case 'GLNXA64'
            fig_1 = figure('Position',[1689 263 388 703]);
            fig_2 = figure('Position',[2083 263 388 703]);
            fig_3 = figure('Position',[2477 263 388 703]);
        case 'PCWIN64'
            fig_1 = figure('Position',[1965 220 388 703]);
            fig_2 = figure('Position',[2429 220 388 703]);
            fig_3 = figure('Position',[2917 220 388 703]);
    end
end

for f = 1:length(fileIdx)
    
    fid = fileIdx{f};
    subs = cellfun(@str2num,strsplit(fid,'to'));
    subs = subs(1):subs(end);
    
    foldDir = dir(fullfile(infDir,fid,'FOLD_*'));
    nFolds = length(foldDir);
    
    labData = importdata(fullfile(datDir,[CONDITION,'_SleepLabelData_',fid,'_',MODE,'.mat']));
    
    for fold = 1:nFolds
        fName = fullfile(infDir,fid,foldDir(fold).name,[CONDITION,'_ncrp_fixed_chain_',chain,'.mat']);
        
        if ~exist(fName, 'file')
            continue;
        end
        
        infData = importdata(fName);
        testSub = infData.testSub;
        label = labData(subs == testSub);
        label = label{1}';
        power = infData.testData{1}';
        
        KK = infData.input.KK;
        
        n = infData.output.n;
        mu = infData.output.mu;
        sigma = infData.output.sigma;
        
        DD = length(n);
        counts = cell(KK, 1);
        for k = 1:KK
            counts_k = [];
            for d = 1:DD
                counts_k = cat(1, counts_k, n{d}{k});
            end
            counts{k} = counts_k;
        end
        counts = cellfun(@(x) sum(x,1), counts, 'UniformOutput', false);
        
        gmms = cell(KK, 1);
        for k = 1:KK
            mu_k = mu{k}';
            sigma_k = sigma{k};
            p_k = counts{k} ./ sum(counts{k});
            p_k(p_k == 0) = eps;
            
            gmms{k} = gmdistribution(mu_k, sigma_k, p_k);
        end
        
        counts = cellfun(@sum, counts);
        N = size(power, 1);
        posteriors = zeros(N,KK);
        
        for k = 1:KK
            posteriors(:,k) = log(gmms{k}.pdf(power)) + log(counts(k) ./ sum(counts));
        end
        
        [~, Z] = max(posteriors, [], 2);
        Z = Z';
        
%         hacky_matching(label, Z, fig_1, fig_2, fig_3, DO_PLOT);
        
        results = struct('labels', label, 'clusters', Z);
        save(fullfile(infDir,fid,foldDir(fold).name,['test_results_chain_',chain,'.mat']),'results');
    end
    
end

function hacky_matching(l, z, fig_1, fig_2, fig_3, DO_PLOT)

[lprops,~] = histcounts(l,1:7);
[zprops,~] = histcounts(z,1:7);

[~,lidx] = sort(lprops,'descend');
[~,zidx] = sort(zprops,'descend');

if DO_PLOT
    figure(fig_1);
    subplot(2,1,1);
    bar(lprops./sum(lprops),0.7);
    title('labels prop');
    subplot(2,1,2);
    bar(zprops./sum(zprops),0.7);
    title('old z prop');
    
%     if strcmp(compSys, 'PCWIN64')
%         suptitle(sprintf('SUB %d\n',sub));
%     end
end

% fprintf('SUB %d. ',subID);

z_new = z;
for j = 1:6
    z_new(z == zidx(j)) = lidx(j);
end

if DO_PLOT
    cmap = hsv(6);
    
    figure(fig_2);
    subplot(3,1,1);
    imagesc(l);
    colormap(cmap);
    h = colorbar;
    set(h,'Ticks',unique(l));
    title('labels');
    
    subplot(3,1,2);
    imagesc(z);
    colormap(cmap);
    h = colorbar;
    set(h,'Ticks',unique(z));
    title('old Z');
    
    subplot(3,1,3);
    imagesc(z_new);
    colormap(cmap);
    h = colorbar;
    set(h,'Ticks',unique(z_new));
    title('new Z');
    
%     if strcmp(compSys, 'PCWIN64')
%         suptitle(sprintf('SUB %d\n',sub));
%     end
end

[znewprops,~] = histcounts(z_new,1:7);

if DO_PLOT
    figure(fig_3);
    subplot(2,1,1);
    bar(lprops./sum(lprops),0.7);
    title('labels prop');
    subplot(2,1,2);
    bar(znewprops./sum(znewprops),0.7);
    title('new z prop');
%     if strcmp(compSys, 'PCWIN64')
%         suptitle(sprintf('SUB %d\n',sub));
%     end
end

pc = prtScorePercentCorrect(z_new', l');
pc = 100*pc;
fprintf('Accuracy: %3.2f\n',pc);
% allAcc = cat(2, allAcc, pc);

if DO_PLOT
    keyboard
end

end