% run_examples.m
%
% Example MATLAB workflow for reading STEPS-V inference output and selecting
% candidate step epochs from the model probability time series.
%
% IMPORTANT
% ---------
% These examples are provided to illustrate the general STEPS-V workflow
% and downstream MATLAB postprocessing. They are intended as simple usage
% examples, not as a full reproduction of the complete operational input
% preparation workflow.
%
% In particular, the example station files used here do NOT include the
% additional preprocessing steps typically applied in the standard STEPS-V
% v1.0 workflow before inference. These omitted steps include:
%
%   1. Outlier cleaning
%   2. NTAOL corrections
%
% As a result, these examples should be viewed as workflow demonstrations
% only. For research or operational use, users should prepare input vertical
% displacement time series in a manner consistent with the STEPS-V model
% training workflow.
%
% This script demonstrates how to:
%   1. Prepare example station files
%   2. Run STEPS-V inference
%   3. Load .npz inference output using load_inference_npz
%   4. Select candidate steps using pick_steps_from_prob
%   5. Plot displacement and probability results





% !rm example_inference_out/*
% !rm tenv3/*
% !rm GPS_data/*

%% Variables 
inference_dir='example_inference_out/'
extn1='_inference.npz'
figon=1;

%% Add stations to inference test
% Users may provide their own GPS data files in GPS_data/ or populate the
% following "stlist" , if stlist is populated the script will download the
% most recent tenv3 file from the Nevada Geodetic Laboratory and prepare
% the files for STEPSV use. 

% stlst=[];
stlst=['ECHO';'P310';'P006';'MSMC';'TXWH'];


if ~isempty(stlst)
prepare_vert_files(stlst);
end


%% Run Inference on all files in GPS_data/
!./run_inference.sh



!find example_inference_out/ -type f -name "*.npz" | sort > tmp11
 
fn1=importdata('tmp11')
 !rm tmp11

% Define probability threshold
% thresh=[0.1;0.25;0.5;0.75;0.9]
% threshl=['0.10';'0.25';'0.50';'0.75';'0.90']
thresh=[0.5 ]
threshl=['0.50'] 
 
% optional loop through thresholds
% for jj=1:length(thresh(:,1))
%     tt=thresh(jj,1)
%     ttl=threshl(jj,:)
 
    tt=thresh(1,1)
    ttl=threshl(1,:)
 
% % % Preallocate res as a cell array
res = cell(length(fn1),4);  

for a = 1:length(fn1)
% parfor a = 1:length(fn1) 
 
    s1 = strsplit(fn1{a}, '/');
    s2 = strsplit(s1{1,2}, '.');
    s3 = strsplit(s2{1,1}, '_');    
    st = s3{1};
 
    nn=length(st); 
    
    row = cell(1,4);
    row{1} = st;
    dat=importdata(['GPS_data/' st '.data']);
    row{2}= dat;
   
    fnm = [ inference_dir st extn1 ];
    [t, p, pstd, lo1, hi1] = load_inference_npz(fnm);
    stept = pick_steps_from_prob(t, p, 'Thresh', tt, 'MinSepDays', 5, 'IgnoreFirstN', 5);
    prob=[ t p];

    row{3} = prob; 
    row{4} = stept;
 
    res(a,:) = row;
 
    %optional plot showing time series, picked steps, and probability trace
    
    if figon==1
    
        figure
        subplot(2,1,1)
        hold on
        if ~isempty(stept)
            for aa=1:length(stept(:,1))
                plot([stept(aa,1) stept(aa,1)],[min(dat(:,2)) max(dat(:,2))],'r')
            end
        end
        scatter(dat(:,1),dat(:,2),6,'k','filled')
        ylabel('Vertical Displacement (mm)')
        title([st ' STEPS-V Results'])
 
        subplot(2,1,2)
        plot([min(prob(:,1)) max(prob(:,1))],[tt tt],'-.k')
        hold on 
        plot(prob(:,1),prob(:,2),'k')
        ylim([0 1])
        ylabel('Step Probability')

    end
 
end

% end



