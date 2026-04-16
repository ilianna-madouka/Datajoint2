%{
# Object decoding sith simulated responses
-> simfilt.RFParams
-> simfilt.ActivityParams
-> simfilt.DecodeOpt
---
p                     : longblob                      # performance
p_shuffle             : longblob                      # chance performance
train_groups          : longblob                      # train group info
test_groups           : longblob                      # test group info
trial_info            : longblob                      # trial info [index, clip #, movie, bin #]
score                 : longblob                      # svm score (distance to boundary)
%}

classdef Dec < dj.Computed
    %#ok<*AGROW>
    %#ok<*INUSL>
    
    properties
        keySource  = (simfilt.DecodeOpt & 'process = "yes"') * proj(simfilt.RFParams & simfilt.RFRaw) * proj(simfilt.ActivityParams & simfilt.Activity)
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            
            % get train & test groups
            [train_set,test_set] = fetch1(simfilt.DecodeOpt & key,'train_set','test_set');
            assert(~isempty(train_set));
            [train_groups,test_groups] = getGroups(self,train_set,test_set);
            stims = cellfun(@(x) unique(cellfun(@(y) y,x,'uni',0)),train_groups{1},'uni',0);
            Stims = unique([stims{:}])';
            if ~isempty(test_groups)
                stims = cellfun(@(x) unique(cellfun(@(y) y,x,'uni',0)),test_groups{1},'uni',0);
                Stims = [Stims;unique([stims{:}])'];
            end
            
            % get DAta
            [Traces, Unit_ids,ClipData] = getData(self,Stims,key); % [Cells, Obj, Trials]
            if ~isempty(test_groups)
                train_sz = cell2mat(cellfun(@size,train_groups,'uni',0));
                test_sz = cell2mat(cellfun(@size,test_groups,'uni',0));
                assert(all(train_sz==test_sz),'Test groups must match train groups')
            end
            
            % create StimInfo
            StimInfo.clips = ClipData;
            for istim =1:length(Stims)
                StimInfo.bins{istim} = (1:size(Traces{istim},2))';
                StimInfo.names{istim} = repmat(Stims(istim),size(Traces{istim},2),1);
            end
            
            % run the decoding
            P = cell(length(train_groups),length(train_groups{1})); P_shfl = P;score = P;
            for iGroup = 1:length(train_groups)
                fprintf('Group# %d/%d ',iGroup,length(train_groups));
                
                train_data = [];test_data = [];
                for iClass = 1:length(train_groups{iGroup})
                    tgroup = train_groups{iGroup}{iClass};
                    stim_idx = any(strcmp(...
                        repmat(tgroup,size(Stims,1),1),...
                        repmat(Stims,1,size(tgroup,2)))',1);
                    train_data{iClass} = cell2mat(Traces(stim_idx));
                    train_info.bins{iGroup,iClass} = cell2mat(reshape(StimInfo.bins(stim_idx),[],1));
                    train_info.clips{iGroup,iClass} = cell2mat(reshape(StimInfo.clips(stim_idx),[],1));
                    
                    names = cellfun(@(x) reshape(x,1,[]),StimInfo.names,'uni',0);
                    train_info.names{iGroup,iClass} = [names{stim_idx}];
                    
                    
                    if ~isempty(test_groups)
                        tgroup = test_groups{iGroup}{iClass};
                        stim_idx = any(strcmp(...
                            repmat(tgroup,size(Stims,1),1),...
                            repmat(Stims,1,size(tgroup,2)))',1);
                        test_data{iClass} = cell2mat(Traces(stim_idx));
                        test_info.bins{iGroup,iClass} = cell2mat(reshape(StimInfo.bins(stim_idx),[],1));
                        test_info.clips{iGroup,iClass} = cell2mat(reshape(StimInfo.clips(stim_idx),[],1));
                        
                        names = cellfun(@(x) reshape(x,1,[]),StimInfo.names,'uni',0);
                        test_info.names{iGroup,iClass} = [names{stim_idx}];
                    else, test_info = train_info;
                    end
                end
                
                [P(iGroup,:), P_shfl(iGroup,:), unit_idx(iGroup,:), score(iGroup,:)]= ...
                    decodeMulti(self, train_data, test_data, key, train_info, test_info);
            end
            
            % find unit ids from randomization indexes
            test_info.units = cellfun(@(x) cellfun(@(xx) Unit_ids(xx),x,'uni',0),unit_idx,'uni',0);
            
            % insert
            key.p = P;
            key.p_shuffle = P_shfl;
            key.train_groups = train_groups;
            key.test_groups = test_groups;
            key.trial_info = test_info;
            key.score = score;
            self.insert(key)
        end
    end
    
    methods
        function [Data, Unit_ids, Clip] = getData(self,Stims,key)
            Data = [];Clip = [];
            clipinfo = fetch1(simfilt.DecodeOpt & key,'clips');
            
            for iStim = 1:length(Stims)
                k.movie_name = Stims{iStim};
                if clipinfo
                    clips = eval(clipinfo);
                    Data{iStim} = [];
                    Clip{iStim} = [];
                    for clip = clips
                        k.clip_number = clip;
                        [traces, ids] = fetchn(simfilt.ActivityTrace & key & k,'trace','filter_id');
                        Data{iStim} = [Data{iStim},cell2mat(traces)];
                        Clip{iStim} = [Clip{iStim};ones(size(traces{1},2),1)*clip];
                    end
                else
                    [traces, ids, clip_number] = fetchn(simfilt.ActivityTrace & key & k,'trace','filter_id','clip_number');
                    mn_l = min(cellfun(@length,traces));
                    traces = cellfun(@(x)  x(1:mn_l),traces,'uni',0);

                    un_idx = unique(ids);
                    Traces = [];Clips = [];
                    for i =1:length(un_idx)
                        Traces{i} = cell2mat(traces(ids==un_idx(i))');
                        if i==1
                            clip_ids = clip_number(ids==un_idx(i));
                            Clips = [];
                            for iclip = 1:length(clip_ids)
                                Clips{iclip} = ones(length(traces{1}),1)*clip_ids(iclip);
                            end
                            Clip{iStim} = cell2mat(Clips');
                        end
                    end
                    Data{iStim} = cell2mat(Traces');
                end
            end
            Unit_ids = ids;
        end
        
        function [train_groups, test_groups] = getGroups(obj,train_set,test_set)
            
            train_groups = splitGroups(train_set);
            test_groups = splitGroups(test_set);
            
            function groups = splitGroups(set)
                % output {group}{class}{obj}
                group_ids = regexp(set,'([^:\{\}$]*)','match');
                groups = [];
                for igroup = 1:length(group_ids)
                    classes = strsplit(group_ids{igroup},';');
                    for iclass = 1:length(classes)
                        groups{igroup}{iclass} = strsplit(classes{iclass},',');
                    end
                end
            end
        end
        
        function [PP, RR, Cells, SC] = decodeMulti(self,Data,test_Data, key, train_info, test_info)
            % performs a svm classification
            % data: {classes}[cells trials]
            % output: {classes}[reps trials]
            
            % get decoder parameters
            [decoder,k_fold,shuffle,repetitions,select_method,...
                dec_params, neurons,fold_selection,noise,samples,dat_norm,firing_rate] = ...
                fetch1(simfilt.DecodeOpt & key,...
                'decoder','k_fold','shuffle','repetitions','select_method',...
                'dec_params','neurons','fold_selection','noise','samples','normalize','firing_rate');
            binsize = fetch1(simfilt.RFParams & key,'binsize');
            
            % define decoder function
            if ~isempty(dec_params);dec_params = [',' dec_params];end
            decoder_func = eval(sprintf('@(X,IDs) %s(X, IDs%s)',decoder,dec_params));
            
            % get test data
             if isempty(test_Data)
                test_Data = Data;
            end
            
            PP = cell(repetitions,1); RR = PP;Cells = [];SC = PP;
            txt = '';
            for irep = 1:repetitions
                fprintf(repmat('\b',1,length(txt)));
                txt= sprintf('Rep# %d/%d ',irep,repetitions);
                fprintf('%s',txt);
                rseed = RandStream('mt19937ar','Seed',irep);
                RandStream.setGlobalStream(rseed)
                
                % initialize
                groups = []; test_groups = []; train_idx = []; test_idx = [];
                group_num = length(Data);
                
                % equalize by undersampling shorter class & randomize trial sequence
                if samples == 0
                    msz = min(cellfun(@(x) size(x,2),Data)); % calculate minimum class length
                    msz_test = min(cellfun(@(x) size(x,2),test_Data)); % calculate minimum class length
                else
                    msz = samples;
                    msz_test = samples;
                end
                
                % calculate fold bin size and recompute minimum length of data
                if k_fold<2;bins = msz;else;bins = k_fold;end
                bin_sz = floor(msz/bins);
                msz = bin_sz*bins;
                
%                 % undersample data
%                 train_data_idx = cellfun(@(x) randperm(rseed,msz,msz),Data,'uni',0);% create bin index

%                 % equalize by undersampling shorter class & randomize trial sequence
%                 msz = min(cellfun(@(x) size(x,2),test_Data)); % calculate minimum class length
%                 test_bin_sz = floor(msz/bins); % calculate fold bin size and recompute minimum length of data
%                 msz = test_bin_sz*bins;
%                 rseed.reset;
%                 test_data_idx = cellfun(@(x) randperm(rseed,msz,msz),test_Data,'uni',0);
%                 
%                 % undersample data
%                 train_data_idx = cellfun(@(x) randperm(rseed,msz,msz),Data,'uni',0);% create bin index
%                 
%                 % equalize by undersampling shorter class & randomize trial sequence
%                 msz = min(cellfun(@(x) size(x,2),test_Data)); % calculate minimum class length
%                 test_bin_sz = floor(msz/bins); % calculate fold bin size and recompute minimum length of data
%                 msz = test_bin_sz*bins;
%                 rseed.reset;
%                 test_data_idx = cellfun(@(x) randperm(rseed,msz,msz),test_Data,'uni',0);
                
                % equalize by undersampling shorter class & randomize trial sequence
              
                test_bin_sz = floor(msz_test/bins); % calculate fold bin size and recompute minimum length of data
                msz_test = test_bin_sz*bins;
                rseed.reset

                % undersample data
                train_data_idx = cellfun(@(x) randperm(rseed,msz,msz),Data,'uni',0);% create bin index
                test_data_idx = cellfun(@(x) randperm(rseed,msz_test,msz_test),test_Data,'uni',0); % this should be the same as train for the crossvalidation to work?
                % make group identities & build indexes
                for iclass = 1:group_num
                    % make group identities
                    groups{iclass} = ones(1,size(train_data_idx{iclass},2)) * iclass;
                    test_groups{iclass} = ones(1,size(test_data_idx{iclass},2)) * iclass;
                    
                    switch fold_selection
                        case 'random'
                            % buld index
                            for ibin = 1:bins
                                train_idx{iclass}(1 + (ibin-1)*bin_sz      :      bin_sz*ibin) = ibin;
                                test_idx{iclass} (1 + (ibin-1)*test_bin_sz : test_bin_sz*ibin) = ibin;
                            end
                        case 'firing_rate'
                            ctrl_data = exprnd(firing_rate,samples,1)';
                            smd = sort(mean(Data{iclass}));
                            offset = 1;
                            for i = 1:100000
                                 offset = offset + 1;
                                 new_data = smd(1+offset:samples+offset);
                                 if ttest(ctrl_data, new_data)==0
                                     break
                                 end
                            end
                            idx = 1+offset:samples+offset;
                            idx = idx(randsample(500,500));
                            train_data_idx{iclass} = idx;% create bin index
                            test_data_idx{iclass} = idx;

                            % buld index
                            for ibin = 1:bins
                                train_idx{iclass}(1 + (ibin-1)*bin_sz      :      bin_sz*ibin) = ibin;
                                test_idx{iclass} (1 + (ibin-1)*test_bin_sz : test_bin_sz*ibin) = ibin;
                            end
                        case {'x','y','frame_id','scale','rotation','tilt','light1_xloc','light1_yloc','light3_yloc','light2_ene'}
                            keys = cell2struct([num2cell(train_info.clips{iclass}(train_data_idx{iclass}),2),...
                                train_info.names{iclass}(train_data_idx{iclass})']',{'clip_number','movie_name'},1);
                            idx = [1;find(diff(train_info.clips{1})>0)+1];
                            train_info.bins{1} = reshape((repmat(1:min(diff(idx)),length(idx),1))',[],1);
                            idx = [1;find(diff(train_info.clips{2})>0)+1];
                            train_info.bins{2} = reshape((repmat(1:min(diff(idx)),length(idx),1))',[],1);
                            [keys.animal_id] = deal(9254);
                            [keys.session] = deal(1);
                            param = getParam(stimulus.MovieParams, keys, fold_selection, ...
                                train_info.bins{iclass}(train_data_idx{iclass})*binsize/1000 - binsize/2/1000);
                            if bins>2; v = quantile(param,bins-1)'; else v  = prctile(param,50);end
                            [~, sort_idx] = histc(param,[-inf; v; inf]);
                            train_idx{iclass} = sort_idx(:)';
                            keys = cell2struct([num2cell(test_info.clips{iclass}(test_data_idx{iclass}),2),...
                                test_info.names{iclass}(test_data_idx{iclass})']',{'clip_number','movie_name'},1);
                            idx = [1;find(diff(test_info.clips{1})>0)+1];
                            test_info.bins{1} = reshape((repmat(1:min(diff(idx)),length(idx),1))',[],1);
                             idx = [1;find(diff(test_info.clips{2})>0)+1];
                            test_info.bins{2} = reshape((repmat(1:min(diff(idx)),length(idx),1))',[],1);
                             [keys.animal_id] = deal(9254);
                            [keys.session] = deal(1);
                            param = getParam(stimulus.MovieParams, keys, fold_selection, ...
                                test_info.bins{iclass}(test_data_idx{iclass})*binsize/1000 - binsize/2/1000);
                            if bins>2; v = quantile(param,bins-1)'; else v  = prctile(param,50);end
                            [~, sort_idx] = histc(param,[-inf; v; inf]);
                            test_idx{iclass} = sort_idx(:)';
                        case 'clips'
                            for ibin = 1:bins
                                train_clips(1 + (ibin-1)*bin_sz      :      bin_sz*ibin) = ibin;
                                test_clips(1 + (ibin-1)*test_bin_sz : test_bin_sz*ibin) = ibin;
                            end
                            [~,~, t] = unique(train_clips(train_data_idx{iclass}));
                            train_idx{iclass} = t';
                            [~,~,t] = unique(test_clips(test_data_idx{iclass}));
                            test_idx{iclass} = t';
                        otherwise
                            error('Fold selection method not implemented!')
                    end
                end
                
                % combine classes in one vector
                data = cell2mat(cellfun(@(x,y) x(:,y),Data,train_data_idx,'uni',0));
                groups = cell2mat(groups);
                test_data = cell2mat(cellfun(@(x,y) x(:,y),test_Data,test_data_idx,'uni',0));
                test_groups = cell2mat(test_groups);
                train_idx = cell2mat(train_idx);
                test_idx = cell2mat(test_idx);
                test_data_idx = cell2mat(test_data_idx);

                % normalize data
                if dat_norm
                    mdata = mean(data,2);
                    sdata = std(data,[],2);
                else
                    mdata = zeros(size(data,1),1);
                    sdata = ones(size(data,1),1);
                end
                
                data = bsxfun(@rdivide, bsxfun(@minus,data,mdata),sdata);
                test_data = bsxfun(@rdivide, bsxfun(@minus,test_data,mdata),sdata);
                
                % make nan zeros
                data(isnan(data)) = prctile(data(:),1);
                
                % add noise
                if ~isempty(noise)
                    noise_func = eval(noise);
                    data = noise_func(data);
                    test_data = noise_func(test_data);
                end
                
                % create shuffled testing trials
                test_sz = size(test_data,2);
                test_shfl_idx = 1:size(test_groups,2);
                for ishuffle = 1:shuffle
                    test_shfl_idx = test_shfl_idx(randperm(rseed,test_sz));
                end
                test_shfl_groups = test_groups(test_shfl_idx);
                data_shfl_idx = test_data_idx(test_shfl_idx);
                
                % get cell index
                cell_idx = randperm(rseed,size(Data{1},1));
                switch select_method
                    case 'subsample'
                        cell_num = tril(true(numel(cell_idx)),0);
                        cell_num = cell_num([1 2.^(1:log2(numel(cell_idx)-1)) numel(cell_idx)],:);
                    case 'all'
                        cell_num = true(size(cell_idx));
                    case 'single'
                        cell_num = diag(true(size(cell_idx)));
                    case 'fixed'
                        if neurons>size(Data{1},1); error('Recording only has %d neurons!',size(Data{1},1));end
                        cell_num = false(1,numel(cell_idx));
                        cell_num(1:neurons) = true;
                    otherwise
                        error('Cell selection method not supported')
                end
                
                % classify
                P = cellfun(@(x) nan(size(cell_num,1),size(x,2),'single'),test_Data,'uni',0);R = P;S = P;
                for icell = 1:size(cell_num,1) % For each cell permutation
                    for ibin = 1:bins          % For each fold
                        % select training/testing bin
                        idx = train_idx ~= ibin;
                        tidx = test_idx == ibin;
                        
                        % run classifier
                        DEC = decoder_func(data(cell_idx(cell_num(icell,:)),idx)',groups(idx)');
                        [pre, sc] = predict(DEC,test_data(cell_idx(cell_num(icell,:)),tidx)'); % test decoder
                        
                        % Assign performance data into bins
                        p =  pre;
                        r =  pre;
                        for igroup = 1:group_num
                            P{igroup}(icell,test_data_idx(tidx & test_groups==igroup)) = p(test_groups(tidx)==igroup);
                            R{igroup}(icell,data_shfl_idx(tidx & test_shfl_groups==igroup)) = ...
                                r(test_shfl_groups(tidx)==igroup);
                            S{igroup}(icell,test_data_idx(tidx & test_groups==igroup)) = sc(test_groups(tidx)==igroup,1);
                            D{igroup}{icell,ibin}(3,:) = mdata(cell_idx(cell_num(icell,:)));
                            D{igroup}{icell,ibin}(4,:) = sdata(cell_idx(cell_num(icell,:)));

                        end
                    end
                    Cells{irep}{icell} = cell_idx(cell_num(icell,:));
                end
                PP{irep} = P;
                RR{irep} = R;
                SC{irep} = S;
            end
            
            % convert {reps}{obj}[cells trials] to {obj}[reps trials cells]
            PP = cellfun(@cell2mat,mat2cell(cellfun(@(x) permute(x,[3 2 1]),permute(reshape([PP{:}],...
                length(Data),repetitions),[2 1]),'uni',0),repetitions,ones(1,length(Data))),'uni',0);
            RR = cellfun(@cell2mat,mat2cell(cellfun(@(x) permute(x,[3 2 1]),permute(reshape([RR{:}],...
                length(Data),repetitions),[2 1]),'uni',0),repetitions,ones(1,length(Data))),'uni',0);
            SC = cellfun(@cell2mat,mat2cell(cellfun(@(x) permute(x,[3 2 1]),permute(reshape([SC{:}],...
                length(Data),repetitions),[2 1]),'uni',0),repetitions,ones(1,length(Data))),'uni',0);
            
            
        end
        
        function [perf, keys] = getPerformance(self,varargin)
            
            params.target_cell_num = [];
            params.perf = 'p';
            params.mi = 1;
            params.autoconvert = true;
            params.reps = 0;
            
            params = getParams(params,varargin);
            
            [p, ti, keys] = fetchn(self, params.perf,'trial_info');
            
            % remove empty classes
            p = cellfun(@(x) x(~cellfun(@isempty,x)),p,'uni',0);
            perf = cell(length(p),1);
            for ikey = 1:length(p)
                if isempty(params.target_cell_num)
                    ncel =1:size(p{ikey}{1},3);
                elseif params.target_cell_num==0
                    ci = cellfun(@(x) max([find(cell2mat(cellfun(@length,x.units{1},'uni',0))>...
                        params.target_cell_num,1,'last'),0]),ti);
                    ncel = ci(ikey);
                    if ncel==0;continue;end
                else
                    ci = cellfun(@(x) max([find(cell2mat(cellfun(@length,x.units{1},'uni',0))>...
                        params.target_cell_num-1,1,'first'),0]),ti);
                    ncel = ci(ikey);
                    if ncel==0; perf{ikey} = nan;continue;end
                end
                for icell = ncel
                    
                    if ~params.reps
                        P = cellfun(@(x) x(:,:,icell),p{ikey},'uni',0);
                        CM = nan(length(P));
                        for igroup = 1:length(P)
                            for itest = 1:length(P)
                                
                                CM(igroup,itest) = nansum(P{igroup}(:)==itest);
                            end
                        end
                        if params.mi
                            perf{ikey}(end+1) = self.getMI(CM);
                        else
                            perf{ikey}(end+1) = sum(diag(CM))/sum(CM(:));
                        end
                        
                    else
                        for irep = 1:size(p{1}{1},1)
                            P = cellfun(@(x) x(irep,:,icell),p{ikey},'uni',0);
                            CM = nan(length(P));
                            for igroup = 1:length(P)
                                for itest = 1:length(P)
                                    
                                    CM(igroup,itest) = nansum(P{igroup}(:)==itest);
                                end
                            end
                            if params.mi
                                perf{ikey}(end+1) = self.getMI(CM);
                            else
                                perf{ikey}(end+1) = sum(diag(CM))/sum(CM(:));
                            end
                        end
                    end
                    
                end
            end
            if all(cellfun(@length,perf)==1) && params.autoconvert
                perf = cell2mat(perf);
            end
        end
        
        function [perf, keys] = getPerformance2(self,varargin)
            
            params.target_cell_num = [];
            params.perf = 'p';
            params.mi = 1;
            params.data = [];
            params.autoconvert = true;
            params.reps = false;
            params.median_over_trials = false;
            
            params = getParams(params,varargin);
            
            [p, ti, keys] = fetchn(self, params.perf,'trial_info');
            if ~isempty(params.data)
                p = {params.data};
            end
            
            % remove empty classes
            p = cellfun(@(x) x(:,any(~cellfun(@isempty,x),1)),p,'uni',0);
            perf = cell(length(p),1);
            for ikey = 1:length(p)
                if params.reps
                    p{ikey} = cellfun(@(x) permute(x(:,:,size(p{ikey}{1},3)),[3 2 1]),p{ikey},'uni',0);
                    ncel =1:size(p{ikey}{1},3);
                elseif isempty(params.target_cell_num)
                    ncel =1:size(p{ikey}{1},3);
                elseif params.target_cell_num==0
                    ci = cellfun(@(x) max([find(cell2mat(cellfun(@length,x.units{1},'uni',0))>...
                        params.target_cell_num,1,'last'),0]),ti);
                    ncel = ci(ikey);
                    if ncel==0;continue;end
                else
                    ci = cellfun(@(x) max([find(cell2mat(cellfun(@length,x.units{1},'uni',0))>...
                        params.target_cell_num-1,1,'first'),0]),ti);
                    ncel = ci(ikey);
                    if ncel==0; perf{ikey} = nan;continue;end
                end
                if params.median_over_trials
                     for icell = ncel
                        P = cellfun(@(x) x(:,:,icell),p{ikey},'uni',0);
                        for iclass = 1:size(P,1)
                            prf = nan(size(P{1},1),1);
                            for itrial = 1:size(P{1},1)
                                CM = nan(size(P,2));
                                for igroup = 1:size(P,2)
                                    for itest = 1:size(P,2)
                                        CM(igroup,itest) = nansum(P{iclass,igroup}(itrial,:)==itest);
                                    end
                                end
                                if params.mi
                                    prf(itrial) = self.getMI(CM);
                                else
                                    prf(itrial) = sum(diag(CM))/sum(CM(:));
                                end
                            end
                            perf{ikey}(end+1) = nanmedian(prf);
                        end
                    end
                else
                    for icell = ncel
                        P = cellfun(@(x) x(:,:,icell),p{ikey},'uni',0);
                        for iclass = 1:size(P,1)
                            CM = nan(size(P,2));
                            for igroup = 1:size(P,2)
                                for itest = 1:size(P,2)
                                    CM(igroup,itest) = nansum(P{iclass,igroup}(:)==itest);
                                end
                            end
                            if params.mi
                                perf{ikey}(end+1) = self.getMI(CM);
                            else
                                perf{ikey}(end+1) = sum(diag(CM))/sum(CM(:));
                            end
                        end
                    end
                end
            end
            if all(cellfun(@length,perf)==1) && params.autoconvert
                perf = cell2mat(perf);
            end
        end
    end
    
    methods (Static)
        function mi = getMI(CM)
            CM = CM+eps;
            p = CM/sum(CM(:));
            pi = sum(CM,2)/sum(CM(:));
            pj = sum(CM,1)/sum(CM(:));
            pij = pi*pj;
            mi = sum(sum(p.*log2(p./pij)));
        end
    end
end
