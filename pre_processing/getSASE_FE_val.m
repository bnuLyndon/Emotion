if(isunix)
    executable = '"../build/bin/FeatureExtraction"';
else
    executable = '"../x64/Release/FeatureExtraction.exe"';
end

for j=1:6
    mkdir(strcat(hogPath,emo_list{j}));
end

for j=1:6
    mkdir(strcat(lmPath,emo_list{j}));
end

if(~exist(output, 'file'))
    mkdir(output)
end
    
%in_files = dir('/home/lyndon/Downloads/Val_SASE-FE/');
%in_files = dir('/home/lyndon/Downloads/CTest/');
% some parameters
verbose = true;

command = executable;

% Remove for a speedup
command = cat(2, command, ' -verbose ');

% add all videos to single argument list (so as not to load the model anew
% for every video)
for i=3:numel(in_files)
    
    inputFile = [inputTempPath, in_files(i).name];
    [~, name, ~] = fileparts(inputFile);
    
    % where to output tracking results
    outputFile = [output name '.txt'];
            
    if(~exist([output name], 'file'))
        mkdir([output name]);
    end
    
    outputDir_aligned = [output name];
    
    outputHOG_aligned = [output name '.hog'];
    
    output_shape_params = [output name '.params.txt'];
    
    command = cat(2, command, [' -f "' inputFile '" -of "' outputFile '"']);        
    command = cat(2, command, [' -simalign "' outputDir_aligned '" -hogalign "' outputHOG_aligned '"' ]);    
                 
end

if(isunix)
    unix(command);
else
    dos(command);
end

for i=3:numel(in_files)
    
    inputFile = [inputTempPath, in_files(i).name];
    [~, name, ~] = fileparts(inputFile);
    
    % where to output tracking results
    outputFile = [output name '.txt'];

    outputDir_aligned = [output name];
    
    outputHOG_aligned = [output name '.hog']; 
    [hog_data, valid_inds, vid_id] = my_Read_HOG_files(name, output);
    
    try
        tab = readtable(outputFile);
        column_names = tab.Properties.VariableNames;

        all_params  = dlmread(outputFile, ',', 1, 0);

        % This indicates which frames were succesfully tracked
        valid_frames = logical(all_params(:,4));
        time = all_params(valid_frames, 2);

        %% Finding which header line starts with p_ (basically model params)
        shape_inds = cellfun(@(x) ~isempty(x) && x==1, strfind(column_names, 'p_'));

        % Output rigid (first 6) and non-rigid shape parameters
        shape_params = all_params(valid_frames, shape_inds);

    catch
        shape_params = zeros(1, 40);
        'error'
    end
    
    for j=1:6
        ind=findstr(emo_list{j}, name);
        if ~isempty(ind)
            matName_hog=strcat(hogPath,emo_list{j},'/',name,'.mat');
            matName_lm=strcat(lmPath,emo_list{j},'/',name,'.mat');
        end
    end
    
    save(matName_lm,'shape_params');   
    save(matName_hog,'hog_data');
end
