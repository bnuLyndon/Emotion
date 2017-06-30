%clear

if(isunix)
    executable = '"../build/bin/FeatureExtraction"';
else
    executable = '"../x64/Release/FeatureExtraction.exe"';
end

%output = './output_features_vid/';

if(~exist(output, 'file'))
    mkdir(output)
end
    
%in_files = dir('/home/lyndon/Downloads/Train_SASE-FE/1/D2N2Sur.MP4');
in_files = dir(in_files_path);
%in_files(1)=[];
%in_files(1)=[];

% some parameters
verbose = true;

command = executable;

% Remove for a speedup
command = cat(2, command, ' -verbose ');

% add all videos to single argument list (so as not to load the model anew
% for every video)
for i=1:numel(in_files)
    
    inputFile = [strcat(dataPath, dataDir(ii).name,'/'), in_files(i).name];
    [~, name, ~] = fileparts(inputFile);
    the_name=name;
    name=[];
    
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

%% Output HOG files
[hog_data, valid_inds, vid_id] = Read_HOG_files({[]}, output_forhog);

try
    outputFile=strcat(output_forhog,num2str(ii-2),'.txt');
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
