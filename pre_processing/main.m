%% Please set up data path here

trainDataPath='/home/lyndon/Downloads/Train_SASE-FE/'; %Training video path
valDataPath='/home/lyndon/Downloads/Val_SASE-FE/'; %Validation video path
testDataPath='/home/lyndon/Downloads/CTest/'; % Test video path

%%
dataPath=trainDataPath;
dataDir=dir(dataPath);

labelList={'N2H','N2S','N2D','N2A','N2C','N2Sur','S2N2H','H2N2S','H2N2D','H2N2A','H2N2C','D2N2Sur'};
emo_list={'HAPPINESS','SADNESS','DISGUST','ANGER','CONTENTMENT','SURPRISE'};

for jj=1:12
    mkdir(strcat('./CL_train/',labelList{jj}));
    mkdir(strcat('./CL_LandMarks_train/',labelList{jj}));
end

for ii=3:length(dataDir)
    
    for jj=1:12
    
    	in_files_path=strcat(dataPath, dataDir(ii).name,'/',labelList{jj},'.MP4');
        output = strcat('./output_features_train/',labelList{jj},'/',dataDir(ii).name);
        output_forhog = strcat('./output_features_train/',labelList{jj},'/');
        getSASE_FE;
        matName=strcat('./CL_train/',labelList{jj},'/',dataDir(ii).name,'.mat');
        save(matName,'hog_data');
        
        matName=strcat('./CL_LandMarks_train/',labelList{jj},'/',dataDir(ii).name,'.mat');
        save(matName,'shape_params');
    end
    
end


in_files = dir(valDataPath);
output = './output_features_val/';
inputTempPath=valDataPath;
hogPath='./CL_val/';
lmPath='./CL_LandMarks_val/';
getSASE_FE_val;

in_files = dir(testDataPath);
output = './output_features_test/';
inputTempPath=testDataPath;
hogPath='./CL_test/';
lmPath='./CL_LandMarks_test/';
getSASE_FE_val;
