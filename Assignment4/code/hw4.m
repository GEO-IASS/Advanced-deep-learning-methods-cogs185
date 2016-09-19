BASEDIR = '/Users/yunfanyang/Downloads/CroppedYale/';
% # of face classes to load
CLASSES = 39;
% test size per class
TEST_SZ = 35;
% train size per class
TRAIN_SZ = 40;
% set rnd seed
SEED = 13;
% valid image size
IMG_SIZE = [192, 168];

X_train = {};
Y_train = [];
X_test = {};
Y_test = [];

%%%%%%%%%%%%%%%%%%%%%%%%%     load data   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Loading data..');
% load all the files
dirs = dir(BASEDIR);
% counter for # dirs loaded
ii = 1;
for i = 1 : size(dirs,1)
    PREFIX = 'yaleB';
    % match folder prefix
    if(strncmpi(dirs(i).name, PREFIX, size(PREFIX,2)))
        disp(['Loading ' dirs(i).name '..']);
        files = dir([BASEDIR dirs(i).name]);
        % shuffle files
         rng(SEED);
         files = files(randperm(size(files, 1)));
        
        % counter for # imgs loaded in the subdir
        jj = 1;
        % load images from subfolders
        for j = 1: size(files,1)
            % match pgm files
            if(any(regexp(files(j).name ,'.pgm$')))
                [pic, maxgray] = getpgmraw([BASEDIR '/' dirs(i).name '/' files(j).name]);
                
                % validate image size
                if((size(pic) ~= IMG_SIZE))
                    continue;
                end
                
                if(jj <= TRAIN_SZ)
                    % reshape 2D array to 1D
                    X_train = [X_train;reshape(pic,1,size(pic,1) * size(pic,2))];
                    Y_train = [Y_train;i];
                else
                    % reshape 2D array to 1D
                    X_test = [X_test;reshape(pic,1,size(pic,1) * size(pic,2))];
                    Y_test = [Y_test;i];
                end
                jj = jj + 1;
            end
            
            % limit % of imgs per class to load
            if(jj > TEST_SZ + TRAIN_SZ)
                break;
            end
        end
        disp(['# of images loaded ' num2str(jj)]);
        ii = ii + 1;
    end
    % limit # of classes to load
    if(ii > CLASSES)
        break;
    end
end
disp(['# of classes loaded: ' num2str(ii)]);
disp(['# of training size: ' num2str(size(X_train,1))]);
disp(['# of testing size: ' num2str(size(X_test,1))]);
% unpack cell array to array
X_train = vertcat(X_train{:});
X_test = vertcat(X_test{:});
%%%%%%%%%%%%%%%%%%%%%%%%%     PCA anaylsis   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[COEFF, SCORE, latent, tsquared, explained] = pca(X_train, 'Algorithm', 'svd');
%%%%%%%%%%%%%%%%%%%%%%%%%     Train Classifier   %%%%%%%%%%%%%%%%%%%%%
disp('Benchmark starts..');
ERR_TEST = [];
ERR_TRAIN = [];
ERR_TEST2 = [];
ERR_TRAIN2 = [];
for i = 1: 200
    disp(['# of Eigenvectors used ' num2str(i)]);
    
    meData = X_train-(ones(size(X_train,1), 1) * mean(X_train));
    X_train_prj = meData * COEFF(:, 1:i);

    meData = X_test-(ones(size(X_test,1), 1) * mean(X_test));
    X_test_prj = meData * COEFF(:, 1:i);

    % Multiclass SVM
    % Mdl = fitcecoc(X_train_prj, Y_train);

    % KNN
     Mdl = fitcknn(X_train_prj, Y_train, 'NumNeighbors',1,'Standardize',1);

    %disp('Training error rate: ')
    %ERR_TRAIN(i) = sum(predict(Mdl, X_train_prj) ~= Y_train) / size(Y_train, 1);
    %disp(ERR_TRAIN(i));
    
    disp('Testing error rate: ')
    ERR_TEST(i) = sum(predict(Mdl, X_test_prj) ~= Y_test) / size(Y_test, 1);
    disp(ERR_TEST(i));
end

%plot(ERR_TRAIN, 'LineWidth',3);
hold on;
plot(ERR_TEST, 'LineWidth',3);


