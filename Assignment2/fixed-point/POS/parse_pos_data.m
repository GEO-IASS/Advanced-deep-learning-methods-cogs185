function  [X, labels] = parse_pos_data(data_file, cate_file)
%   parse the data file for pos dataset
%   data_file:          the file name of the data set
%   cate_file:          a file store the the collection of the labels 

load(cate_file)
fid = fopen(data_file); 

line = fgetl(fid);
X = []; 
labels = []; 

c = 0; 
while ischar(line)
    
%    disp(c)
    
    idx = find(line == ' '); 
    idx2 = find(line == ':'); 
    idx3 = find(line == '#'); 
    line(idx3-1:end) = []; 
    cate = line(1:idx(1)-1); 

    pos = line(idx2(1)+1:idx(2)-1); 
    idx3 = find(pos == '.'); 
    a = str2num(pos(1:idx3-1)); 
    b = str2num(pos(idx3+1:end)); 

    line(idx2) = ' '; 
    feat_line = str2num(line(idx(2)+1:end));

    for i = 1 : length(cates)
        if length(cate) == length(cates{i}) & all(cate == cates{i})
            label = i-1; 
            break; 
        end
    end
%    label
    labels = [labels; label]; 

    lhalf = 1:length(feat_line)/2; 
    c = c + 1; 
    
    info = [c, label, b+1, a, b, 0]; 
    
    X(c, 1:6) = info; 
    X(c, 6+feat_line((lhalf-1)*2+1)) = 1; 
    
    if c > 1
        if X(c, 4) ~= X(c-1, 4)
            X(c-1, 3) = -1; 
        end
    end
    
    if c == 1 
        X = sparse(X); 
    end
    
    line = fgetl(fid);
end

X(end, 3) = -1; 
fclose(fid);
end