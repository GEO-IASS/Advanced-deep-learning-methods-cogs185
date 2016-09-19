

data_file = 'data/pos/test.shin'; 
data_file = 'data/pos/train500_train.shin';   

categories = cell(1); 

fid = fopen(data_file)

line = fgetl(fid);
c = 0; 
while ischar(line)
    idx = find(line == ' '); 
    cate = line(1:idx(1)-1); 
    c = c + 1; 
    categories{c} = cate; 
    
    disp([num2str(c) '\t' cate])
    line = fgetl(fid);
end

fclose(fid); 

cates = unique(categories);

% for cate 
idx = find(line == ' '); 
idx2 = find(line == ':'); 
idx3 = find(line == '#'); 
line(idx3-1:end) = []; 
cate = line(1:idx(1)-1)

% for the pos
% line(1:idx) = []; 
pos = line(idx2(1)+1:idx(2)-1) 
idx3 = find(pos == '.'); 
a = str2num(pos(1:idx3-1)); 
b = str2num(pos(idx3+1:end)); 

line(idx2) = ' '; 
feat_line = str2num(line(idx(2)+1:end));
disp(feat_line)

for i = 1 : length(cates)
    if length(cate) == length(cates{i}) & all(cate == cates{i})
        label = i; 
        break; 
    end
end

lhalf = 1:length(feat_line)/2; 
feats(1, (lhalf-1)*2+1) = 1; 
