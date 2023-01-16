clc;
clear;
%% 
path='C:\Users\DELL\Desktop\BlochSim\mapping_130_1001\';
all_img = dir(path);

% 读取dicom数据
for k = 3:size(all_img)
    info = dicominfo([path,all_img(k).name]);
    ac_time(k-2)=str2double(info.AcquisitionTime);
    img(:,:,k-2) = double(dicomread([path,all_img(k).name]));
end

% figure,
% for i=1:size(img,3)
%     subplot(3,3,i)
%     imshow(img(:,:,i),[])
% end

acq_Durations = zeros(1,8);
for t = 1:7
    acq_Durations(t) = (ac_time(t+1)-ac_time(t))*1000;
end
acq_Durations(1,8) = acq_Durations(1,7);
TI = info.InversionTime;
TR = info.RepetitionTime;
ex_num = info.EchoTrainLength;

%% 
load("img_0911_mapping.mat");
img1 = img_0911_mapping;

img_norm=zeros(size(img1,1),size(img1,2),size(img1,3));
for i=1:size(img1,1)
    for j=1:size(img1,2)
        img_norm(i,j,:)=img1(i,j,:)/norm(squeeze(img1(i,j,:)));
    end
end
%% 

figure,
for i=1:size(img_norm,3)
    subplot(3,3,i),imshow(img_norm(:,:,i),[])
end
%% 

% [t1map,M0map,alphamap,inverfmap] = t1fit(img_norm,acq_Durations,TI,TR,ex_num);
%% 执行fit函数

m=size(img_norm,1);
n=size(img_norm,2);
t1map=zeros(m,n);
M0map=zeros(m,n);
alphamap=zeros(m,n);
inverfmap=zeros(m,n);

parfor i = 1:m
    for j = 1:n
        if(mod(i*j+j,1000)==0)
        disp(i*j+j);
        end
        signal=squeeze(img_norm(i,j,:))';
        smax=max(abs(signal));
        m0_initial=10*smax;
        [t1fit,M0fit,alphafit,inverfit,resnorm] = t1fit_onepx2(signal,acq_Durations,TI,TR,ex_num,m0_initial);  
        t1map(i,j)=t1fit;
        M0map(i,j)=M0fit;
        alphamap(i,j)=alphafit;
        inverfmap(i,j)=inverfit;
    end
end

%% 
a=[0,3000];
figure,subplot(2,2,1)
imshow(t1map,a)
colorbar,caxis(a),colormap(gca,'jet')
set(gca,'fontsize',15)
title('t1fit')

a=[min(min(M0map)) max(max(M0map))];
subplot(2,2,2)
imshow(M0map,a)
colorbar,caxis(a),colormap(gca,'jet')
set(gca,'fontsize',15)
title('M0 fit')

a=[0 20];
subplot(2,2,3)
imshow(alphamap,a)
colorbar,caxis(a),colormap(gca,'jet')
set(gca,'fontsize',15)
title('alpha fit')

a=[0 1];
subplot(2,2,4)
imshow(inverfmap,a)
colorbar,caxis(a),colormap(gca,'jet')
set(gca,'fontsize',15)
title('inverfactor fit')

