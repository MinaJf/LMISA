clear all;
close all;
tic
folder2='D:\TensorFlow\Data\WristXray\GM_Sum\CT\validationCT\';
pathDir1='D:\TensorFlow\Data\WristXray\CT\validationCT\';
Files=dir([pathDir1 '*_org.nii.gz']);
for k=1:length(Files)

    FileNames=Files(k).name;
    
    Si = load_nii([pathDir1 FileNames]);
    I = Si.img;


    % I=imread('.\Medical_05.jpg');
    % I=rgb2gray(I);
%     figure,imshow(I(:,:,10),[]);
    [GI,Gazimuth,Gelevation] = imgradient3(I);
    % figure,imshow(GI,[]); title('Global gradient magnitude');
    %
    % eI=edge(I,'canny'); % low high threshold
    % figure,imshow(eI,[]); title('Canny edge');
    
    % eI=edge(I,'approxcanny',[0.08 0.2]); % low high threshold
    % figure,imshow(eI,[]); title('Canny edge');
    
    nLevel=4;  % multi-level
    imgScale=1./(2.^[0:nLevel-1]);
    hW=1;       % window size for normalisation
    mulGM=zeros(size(I));
    mulEdge=zeros(size(I));
    
    for i=1:nLevel
        tmpI=imresize3(I,imgScale(i));
        [GtmpI,~,~]=imgradient3(tmpI);
        stdI=std(GtmpI(:));
        norm_GM=zeros(size(GtmpI));
        for r=hW+1:size(GtmpI,1)-hW
            for c=hW+1:size(GtmpI,2)-hW
                for s=1:size(GtmpI,3)
                    tmpI=GtmpI(r-hW:r+hW,c-hW:c+hW, s);

                    if std(tmpI(:))<stdI/1000  % threshold to remove homogenous regions
                        norm_GM(r,c, s)=0;
                    else
                        tmpI=(tmpI-min(tmpI(:)))/(max(tmpI(:))-min(tmpI(:)));
                        norm_GM(r,c, s)=tmpI(hW+1,hW+1);
                    end
                end
            end
        end
%         figure,imshow(norm_GM(:,:,10),[]);
        mulGM=mulGM + imresize3(norm_GM,[size(I,1), size(I,2), size(I,3)]);
    end
    
    mulGM=(mulGM-min(mulGM(:)))/(max(mulGM(:))-min(mulGM(:)));
%     figure,imshow(mulGM(:,:,10),[]); title('Locally normalised multi-resolution gradient magnitude');
    
toc
    FileNamesN = erase(FileNames,'.gz');
    name_img=[folder2 FileNamesN];
    niftiwrite(double(mulGM), name_img)
    gzip(name_img)
end

