% Author: Lijiang Guo
% Date: Feb 10, 2017
% This function implements the Sobel Operator for edge detection in images.
% Input: 
%   img: greyscale image, 
%   a: threshold for x dimension; high value is more sensitive to edges;
%   b: threshold for y dimension; high value is more sensitive to edges;
%   dx: filter size in x dimension, 
%   dy: filter size in y dimension,
%
% Output: 
%   img_xy: image with pixel value equals norm of gradient.
%   img_x: image with pixel value equals x direction derivative.
%   img_y: image with pixel value equals y direction derivative.
%   img_dir: direction of gradient in radiant.

%%
function [img_xy, img_x, img_y, img_dir] = sobel(img, a, b)
[m,n] = size(img);
% regular sobel kernel
dG_x = a*[-1 0 1; -2 0 2; -1 0 1]/8;
% improved sobel kernel for minimizing orientation error
dG_x = a*[-3 0 3; -10 0 10; -3 0 3]/32;
dG_y = -b*dG_x'/8;
d_G_x = reshape(dG_x,9,1);
d_G_y = reshape(dG_y,9,1);

%% Construct local patch matrix
mar = 1;
X = zeros(9,(m-2*mar)*(n-2*mar));
count = 1;
for i= 1+mar : m-mar
    for j = 1 + mar : n-mar
        X(:,count) = reshape(img(i-mar:i+mar,j-mar:j+mar),9,1);
        count = count + 1;
    end
end

%% Running Sobel operator
% sobel going x direction alone
img_x = reshape(X'*d_G_x, n-2*mar, m-2*mar)';
% sobel going y direction alone
img_y = reshape(X'*d_G_y, n-2*mar, m-2*mar)';
% Norm of gradient
img_xy = (img_x.^2 + img_y.^2).^(1/2);
% Direction of gradient
img_dir = atan(img_y./img_x);

%% Plots
figure;
imshow(img_x);
title('x direction');
colorbar;
%
figure;
imshow(img_y);
title('y direction');
colorbar;

%
figure;
imshow(img_xy);
title('gradient norm');
colorbar;

end

