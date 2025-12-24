
% vs 0.0.0.2
% 2013-08-24

% SPYC tests

% Generate a random sparse matrix and run a battery of tests

clear all
clc
close all

N = 100;
f = 0.01;
zmax=1000;
A = zeros(N);
pts=unique(round((N-1)*rand(round(N^2*f),2))+1,'rows');
s = zmax*rand(length(pts),1);
ns = length(s);
[iA jA] = size(A);
sA = sparse(pts(:,1),pts(:,2),s,iA,jA);


% run test battery

% disp('Press RET to start tests...')
% pause
% disp('spy(sA)')
figure
spy(sA)
% pause
disp('spyc(sA,[])')
spyc(sA,[]);
% pause
disp('spyc(sA,[],[]);')
spyc(sA,[],[]);
% pause

cmap = 'summer';
disp('spyc(sA,cmap);')
spyc(sA,cmap);
% pause
disp('spyc(sA,cmap,[]);')
spyc(sA,cmap,[]);
% pause
disp('spyc(sA,cmap,0);')
spyc(sA,cmap,0);
pause
disp('spyc(sA,cmap,1);')
spyc(sA,cmap,1);
pause

cmap = 'winter';
cmap = flipud(colormap(cmap));
disp('spyc(sA,cmap);')
spyc(sA,cmap);
pause
disp('spyc(sA,cmap,[]);')
spyc(sA,cmap,[]);
pause
disp('spyc(sA,cmap,0);')
spyc(sA,cmap,0);
pause
disp('spyc(sA,cmap,1);')
spyc(sA,cmap,1);
pause

disp('done!')

% any of the following should produce error
% spyc;
% spyc();
% spyc(sA,0,[]);
% spyc([],[],[]);

return

