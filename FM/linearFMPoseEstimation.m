% This function estimates the pose of three views based on corresponding
% triplets of points, using linear fundamental matrix estimation
%
% Input:
%
% Output:
%

function [R_t_2,R_t_3,Reconst,T,iter] = LinearFPoseEstimation(Corresp,CalM)