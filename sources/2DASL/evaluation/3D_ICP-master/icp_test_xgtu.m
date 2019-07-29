%%
clear;
close all;
clc;

data_g = load('./NME_3dFR/andrew_j_feustel_0001.mat');
data_g = double(data_g.vertex');
data_p = load('./NME_3dFR/andrew_j_feustel_0002.mat');
data_p = double(data_p.vertex');
save('ori_vertexes', 'data_g', 'data_p')
%%
% data_g = load('face1.asc');     % ����face1�ĵ㼯
% data_p = rotate(data_g, 60);    % ��face1�ĵ㼯������ת20�ȣ���ʾΪface2
save_3d_data('face2_xgtu.txt', data_p);

plot_3d_2(data_g, data_p, -90); % ��ʾ����ǰ�����㼯

%%
[ data_g, data_p, error, data_pp, R ] = icp_process( data_g, data_p );
log_info(strcat('����������1����', num2str(error)));
log_info('��ǰ��ת����Ϊ��');
disp(R);

cnt = 1;
last_error = 0;
last_R = R;
% ���������ʱ��ֹͣѭ��
while abs(error - last_error) > 0.01
    cnt = cnt + 1;
    last_error = error;
    last_R = R;
    [ data_g, data_p, error, data_pp, R ] = icp_process( data_g, data_p );
    R = last_R * R;
    log_info(strcat('����������', num2str(cnt), '����', num2str(error)));
    log_info('��ǰ��ת����Ϊ��');
    disp(R);
end

plot_3d_2(data_g, data_p, -90);
save('icp_vertexes', 'data_g', 'data_p')

%%
% data_g = load('face1.asc');
% data_q = load('face3.txt');
% 
% [ data_g, data_q, error, data_qq, R ] = icp_process( data_g, data_q );
% log_info(strcat('����������1����', num2str(error)));
% log_info('��ǰ��ת����Ϊ��');
% disp(R);

% cnt = 1;
% last_error = 0;
% last_R = R;
% % ���������ʱ��ֹͣѭ��
% while abs(error - last_error) > 0.01
%     cnt = cnt + 1;
%     last_error = error;
%     last_R = R;
%     [ data_g, data_q, error, data_qq, R ] = icp_process( data_g, data_q );
%     R = last_R * R;
%     log_info(strcat('����������', num2str(cnt), '����', num2str(error)));
%     log_info('��ǰ��ת����Ϊ��');
%     disp(R);
% end
% 
% plot_3d_2(data_g, data_q, -90);
