function [data_g,  data_p] = icp_process_xgtu(data_g,  data_p)

[ data_g, data_p, error, data_pp, R ] = icp_process(data_g, data_p);
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
    [data_g, data_p, error, data_pp, R] = icp_process(data_g, data_p);
    R = last_R * R;
    log_info(strcat('����������', num2str(cnt), '����', num2str(error)));
    log_info('��ǰ��ת����Ϊ��');
    disp(R);
end
end

