function ret = normal_gravity( data )
% ���㼯dataȥ���Ļ�
% ������Ϊ��ͬ��������Ϊһ������������

    [m, n] = size(data);
    data_mean = mean(data);
    ret = data - ones(m, 1) * data_mean;

end

