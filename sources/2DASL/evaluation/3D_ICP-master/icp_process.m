function [ data_g, data_p, err, data_pp, R ] = icp_process( data_g, data_p )
% �������㼯data_g��data_pӦ��ICP�㷨
% data_g\data_P:�����㼯
% ������ת��������㼯���Լ�����ֵ�Լ�data_p��Ӧdata_g�Ķ�Ӧ�㼯data_pp

    [k1, n] = size(data_g);
    [k2, m] = size(data_p);
    
    data_p1 = zeros(k2, 3);     % �м�㼯
    data_pp = zeros(k1, 3);     % ��Ӧ�㼯
    distance = zeros(k1, 1);    % �㼯֮�����ľ���
    error = zeros(k1, 1);       % ��Ӧ��֮������
    
    % �������㼯��ȥ���Ļ�
    data_g = normal_gravity(data_g);
    data_p = normal_gravity(data_p);
    
    % ��������data_g�㼯��Ѱ��ÿ�����Ӧdata_p�㼯�о�����С�ĵ㣬��Ϊ��Ӧ��
    for i = 1:k1
        data_p1(:, 1) = data_p(:, 1) - data_g(i, 1);    % �����㼯�еĵ�x����֮��
        data_p1(:, 2) = data_p(:, 2) - data_g(i, 2);    % �����㼯�еĵ�y����֮��
        data_p1(:, 3) = data_p(:, 3) - data_g(i, 3);    % �����㼯�еĵ�z����֮��
        distance = data_p1(:, 1).^2 + data_p1(:, 2).^2 + data_p1(:, 3).^2;  % ŷ�Ͼ���
        [min_dis, min_index] = min(distance);   % �ҵ�������С���Ǹ���
        data_pp(i, :) = data_p(min_index, :);   % ���Ǹ��㱣��Ϊ��Ӧ��
        error(i) = min_dis;     % ��������ֵ
    end

    % ���Э�������
    V = (data_g' * data_pp) ./ k1;
    
    % ������������Q���ⲿ�ֲ��Ǻ���⣬ֱ���׹�ʽ�ˣ�
    matrix_Q = [V(1,1)+V(2,2)+V(3,3),V(2,3)-V(3,2),V(3,1)-V(1,3),V(1,2)-V(2,1);  
                V(2,3)-V(3,2),V(1,1)-V(2,2)-V(3,3),V(1,2)+V(2,1),V(1,3)+V(3,1);  
                V(3,1)-V(1,3),V(1,2)+V(2,1),V(2,2)-V(1,1)-V(3,3),V(2,3)+V(3,2);  
                V(1,2)-V(2,1),V(1,3)+V(3,1),V(2,3)+V(3,2),V(3,3)-V(1,1)-V(2,2)];
    
    [V2, D2] = eig(matrix_Q);       % �Ծ���Q������ֵ�ֽ�
    lambdas = [D2(1, 1), D2(2, 2), D2(3, 3), D2(4, 4)]; % ȡ������ֵ
    [lambda, ind] = max(lambdas);   % ��������Ǹ�����ֵ
    Q = V2(:, ind); % ȡ���Ǹ���������ֵ����Ӧ����������
    
    % ������ת������Ԫ����
    R=[Q(1,1)^2+Q(2,1)^2-Q(3,1)^2-Q(4,1)^2,     2*(Q(2,1)*Q(3,1)-Q(1,1)*Q(4,1)),        2*(Q(2,1)*Q(4,1)+Q(1,1)*Q(3,1));  
       2*(Q(2,1)*Q(3,1)+Q(1,1)*Q(4,1)),         Q(1,1)^2-Q(2,1)^2+Q(3,1)^2-Q(4,1)^2,    2*(Q(3,1)*Q(4,1)-Q(1,1)*Q(2,1));  
       2*(Q(2,1)*Q(4,1)-Q(1,1)*Q(3,1)),         2*(Q(3,1)*Q(4,1)+Q(1,1)*Q(2,1)),        Q(1,1)^2-Q(2,1)^2-Q(3,1)^2-Q(4,1)^2;  
    ];
    
    % ��data_p�㼯���еĵ㶼��R����ת�仯��Ȼ����������ƽ��
    data_p = data_p * R;
    data_pp = data_pp * R;
    data_p = normal_gravity(data_p);
    data_pp = normal_gravity(data_pp);
    err = mean(error);
    
end

