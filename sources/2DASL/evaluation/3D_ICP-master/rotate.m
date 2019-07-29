function new_data = rotate( data, theta )
% ��ԭʼ�㼯data������תtheta�Ƕȣ�����new_data��
% data��ԭʼ�㼯��ÿ�ж���һ�����x��y��z���ꣻ
% theta����ת�ĽǶȣ���x��˳ʱ����תΪ������������תΪ����
% new_data����ת��ĵ㼯��

    theta = - theta * pi / 180; % �Ƕ�תΪ���ȣ�ȡ��ֵ����ʾ������ת
    % ��ת����
    matrix_rotate = [1, 0, 0, 0; 
                     0, cos(theta), sin(theta), 0;
                     0, -sin(theta), cos(theta), 0;
                     0, 0, 0, 1];
    rows = size(data, 1);   % ����һ����data�㼯��Ӧ�ľ���
    row_ones = ones(rows, 1);   % ��1����data�㼯��չ�������ʽ
    new_data = [data, row_ones] * matrix_rotate;    % ������ת����
    new_data = new_data(:, 1:3);    % �������ʽ��ԭ

end

