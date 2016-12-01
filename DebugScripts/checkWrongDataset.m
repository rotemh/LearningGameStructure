close all

load('1.mat')
winner_id = winner_train_data{1,1};
if winner_id == 0
    sprintf('winner id is RED')
end
episode_length = size(winner_train_data,1);

figure;imagesc(winner_train_data{1,2})

for idx=1:episode_length
    figure(idx);
    subplot(1,2,1)
    imagesc(winner_train_data{idx,2})
    title(['s',num2str(idx)])
    action = winner_train_data{idx,3}
    subplot(1,2,2)
    imagesc(winner_train_data{idx,4})
    title(['sprime',num2str(idx)])
end