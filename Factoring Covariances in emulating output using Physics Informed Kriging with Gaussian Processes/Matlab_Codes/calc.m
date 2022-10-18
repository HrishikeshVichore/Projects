rng('default') % For reproducibility
X = linspace(0,10,100)';
fig = figure;
fig.Position(3) = fig.Position(3)*2;
tiledlayout(1,4,'TileSpacing','compact')
GPR1 = GP1(X,true);
GPR2 = GP2(X,true); 
GPR3 = GP3(X,true);

y = linspace(2,5,10)';
[ypred1,~,yint1] = predict(GPR2,y);
[ypred2,~,yint2] = predict(GPR1,ypred1);
[ypred3,~,yint3] = predict(GPR3,y);

nexttile
hold on
plot(y,ypred2,'g')                   % Composite GPR predictions
plot(y,ypred3,'b')                   % GPR of Composite Functions predictions
fplot(@(x)x.*cos(x).* sin(x.*cos(x)),[2,5],'--r')
%patch([y;flipud(y)],[yint2(:,1);flipud(yint2(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
hold off
title('Comparison')
legend({'GP1(GP2)','GP(f(g(x)))','Actual Output'},'Location','best')



