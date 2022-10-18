function [GPR] = GP1(X,display_graph)
%GP1 Summary of this function goes here
%   Detailed explanation goes here
y = X.*sin(X);
y = y + 0.5*randn(size(X));
GPR = fitrgp(X,y);

if display_graph
    nexttile
    x = linspace(0,10)';
    [ypred,~,yint] = predict(GPR,x);
    hold on
    scatter(X,y,'xr') % Observed data points
    fplot(@(x) x.*sin(x),[0,10],'--r')   % Function plot of x*sin(x)
    plot(x,ypred,'g')                   % GPR predictions
    patch([x;flipud(x)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
    hold off
    title('GPR Fit of f(x)')
    legend({'Noisy observations','f(x) = x*sin(x)','GPR predictions','95% prediction intervals'},'Location','best')

end

