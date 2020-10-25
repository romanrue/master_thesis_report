close all; clear all; clc;

[mdir, filename, ~] = fileparts(mfilename('fullpath'));
pdir = fileparts(mdir);

%%%------------------------------------------------------------------------
%%% OUPTUT FILES
figdir = 'fig';
%filename = '';

%%%------------------------------------------------------------------------
%%% Parameters
saveplots = true;
showpreview = true;    % not very accuarte preview (open emf to see result)
aspRatio = 4/3;         % ratio x-axis over y-axis
keepAspRatio = true;
axislables = ['t','x'];
axisOverlap = [0 .14 .14 .14];  % [-] x-axis normalized
frameSep = 0.08*ones(1,4);      % relative value to width
width = 2.5;        % [cm]
height = 2.5;       % [cm] iff ~keepAspectratio
arrowLength = 0.05; % relative to width
arrowAng = 35;      % [deg]
plotLineWidth = 1.2;
axisLineWidth = 0.9;
frameLineWidth = 0.9;

%%%------------------------------------------------------------------------
%%% FUNCTION
x = linspace(0,4*pi,80);
y = wgn(80,1,0);

%%%------------------------------------------------------------------------
%%% CHANGE Y TO GET DESIRED ASPECT RATIO
[oXlim, oXdif] = outerXlim([x(1),x(end)],axisOverlap(1:2),frameSep(1:2));
oYdif = oXdif/aspRatio;
iYdif = oYdif - sum([axisOverlap(3:4); frameSep(3:4)],'all') * oXdif;

% function dependent
y = y * iYdif/(2*max(abs(y)));
oYlim = outerYlim([min(y), min(y) + oYdif - sum([axisOverlap(3:4);frameSep(3:4)],'all') * oXdif], ...
    oXdif,axisOverlap(3:4),frameSep(3:4));

%%%------------------------------------------------------------------------
%%% COLORS
pttColors = cell(1,12);
pttColors{ 1} = [  0,   0,   0]/255;
pttColors{ 2} = [255, 255, 255]/255;
pttColors{ 3} = [114, 121,  28]/255;
pttColors{ 4} = [ 18, 105, 176]/255;
pttColors{ 5} = [145,   5, 106]/255;
pttColors{ 6} = [111, 111, 100]/255;
pttColors{ 7} = [168,  50,  45]/255;
pttColors{ 8} = [  0, 122, 150]/255;
pttColors{ 9} = [149,  96,  19]/255;
pttColors{10} = [255, 255, 255]/255;
pttColors{11} = [ 18, 105, 176]/255;
pttColors{12} = [140, 182,  60]/255;


%%%------------------------------------------------------------------------
%%% PLOT
if keepAspRatio
    height = width/aspRatio;
end

figure()
if ~showpreview
    set(gcf, 'Visible', 'off');
end
plot(x,y,'LineWidth',plotLineWidth,'Color',pttColors{4});
for i = 1:2
    ah(i) = annotation('arrow','HeadStyle','plain', ...
        'LineWidth',axisLineWidth,'HeadLength',6*oXdif*arrowLength, ...
        'HeadWidth',2*6*oXdif*arrowLength*tand(arrowAng/2));
    lh(i) = annotation('textbox','String',axislables(i), ...
        'FontName','Arial','FontSize',14,'LineStyle','none');
    set(ah(i), 'parent',gca);
    set(lh(i), 'parent',gca);
    if i == 1
        posmat = [(oXlim + [1 -1].*frameSep(1:2)*oXdif); zeros(1,2)];
        set(ah(i), 'position',[posmat(1:2),diff(posmat,1,2).']);
        set(lh(i), 'position',[posmat(3:4)+[-0.15, 0.32]*oXdif,0.1,0.1]);
    else
        posmat = [zeros(1,2); (oYlim + [1 -1].*frameSep(3:4)*oXdif)];
        set(ah(i), 'position',[posmat(1:2),diff(posmat,1,2).']);
        set(lh(i), 'position',[posmat(3:4)+[0.05,0.15]*oXdif,0.1,0.1]);
    end
end
h = get(gca, 'Children');
set(gca, 'Children', [h(1),h(3),h(5),h(2),h(4)]);

xlim(oXlim);
ylim(oYlim);
axis equal;
set(gca,'XTick',[],'YTick',[],'LineWidth',axisLineWidth);
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0,0,width,height]);

if saveplots
    saveas(gcf,fullfile(pdir,figdir,filename),'meta');   % emf for MS-Office
end

%%%------------------------------------------------------------------------
%%% FUNCTIONS
function varargout = outerXlim(iXlim, xAxOverlap, xFrameSep)
    oXlim = zeros(1,2);
    oXdif = diff(iXlim)/(1-sum(xAxOverlap)-sum(xFrameSep));
    oXlim(1) = iXlim(1) - (xFrameSep(1) + xAxOverlap(1)) * oXdif;
    oXlim(2) = iXlim(2) + (xFrameSep(2) + xAxOverlap(2)) * oXdif;
    if nargout > 0
        varargout{1} = oXlim;
    end
    if nargout > 1
        varargout{2} = oXdif;
    end    
end

function oYlim = outerYlim(iYlim, oXdif, yAxOverlap, yFrameSep)
    oYlim = iYlim + sum(repmat([-1 1],2,1).* [yAxOverlap; yFrameSep]) * oXdif;
end
