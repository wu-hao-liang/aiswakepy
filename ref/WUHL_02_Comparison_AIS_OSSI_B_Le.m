% Compare results from AIS data processing using empirical formulation to
% measurements
% CHAP DHI-SG
% 26/05/2023

clear all; close all; clc;

% Load toolboxes

addpath(genpath('\\zeus\DHI Softwares\PROCESSING tools\MATLAB\Toolboxes\wafo26'))
addpath(genpath('\\zeus\DHI Softwares\PROCESSING tools\MATLAB\Toolboxes\m_tools\m_tools_20201012'))
addpath(genpath('\\zeus\DHI Softwares\PROCESSING tools\MATLAB\Toolboxes\DHIMatlabToolbox\DHIMatlabToolbox-Mz2020'))
addpath(genpath('\\sg-ncr04\projects\61802983 JI SSES\MATLAB\0_Scripts\ShipwakeCalculation\toolboxes\WaveProcessing_ChapTool')) % toolbox for wave processing

NET.addAssembly('DHI.Mike.Install');
import DHI.Mike.Install.*;
DHI.Mike.Install.MikeImport.SetupLatest({DHI.Mike.Install.MikeProducts.MikeCore});
NET.addAssembly('DHI.Generic.MikeZero.DFS');
NET.addAssembly('DHI.Generic.MikeZero.EUM');
import DHI.Generic.MikeZero.*
import DHI.Generic.MikeZero.DFS.*;
import DHI.Generic.MikeZero.DFS.dfs0.*;
import DHI.Generic.MikeZero.DFS.dfs123.*;


OutPlots = '\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\Plot_comparison_measurements_WUHL_B_Le\';
mkdir(OutPlots);

clean_draft = 1 ;

%% load processed AIS data by CHAP
AIS = load('\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\MATFILES\1_PROCESSED\AISproc_OSSI_EmpiricalFormula_WUHL_B_Le.mat');
            
% USarmyfile = xlsread('\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\4_Final\Shipwake\OSSI\AISproc_OSSI_EmpiricalFormula_USarmyTool.xlsx');

%% Load measurements

meas = xlsread('\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\4_Final\Shipwake\OSSI\ShipWake_peaks_event3minutes_minHeight5cm.xlsx','SHIPWAKE');

OSSI.time = meas(:,2);
OSSI.Hmax = meas(:,3);
OSSI.T = meas(:,5);

clear meas


%% Clean data with draft = 0

if clean_draft == 1;
    idx_draft0 = find(AIS.draught == 0);
    
    AIS.Kriebel.Hmax(idx_draft0) = NaN ;
    AIS.PIANC.Hmax(idx_draft0) = NaN ;
    AIS.Sorensen.Hmax(idx_draft0) = NaN ;
    Maynord.Hmax(idx_draft0) = NaN ;
    AIS.Bhowmik.Hmax(idx_draft0) = NaN ;
    AIS.Gates.Hmax(idx_draft0) = NaN ;
    AIS.Blaauw.Hmax1(idx_draft0) = NaN ;
    AIS.Blaauw.Hmax2(idx_draft0) = NaN ;
    AIS.Blaauw.Hmax3(idx_draft0) = NaN ;
end

idf_badpianc = find( AIS.Fr >= 0.7 ) ;
AIS.PIANC.Hmax(idf_badpianc) = NaN ;

%% Plots everything for order of magnitude
cd(OutPlots);
% htmlGray = [0.7 0.7 0.7];
% plot(x, y, 'Color', grayColor)

darkgreen = 1/255*[0,104,87];
gray = 1/255*[200,200,200];
orange = [0.8500, 0.3250, 0.0980];
purple = [0.75, 0, 0.75];
olive = [0.75, 0.75, 0];

figure('units','normalized','outerposition',[0 0 1 1])
% g1=subplot(2,1,1);
hold on
scatter(OSSI.time,OSSI.Hmax,20, 'k','filled')
scatter(AIS.time,AIS.Kriebel.Hmax,20,'r','filled');
scatter(AIS.time,AIS.PIANC.Hmax,20,'b','filled');
scatter(AIS.time,AIS.Sorensen.Hmax,20,'g','filled');
scatter(AIS.time,AIS.Maynord.Hmax,20,'m','filled');
scatter(AIS.time,AIS.Bhowmik.Hmax,20,'MarkerEdgeColor',darkgreen,'MarkerFaceColor',darkgreen);
scatter(AIS.time,AIS.Gates.Hmax,20,'MarkerEdgeColor',gray,'MarkerFaceColor',gray);
scatter(AIS.time,AIS.Blaauw.Hmax1,20,'MarkerEdgeColor',orange,'MarkerFaceColor',orange);
scatter(AIS.time,AIS.Blaauw.Hmax2,20,'MarkerEdgeColor',purple,'MarkerFaceColor',purple);
scatter(AIS.time,AIS.Blaauw.Hmax3,20,'MarkerEdgeColor',olive,'MarkerFaceColor',olive);
% scatter(AIS.time,AIS.BhowmikScaled.Hmax,20,'MarkerEdgeColor',color1);
xData = linspace(OSSI.time(1),OSSI.time(end),8);
ax = gca;
ax.XTick = xData;
datetick('x','dd-mmm HH:MM','keepticks')
title(['Point 1' ]);
ylabel('H_{max} (m)');
xlabel('Time');
% legend('Measurements','Kriebel (2005)','PIANC (1987)','Sorensen (1984)','Maynord (2005)','Bhowmik (1982)','Bhowmik (1982) scaled');
legend('Measurements','Kriebel (2005)','PIANC (1987)','Sorensen (1984)','Maynord (2005)','Bhowmik (1982)','Gates (1977)','Blaauw (1985) 1','Blaauw (1985) 2','Blaauw (1985) 3');
grid on



%% Find common events

event_window = 0.5 ;  % in minutes

for i = 1: length(AIS.time);
    
    event = AIS.time(i);
    
    indx_ossi = find(OSSI.time < event + event_window/60/24 & OSSI.time > event - event_window/60/24);
    
    if isempty(indx_ossi) == 1 ;
        AIS.CorrespondingEvent.time(i,1) = NaN;
        AIS.CorrespondingEvent.Hmax(i,1) = NaN;
    elseif length(indx_ossi) > 1;
        AIS.CorrespondingEvent.time(i,1) = NaN;
        AIS.CorrespondingEvent.Hmax(i,1) = NaN;
    else
        AIS.CorrespondingEvent.time(i,1) = OSSI.time(indx_ossi);
        AIS.CorrespondingEvent.Hmax(i,1) = OSSI.Hmax(indx_ossi);;
        
    end
end

% perfect fit
x = [0 1];
y = [0 1];

figure('units','normalized','outerposition',[0 0 1 1])
hold on
scatter(AIS.time,AIS.CorrespondingEvent.Hmax,20, 'k','filled')
scatter(AIS.time,AIS.Kriebel.Hmax,20,'MarkerEdgeColor',olive,'MarkerFaceColor',olive);
scatter(AIS.time,AIS.PIANC.Hmax,20,'b','filled');
scatter(AIS.time,AIS.Sorensen.Hmax,20,'g','filled');
% scatter(AIS.time,AIS.Maynord.Hmax,20,'MarkerEdgeColor',purple,'MarkerFaceColor',purple);
scatter(AIS.time,AIS.Bhowmik.Hmax,20,'MarkerEdgeColor',darkgreen,'MarkerFaceColor',darkgreen);
scatter(AIS.time,AIS.Gates.Hmax,20,'MarkerEdgeColor',gray,'MarkerFaceColor',gray);
scatter(AIS.time,AIS.Blaauw.Hmax1,20,'MarkerEdgeColor',orange,'MarkerFaceColor',orange);
% scatter(AIS.time,AIS.Blaauw.Hmax2,20,'MarkerEdgeColor',purple,'MarkerFaceColor',purple);
% scatter(AIS.time,AIS.Blaauw.Hmax3,20,'MarkerEdgeColor',olive,'MarkerFaceColor',olive);
% scatter(AIS.time,AIS.BhowmikScaled.Hmax,20,'MarkerEdgeColor',color1);
xData = linspace(AIS.time(1),AIS.time(end),8);
ax = gca;
ax.XTick = xData;
datetick('x','dd-mmm HH:MM','keepticks')
title(['Point 1' ]);
ylabel('H_{max} (m)');
xlabel('Time');
% legend('Measurements (common events)','Kriebel (2005)','PIANC (1987)','Sorensen (1984)','Maynord (2005)','Bhowmik (1982)','Bhowmik (1982) scaled');
legend('Measurements (common events)','Kriebel (2005)','PIANC (1987)','Sorensen (1984)','Bhowmik (1982)','Gates (1977)','Blaauw (1985)');%,'Blaauw (1985) 2','Blaauw (1985) 3');
grid on
ylim([0 2]);
print('OrderOfMagn_AIS_measurements_Hmax_CommonEvents_window15sec.png','-dpng');




figure;
hold on
grid on
scatter(AIS.CorrespondingEvent.Hmax,AIS.Kriebel.Hmax,20,'MarkerEdgeColor',olive,'MarkerFaceColor',olive);
scatter(AIS.CorrespondingEvent.Hmax,AIS.PIANC.Hmax,20,'b','filled');
scatter(AIS.CorrespondingEvent.Hmax,AIS.Sorensen.Hmax,20,'MarkerEdgeColor','m','MarkerFaceColor','m');
% scatter(AIS.CorrespondingEvent.Hmax,AIS.Maynord.Hmax,20,'MarkerEdgeColor',purple,'MarkerFaceColor',purple);
scatter(AIS.CorrespondingEvent.Hmax,AIS.Bhowmik.Hmax,20,'MarkerEdgeColor',darkgreen,'MarkerFaceColor',darkgreen);
% scatter(AIS.CorrespondingEvent.Hmax,AIS.BhowmikScaled.Hmax,20,'k');
scatter(AIS.CorrespondingEvent.Hmax,AIS.Gates.Hmax,20,'MarkerEdgeColor',gray,'MarkerFaceColor',gray);
scatter(AIS.CorrespondingEvent.Hmax,AIS.Blaauw.Hmax1,20,'MarkerEdgeColor',orange,'MarkerFaceColor',orange);
% scatter(AIS.CorrespondingEvent.Hmax,AIS.Blaauw.Hmax2,20,'MarkerEdgeColor',purple,'MarkerFaceColor',purple);
% scatter(AIS.CorrespondingEvent.Hmax,AIS.Blaauw.Hmax3,20,'MarkerEdgeColor',olive,'MarkerFaceColor',olive);
plot(x,y,'--k')
% legend('Kriebel','PIANC','Sorensen','Maynord','Bhowmik','Bhowmik scaled');
% legend('Kriebel','PIANC','Sorensen','Bhowmik','Bhowmik scaled');
% legend('Kriebel','PIANC','Sorensen','Maynord','Bhowmik','Gates','Blaauw');%,'Blaauw 2','Blaauw 3');
% legend('Kriebel (2005)','PIANC (1987)','Sorensen (1984)','Maynord (2005)','Bhowmik (1982)','Bhowmik SCALED(1982)','Gates (1977)','Blaauw (1985)');%,'Blaauw (1985) 2','Blaauw (1985) 3');
legend('Kriebel (2005)','PIANC (1987)','Sorensen (1984)','Bhowmik (1982)','Gates (1977)','Blaauw (1985)');%,'Blaauw (1985) 2','Blaauw (1985) 3');
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
title(['Empirical formulation vs measurements - missing draft removed'])% - window to select events ' num2str(event_window) ' min']);
ylim([0 0.5]);
xlim([0 0.5]);
print(['Scatter_AIS_measurements_Window_' num2str(event_window) 'min_NoDraft0.png'],'-dpng');



% %% Write an excel file with all results 
% xlsname = '\\sg-ncr04\projects\61802983 JI SSES\MATLAB\4_Final\Shipwake\OSSI\AISproc_OSSI_EmpiricalFormula.xlsx';
% xlswrite(xlsname,AIS.vesselNo,'Sheet1','A2');
% xlswrite(xlsname,AIS.mmsi,'Sheet1','B2');
% xlswrite(xlsname,AIS.time,'Sheet1','C2');
% xlswrite(xlsname,AIS.width,'Sheet1','D2');
% xlswrite(xlsname,AIS.L,'Sheet1','E2');
% xlswrite(xlsname,AIS.draught,'Sheet1','F2');
% xlswrite(xlsname,AIS.lon,'Sheet1','G2');
% xlswrite(xlsname,AIS.lat,'Sheet1','H2');
% xlswrite(xlsname,AIS.dist,'Sheet1','I2');
% xlswrite(xlsname,AIS.sog,'Sheet1','J2');
% xlswrite(xlsname,AIS.waterdepth,'Sheet1','K2');
% xlswrite(xlsname,AIS.CorrespondingEvent.Hmax,'Sheet1','L2');
% xlswrite(xlsname,AIS.Kriebel.Hmax,'Sheet1','M2');
% xlswrite(xlsname,AIS.PIANC.Hmax,'Sheet1','N2');
% xlswrite(xlsname,AIS.Sorensen.Hmax,'Sheet1','O2');
% xlswrite(xlsname,AIS.Blaauw.Hmax1,'Sheet1','P2');
% xlswrite(xlsname,AIS.Bhowmik.Hmax,'Sheet1','Q2');
% xlswrite(xlsname,AIS.Gates.Hmax,'Sheet1','R2');
% xlswrite(xlsname,AIS.W,'Sheet1','S2');


%% Prepare results for dfs0

indx_k = find(isnan(AIS.CorrespondingEvent.Hmax) == 0 & isnan(AIS.Kriebel.Hmax) == 0);
Kriebel.meas = AIS.CorrespondingEvent.Hmax(indx_k);
Kriebel.formula = AIS.Kriebel.Hmax(indx_k);
Kriebel.L = AIS.L(indx_k);
Kriebel.sog = AIS.sog(indx_k);
Kriebel.draught = AIS.draught(indx_k);
Kriebel.Fr = AIS.Fr(indx_k);
Kriebel.Fr_mod = AIS.Fr_mod(indx_k);
Kriebel.dist = AIS.dist(indx_k);
Kriebel.W = AIS.W(indx_k);
Kriebel.time = datenum('14-Nov-2022 08:06:13'):(10/60/24):(datenum('14-Nov-2022 08:06:13')+((length(Kriebel.meas)-1)*(10/60/24))); Kriebel.time = Kriebel.time';

indx_p = find(isnan(AIS.CorrespondingEvent.Hmax) == 0 & isnan(AIS.PIANC.Hmax) == 0);
PIANC.meas = AIS.CorrespondingEvent.Hmax(indx_p);
PIANC.formula = AIS.PIANC.Hmax(indx_p);
PIANC.L = AIS.L(indx_p);
PIANC.sog = AIS.sog(indx_p);
PIANC.draught = AIS.draught(indx_p);
PIANC.Fr = AIS.Fr(indx_p);
PIANC.Fr_mod = AIS.Fr_mod(indx_p);
PIANC.dist = AIS.dist(indx_p);
PIANC.W = AIS.W(indx_p);
PIANC.time = datenum('14-Nov-2022 08:06:13'):(10/60/24):(datenum('14-Nov-2022 08:06:13')+((length(PIANC.meas)-1)*(10/60/24))); PIANC.time = PIANC.time';

indx_s = find(isnan(AIS.CorrespondingEvent.Hmax) == 0 & isnan(AIS.Sorensen.Hmax) == 0);
Sorensen.meas = AIS.CorrespondingEvent.Hmax(indx_s);
Sorensen.formula = AIS.Sorensen.Hmax(indx_s);
Sorensen.L = AIS.L(indx_s);
Sorensen.sog = AIS.sog(indx_s);
Sorensen.draught = AIS.draught(indx_s);
Sorensen.Fr = AIS.Fr(indx_s);
Sorensen.Fr_mod = AIS.Fr_mod(indx_s);
Sorensen.dist = AIS.dist(indx_s);
Sorensen.W = AIS.W(indx_s);
Sorensen.time = datenum('14-Nov-2022 08:06:13'):(10/60/24):(datenum('14-Nov-2022 08:06:13')+((length(Sorensen.meas)-1)*(10/60/24))); Sorensen.time = Sorensen.time';

indx_m = find(isnan(AIS.CorrespondingEvent.Hmax) == 0 & isnan(AIS.Maynord.Hmax) == 0);
Maynord.meas = AIS.CorrespondingEvent.Hmax(indx_m);
Maynord.formula = AIS.Maynord.Hmax(indx_m);
Maynord.L = AIS.L(indx_m);
Maynord.sog = AIS.sog(indx_m);
Maynord.draught = AIS.draught(indx_m);
Maynord.Fr = AIS.Fr(indx_m);
Maynord.Fr_mod = AIS.Fr_mod(indx_m);
Maynord.dist = AIS.dist(indx_m);
Maynord.W = AIS.W(indx_m);
Maynord.time = datenum('14-Nov-2022 08:06:13'):(10/60/24):(datenum('14-Nov-2022 08:06:13')+((length(Maynord.meas)-1)*(10/60/24))); Maynord.time = Maynord.time';

indx_b = find(isnan(AIS.CorrespondingEvent.Hmax) == 0 & isnan(AIS.Bhowmik.Hmax) == 0);
Bhowmik.meas = AIS.CorrespondingEvent.Hmax(indx_b);
Bhowmik.formula = AIS.Bhowmik.Hmax(indx_b);
Bhowmik.L = AIS.L(indx_b);
Bhowmik.sog = AIS.sog(indx_b);
Bhowmik.draught = AIS.draught(indx_b);
Bhowmik.Fr = AIS.Fr(indx_b);
Bhowmik.Fr_mod = AIS.Fr_mod(indx_b);
Bhowmik.dist = AIS.dist(indx_b);
Bhowmik.W = AIS.W(indx_b);
Bhowmik.time = datenum('14-Nov-2022 08:06:13'):(10/60/24):(datenum('14-Nov-2022 08:06:13')+((length(Bhowmik.meas)-1)*(10/60/24))); Bhowmik.time = Bhowmik.time';

indx_g = find(isnan(AIS.CorrespondingEvent.Hmax) == 0 & isnan(AIS.Gates.Hmax) == 0);
Gates.meas = AIS.CorrespondingEvent.Hmax(indx_g);
Gates.formula = AIS.Gates.Hmax(indx_g);
Gates.L = AIS.L(indx_g);
Gates.sog = AIS.sog(indx_g);
Gates.draught = AIS.draught(indx_g);
Gates.Fr = AIS.Fr(indx_g);
Gates.Fr_mod = AIS.Fr_mod(indx_g);
Gates.dist = AIS.dist(indx_g);
Gates.W = AIS.W(indx_g);
Gates.time = datenum('14-Nov-2022 08:06:13'):(10/60/24):(datenum('14-Nov-2022 08:06:13')+((length(Gates.meas)-1)*(10/60/24))); Gates.time = Gates.time';

indx_bl = find(isnan(AIS.CorrespondingEvent.Hmax) == 0 & isnan(AIS.Blaauw.Hmax1) == 0);
Blaauw.meas = AIS.CorrespondingEvent.Hmax(indx_bl);
Blaauw.formula = AIS.Blaauw.Hmax1(indx_bl);
Blaauw.L = AIS.L(indx_bl);
Blaauw.sog = AIS.sog(indx_bl);
Blaauw.draught = AIS.draught(indx_bl);
Blaauw.Fr = AIS.Fr(indx_bl);
Blaauw.Fr_mod = AIS.Fr_mod(indx_bl);
Blaauw.dist = AIS.dist(indx_bl);
Blaauw.W = AIS.W(indx_bl);
Blaauw.time = datenum('14-Nov-2022 08:06:13'):(10/60/24):(datenum('14-Nov-2022 08:06:13')+((length(Blaauw.meas)-1)*(10/60/24))); Blaauw.time = Blaauw.time';

close all;

%% QQ plots
% 
% % for j = 1:1:6;
%         for j = 1;
%     
%     ID_formula = j ;
%     
%     
%     if event_window == 0.75 ;
%         if ID_formula == 1;
%             name = 'bhowmik';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 04 14 20 00]) 10];
%         elseif ID_formula == 2;
%             name = 'sorensen';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 05 01 40 00]) 10];
%         elseif ID_formula == 3;
%             name = 'pianc';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 05 22 10 00]) 10];
%         elseif ID_formula == 4;
%             name = 'gates';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 06 09 10 00]) 10];
%         elseif ID_formula == 5;
%             name = 'blaauw';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 05 23 00 00]) 10];
%         elseif ID_formula == 6;
%             name = 'kriebel';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 04 04 00 00]) 10];
%         else
%         end
%         
%         
%         file = ['\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\OSSI_dfs0_window45sec\' name '.dfs0'];
%         Out = [OutPlots 'QQplots\' name '\']; mkdir (Out);
%         cd(Out);
%         
%     elseif event_window == 0.5 ;
%         if ID_formula == 1;
%             name = 'bhowmik';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 03 09 50 00]) 10];
%         elseif ID_formula == 2;
%             name = 'sorensen';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 03 09 50 00]) 10];
%         elseif ID_formula == 3;
%             name = 'pianc';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 03 06 10 00]) 10];
%         elseif ID_formula == 4;
%             name = 'gates';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 03 09 30 00]) 10];
%         elseif ID_formula == 5;
%             name = 'blaauw';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 03 07 00 00]) 10];
%         elseif ID_formula == 6;
%             name = 'kriebel';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 03 02 50 00]) 10];
%         else
%         end
%         
%         file = ['\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\OSSI_dfs0_window30sec_NoDraft0\' name '.dfs0'];
%         Out = [OutPlots 'QQplots_window30sec_NoDraft0\' name '\']; mkdir (Out);
%         cd(Out);
%         
%     elseif event_window == 0.25 ;
%         if ID_formula == 1;
%             name = 'bhowmik';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 02 04 50 00]) 10];
%         elseif ID_formula == 2;
%             name = 'sorensen';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 02 08 00 00]) 10];
%         elseif ID_formula == 3;
%             name = 'pianc';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 02 15 10 00]) 10];
%         elseif ID_formula == 4;
%             name = 'gates';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 02 18 30 00]) 10];
%         elseif ID_formula == 5;
%             name = 'blaauw';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 02 15 30 00]) 10];
%         elseif ID_formula == 6;
%             name = 'kriebel';
%             ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 02 01 10 00]) 10];
%         else
%         end
%         
%         file = ['\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\OSSI_dfs0_window15sec\' name '.dfs0'];
%         Out = [OutPlots 'QQplots_window15sec\' name '\']; mkdir (Out);
%         cd(Out);        
%         
%     end
%     name_ADCP1='meas';
%     bins_CS= [0:0.1:1];
%     xyz_ADCP1 = [103.662160 1.405210];
%     legend_ADCP1 = 'meas';
%     meas.item= 'Maximum wave height';
%     meas.unit= 'm';
%     
%     
%     %     formula=m_structure(name,xyz_ADCP1,ttt,name,file,'Speed',1,bins_CS);
%     meas= m_structure('Measurements',xyz_ADCP1,ttt,'Measurements',file,'Speed',1,bins_CS);
%     formula=m_structure('Empirical formulation',xyz_ADCP1,ttt,'Empirical formulation',file,'Speed',2,bins_CS);
%     
%     
%     meas.item= 'Significant wave height';
%     meas.unit= 'm';
%     formula.item= 'Maximum wave height';
%     formula.unit= 'm';
%     
%     m_compare(meas, formula,'plotflag',[1,2]);
%     
% end
% 
% 
% 
% 

%% Investigate bad fit \

close all;
cd('\\sg-ncr04\projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\Plot_comparison_measurements_WUHL_B_Le\');

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(Kriebel.meas,Kriebel.formula,20,Kriebel.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(Kriebel.meas,Kriebel.formula,20,Kriebel.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(Kriebel.meas,Kriebel.formula,20,Kriebel.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(Kriebel.meas,Kriebel.formula,20,Kriebel.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(Kriebel.meas,Kriebel.formula,20,Kriebel.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(Kriebel.meas,Kriebel.formula,20,Kriebel.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_Kriebel_allVariables.png'],'-dpng');


close all ;

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(Bhowmik.meas,Bhowmik.formula,20,Bhowmik.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(Bhowmik.meas,Bhowmik.formula,20,Bhowmik.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(Bhowmik.meas,Bhowmik.formula,20,Bhowmik.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(Bhowmik.meas,Bhowmik.formula,20,Bhowmik.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(Bhowmik.meas,Bhowmik.formula,20,Bhowmik.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(Bhowmik.meas,Bhowmik.formula,20,Bhowmik.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_Bhowmik_allVariables.png'],'-dpng');


close all ;

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(Gates.meas,Gates.formula,20,Gates.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(Gates.meas,Gates.formula,20,Gates.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(Gates.meas,Gates.formula,20,Gates.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(Gates.meas,Gates.formula,20,Gates.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(Gates.meas,Gates.formula,20,Gates.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(Gates.meas,Gates.formula,20,Gates.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_Gates_allVariables.png'],'-dpng');


close all ;

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(Blaauw.meas,Blaauw.formula,20,Blaauw.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(Blaauw.meas,Blaauw.formula,20,Blaauw.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(Blaauw.meas,Blaauw.formula,20,Blaauw.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(Blaauw.meas,Blaauw.formula,20,Blaauw.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(Blaauw.meas,Blaauw.formula,20,Blaauw.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(Blaauw.meas,Blaauw.formula,20,Blaauw.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'Blaauw (1985) 2','Blaauw (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_Blaauw_allVariables.png'],'-dpng');

close all ;

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(Sorensen.meas,Sorensen.formula,20,Sorensen.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'Sorensen (1985) 2','Sorensen (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(Sorensen.meas,Sorensen.formula,20,Sorensen.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'Sorensen (1985) 2','Sorensen (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(Sorensen.meas,Sorensen.formula,20,Sorensen.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'Sorensen (1985) 2','Sorensen (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(Sorensen.meas,Sorensen.formula,20,Sorensen.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'Sorensen (1985) 2','Sorensen (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(Sorensen.meas,Sorensen.formula,20,Sorensen.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'Sorensen (1985) 2','Sorensen (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(Sorensen.meas,Sorensen.formula,20,Sorensen.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'Sorensen (1985) 2','Sorensen (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_Sorensen_allVariables.png'],'-dpng');


close all ;

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,20,PIANC.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,20,PIANC.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,20,PIANC.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,20,PIANC.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,20,PIANC.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,20,PIANC.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_PIANC_allVariables.png'],'-dpng');


%% Calculate the diff and then plot again 
% diff = formula - measurement


Kriebel.diff = Kriebel.formula - Kriebel.meas;
PIANC.diff = PIANC.formula - PIANC.meas;
Sorensen.diff = Sorensen.formula - Sorensen.meas;
Gates.diff = Gates.formula - Gates.meas;
Bhowmik.diff = Bhowmik.formula - Bhowmik.meas;
Blaauw.diff = Blaauw.formula - Blaauw.meas;



close all;
cd('\\sg-ncr04\projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\Plot_comparison_measurements_WUHL_B_Le\');

close all;
figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Kriebel.L,Kriebel.diff,10,'filled');
title('Vessel length'); ylabel('diff (m)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Kriebel.sog,Kriebel.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Kriebel.draught,Kriebel.diff,10,'filled');
title('Vessel draft'); ylabel('diff (m)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Kriebel.Fr,Kriebel.diff,10,'filled');
title('Froude number'); ylabel('diff (m)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Kriebel.dist,Kriebel.diff,10,'filled');
title('Distance with OSSI'); ylabel('diff (m)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Kriebel.W,Kriebel.diff,10,'filled');
title('Water displacement'); ylabel('diff (m)'); xlabel('W (m^3)');
print(['diff_AIS_measurements_Kriebel_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(PIANC.L,PIANC.diff,10,'filled');
title('Vessel length'); ylabel('diff (m)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(PIANC.sog,PIANC.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(PIANC.draught,PIANC.diff,10,'filled');
title('Vessel draft'); ylabel('diff (m)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(PIANC.Fr,PIANC.diff,10,'filled');
title('Froude number'); ylabel('diff (m)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(PIANC.dist,PIANC.diff,10,'filled');
title('Distance with OSSI'); ylabel('diff (m)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(PIANC.W,PIANC.diff,10,'filled');
title('Water displacement'); ylabel('diff (m)'); xlabel('W (m^3)');
print(['diff_AIS_measurements_PIANC_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Sorensen.L,Sorensen.diff,10,'filled');
title('Vessel length'); ylabel('diff (m)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Sorensen.sog,Sorensen.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Sorensen.draught,Sorensen.diff,10,'filled');
title('Vessel draft'); ylabel('diff (m)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Sorensen.Fr,Sorensen.diff,10,'filled');
title('Froude number'); ylabel('diff (m)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Sorensen.dist,Sorensen.diff,10,'filled');
title('Distance with OSSI'); ylabel('diff (m)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Sorensen.W,Sorensen.diff,10,'filled');
title('Water displacement'); ylabel('diff (m)'); xlabel('W (m^3)');
print(['diff_AIS_measurements_Sorensen_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Gates.L,Gates.diff,10,'filled');
title('Vessel length'); ylabel('diff (m)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Gates.sog,Gates.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Gates.draught,Gates.diff,10,'filled');
title('Vessel draft'); ylabel('diff (m)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Gates.Fr,Gates.diff,10,'filled');
title('Froude number'); ylabel('diff (m)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Gates.dist,Gates.diff,10,'filled');
title('Distance with OSSI'); ylabel('diff (m)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Gates.W,Gates.diff,10,'filled');
title('Water displacement'); ylabel('diff (m)'); xlabel('W (m^3)');
print(['diff_AIS_measurements_Gates_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Bhowmik.L,Bhowmik.diff,10,'filled');
title('Vessel length'); ylabel('diff (m)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Bhowmik.sog,Bhowmik.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Bhowmik.draught,Bhowmik.diff,10,'filled');
title('Vessel draft'); ylabel('diff (m)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Bhowmik.Fr,Bhowmik.diff,10,'filled');
title('Froude number'); ylabel('diff (m)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Bhowmik.dist,Bhowmik.diff,10,'filled');
title('Distance with OSSI'); ylabel('diff (m)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Bhowmik.W,Bhowmik.diff,10,'filled');
title('Water displacement'); ylabel('diff (m)'); xlabel('W (m^3)');
print(['diff_AIS_measurements_Bhowmik_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Blaauw.L,Blaauw.diff,10,'filled');
title('Vessel length'); ylabel('diff (m)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Blaauw.sog,Blaauw.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Blaauw.draught,Blaauw.diff,10,'filled');
title('Vessel draft'); ylabel('diff (m)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Blaauw.Fr,Blaauw.diff,10,'filled');
title('Froude number'); ylabel('diff (m)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Blaauw.dist,Blaauw.diff,10,'filled');
title('Distance with OSSI'); ylabel('diff (m)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Blaauw.W,Blaauw.diff,10,'filled');
title('Water displacement'); ylabel('diff (m)'); xlabel('W (m^3)');
print(['diff_AIS_measurements_Blaauw_allVariables.png'],'-dpng');

close all;


%% Calculate the error and then plot again 
% error = diff/measurement


Kriebel.error = Kriebel.diff./Kriebel.meas; Kriebel.error = Kriebel.error*100;
PIANC.error = PIANC.diff./PIANC.meas;PIANC.error = PIANC.error*100;
Sorensen.error = Sorensen.diff./Sorensen.meas;Sorensen.error = Sorensen.error*100;
Gates.error = Gates.diff./Gates.meas;Gates.error = Gates.error*100;
Bhowmik.error = Bhowmik.diff./Bhowmik.meas;Bhowmik.error = Bhowmik.error*100;
Blaauw.error = Blaauw.diff./Blaauw.meas;Blaauw.error = Blaauw.error*100;



close all;
cd('\\sg-ncr04\projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\Plot_comparison_measurements_WUHL_B_Le\');

close all;
figure('units','normalized','outerposition',[0 0 1 1]);

g1=subplot(2,3,1);
hold on;yscale log;grid on;
scatter(Kriebel.L,Kriebel.error,10,'filled');
title('Vessel length'); ylabel('error (%)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on;
yscale log;
grid on;
scatter(Kriebel.sog,Kriebel.error,10,'filled');
title('Vessel speed'); ylabel('error (%)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on;
yscale log;
grid on;
scatter(Kriebel.draught,Kriebel.error,10,'filled');
title('Vessel draft'); ylabel('error (%)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on;
yscale log;
grid on;
scatter(Kriebel.Fr,Kriebel.error,10,'filled');
title('Froude number'); ylabel('error (%)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on;
yscale log;
grid on;
scatter(Kriebel.dist,Kriebel.error,10,'filled');
title('Distance with OSSI'); ylabel('error (%)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on;
yscale log;
grid on;
scatter(Kriebel.W,Kriebel.error,10,'filled');
title('Water displacement'); ylabel('error (%)'); xlabel('W (m^3)');
print(['error_AIS_measurements_Kriebel_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);

g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(PIANC.L,PIANC.error,10,'filled');
title('Vessel length'); ylabel('error (%)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(PIANC.sog,PIANC.error,10,'filled');
title('Vessel speed'); ylabel('error (%)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(PIANC.draught,PIANC.error,10,'filled');
title('Vessel draft'); ylabel('error (%)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(PIANC.Fr,PIANC.error,10,'filled');
title('Froude number'); ylabel('error (%)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(PIANC.dist,PIANC.error,10,'filled');
title('Distance with OSSI'); ylabel('error (%)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(PIANC.W,PIANC.error,10,'filled');
title('Water displacement'); ylabel('error (%)'); xlabel('W (m^3)');
print(['error_AIS_measurements_PIANC_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
yscale log;
g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Sorensen.L,Sorensen.error,10,'filled');
title('Vessel length'); ylabel('error (%)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Sorensen.sog,Sorensen.error,10,'filled');
title('Vessel speed'); ylabel('error (%)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Sorensen.draught,Sorensen.error,10,'filled');
title('Vessel draft'); ylabel('error (%)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Sorensen.Fr,Sorensen.error,10,'filled');
title('Froude number'); ylabel('error (%)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Sorensen.dist,Sorensen.error,10,'filled');
title('Distance with OSSI'); ylabel('error (%)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Sorensen.W,Sorensen.error,10,'filled');
title('Water displacement'); ylabel('error (%)'); xlabel('W (m^3)');
print(['error_AIS_measurements_Sorensen_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
yscale log;

g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Gates.L,Gates.error,10,'filled');
title('Vessel length'); ylabel('error (%)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Gates.sog,Gates.error,10,'filled');
title('Vessel speed'); ylabel('error (%)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Gates.draught,Gates.error,10,'filled');
title('Vessel draft'); ylabel('error (%)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Gates.Fr,Gates.error,10,'filled');
title('Froude number'); ylabel('error (%)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Gates.dist,Gates.error,10,'filled');
title('Distance with OSSI'); ylabel('error (%)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Gates.W,Gates.error,10,'filled');
title('Water displacement'); ylabel('error (%)'); xlabel('W (m^3)');
print(['error_AIS_measurements_Gates_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
yscale log;

g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Bhowmik.L,Bhowmik.error,10,'filled');
title('Vessel length'); ylabel('error (%)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Bhowmik.sog,Bhowmik.error,10,'filled');
title('Vessel speed'); ylabel('error (%)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Bhowmik.draught,Bhowmik.error,10,'filled');
title('Vessel draft'); ylabel('error (%)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Bhowmik.Fr,Bhowmik.error,10,'filled');
title('Froude number'); ylabel('error (%)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Bhowmik.dist,Bhowmik.error,10,'filled');
title('Distance with OSSI'); ylabel('error (%)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Bhowmik.W,Bhowmik.error,10,'filled');
title('Water displacement'); ylabel('error (%)'); xlabel('W (m^3)');
print(['error_AIS_measurements_Bhowmik_allVariables.png'],'-dpng');


close all;
figure('units','normalized','outerposition',[0 0 1 1]);
yscale log;

g1=subplot(2,3,1);
hold on; yscale log;grid on;
scatter(Blaauw.L,Blaauw.error,10,'filled');
title('Vessel length'); ylabel('error (%)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on; yscale log
grid on
scatter(Blaauw.sog,Blaauw.error,10,'filled');
title('Vessel speed'); ylabel('error (%)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on; yscale log
grid on
scatter(Blaauw.draught,Blaauw.error,10,'filled');
title('Vessel draft'); ylabel('error (%)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on; yscale log
grid on
scatter(Blaauw.Fr,Blaauw.error,10,'filled');
title('Froude number'); ylabel('error (%)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on; yscale log
grid on
scatter(Blaauw.dist,Blaauw.error,10,'filled');
title('Distance with OSSI'); ylabel('error (%)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on; yscale log
grid on
scatter(Blaauw.W,Blaauw.error,10,'filled');
title('Water displacement'); ylabel('error (%)'); xlabel('W (m^3)');
print(['error_AIS_measurements_Blaauw_allVariables.png'],'-dpng');

close all;


%% Apply a correction on PIANC formula depending on SOG

close all;
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;grid on;
scatter(PIANC.sog,PIANC.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');

close all;
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;grid on;
scatter(PIANC.sog,PIANC.diff,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');

p1 = 0.002873;
p2 = -0.017611;
p3 = 0.045849;
p4 = -0.18392;

PIANC.corr_factor = (p1.*PIANC.sog.^3)+(p2.*PIANC.sog.^2)+(p3.*PIANC.sog)+p4;
PIANC.formula_corrected = PIANC.formula - PIANC.corr_factor ;




figure;
g1=subplot(1,2,1);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,10,'b','filled');
plot(x,y,'--k')
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
title(['PIANC formulation vs measurements'])% - window to select events ' num2str(event_window) ' min']);
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(1,2,2);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_corrected,10,'b','filled');
plot(x,y,'--k')
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
title(['PIANC corrected formulation vs measurements'])% - window to select events ' num2str(event_window) ' min']);
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatter_AIS_measurements_Window_' num2str(event_window) 'min_NoDraft0.png'],'-dpng');


name = 'pianc';
ttt = [datenum([2023 06 01 00 00 00]) datenum([2023 06 03 06 10 00]) 10];

                
file = ['\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\OSSI_dfs0_window30sec_NoDraft0\' name '.dfs0'];
Out = [OutPlots 'QQplots_window30sec_NoDraft0_Scaled\' name '\']; mkdir (Out);
cd(Out);


name_ADCP1='meas';
bins_CS= [0:0.1:1];
xyz_ADCP1 = [103.662160 1.405210];
legend_ADCP1 = 'meas';
meas.item= 'Maximum wave height';
meas.unit= 'm';


%     formula=m_structure(name,xyz_ADCP1,ttt,name,file,'Speed',1,bins_CS);
meas= m_structure('Measurements',xyz_ADCP1,ttt,'Measurements',file,'Speed',1,bins_CS);
formula=m_structure('Empirical formulation corrected',xyz_ADCP1,ttt,'Empirical formulation corrected',file,'Speed',2,bins_CS);


meas.item= 'Maximum wave height';
meas.unit= 'm';
formula.item= 'Maximum wave height';
formula.unit= 'm';

m_compare(meas, formula,'plotflag',[1,2]);

%% Try a new relation based on PIANC --> Power function
% 
% A = 1 ;
% y = AIS.CorrespondingEvent.Hmax ./ (A.*AIS.waterdepth.*((AIS.dist./AIS.waterdepth).^(-0.33))) ;
% x = AIS.Fr_depth ;
% 
% figure('units','normalized','outerposition',[0 0 1 1]);
% hold on;grid on;
% scatter(x,y,10,'filled');
% 



close all ;

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_corrected,20,PIANC.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_corrected,20,PIANC.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_corrected,20,PIANC.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_corrected,20,PIANC.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_corrected,20,PIANC.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_corrected,20,PIANC.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_PIANCmodified_allVariables.png'],'-dpng');


%% Clean results of PIANC for Fr > 0.7 and then investigate again 

cd('\\sg-ncr04\projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\Plot_comparison_measurements_WUHL_B_Le');


% Find Fr numbers > 0.7 and remove 

idx_Fr = find(PIANC.Fr > 0.69);
PIANC.formula_Ffilter = PIANC.formula ; PIANC.formula_Ffilter(idx_Fr) = NaN;


close all ;

figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Ffilter,20,PIANC.L,'filled');
plot(x,y,'--k')
title('Vessel length');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g2=subplot(2,3,2);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Ffilter,20,PIANC.sog,'filled');
plot(x,y,'--k')
title('Vessel speed');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g3=subplot(2,3,3);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Ffilter,20,PIANC.draught,'filled');
plot(x,y,'--k')
title('Vessel draft');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g4=subplot(2,3,4);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Ffilter,20,PIANC.Fr,'filled');
plot(x,y,'--k')
title('Froude number');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g5=subplot(2,3,5);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Ffilter,20,PIANC.dist,'filled');
plot(x,y,'--k')
title('Distance with OSSI');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
g6=subplot(2,3,6);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Ffilter,20,PIANC.W,'filled');
plot(x,y,'--k')
title('Water displacement');%,'PIANC (1985) 2','PIANC (1985) 3');
colorbar;
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
ylim([0 0.5]);
xlim([0 0.5])
print(['Scatters_AIS_measurements_PIANC_Ffilter_allVariables.png'],'-dpng');



PIANC.diff_Frmod = PIANC.formula_Ffilter - PIANC.meas;

close all;
figure('units','normalized','outerposition',[0 0 1 1]);
g1=subplot(2,3,1);
hold on;grid on;
scatter(PIANC.L,PIANC.diff_Frmod,10,'filled');
title('Vessel length'); ylabel('diff_Frmod (m)'); xlabel('LOA (m)');
g2=subplot(2,3,2);
hold on
grid on
scatter(PIANC.sog,PIANC.diff_Frmod,10,'filled');
title('Vessel speed'); ylabel('diff_Frmod (m)'); xlabel('SOG (m/s)');
g3=subplot(2,3,3);
hold on
grid on
scatter(PIANC.draught,PIANC.diff_Frmod,10,'filled');
title('Vessel draft'); ylabel('diff_Frmod (m)'); xlabel('Draft (m)');
g4=subplot(2,3,4);
hold on
grid on
scatter(PIANC.Fr,PIANC.diff_Frmod,10,'filled');
title('Froude number'); ylabel('diff_Frmod (m)'); xlabel('Fr');
g5=subplot(2,3,5);
hold on
grid on
scatter(PIANC.dist,PIANC.diff_Frmod,10,'filled');
title('Distance with OSSI'); ylabel('diff_Frmod (m)'); xlabel('Distance (m)');
g6=subplot(2,3,6);
hold on
grid on
scatter(PIANC.W,PIANC.diff_Frmod,10,'filled');
title('Water displacement'); ylabel('diff_Frmod (m)'); xlabel('W (m^3)');
print(['diff_Frmod_AIS_measurements_PIANC_Ffilter_allVariables.png'],'-dpng');






close all;
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;grid on;
scatter(PIANC.sog,PIANC.diff_Frmod,10,'filled');
title('Vessel speed'); ylabel('diff (m)'); xlabel('SOG (m/s)');

% 4th degree polynomial fitting
p1 = 0.00036948;
p2 = -0.0037683;
p3 = 0.020035;
p4 = -0.030145;
p5 = -0.144;

% PIANC.corr_factor = (p1.*PIANC.sog.^2)+(p2.*PIANC.sog)+p3;

PIANC.corr_factor = (p1.*PIANC.sog.^4) + (p2.*PIANC.sog.^3) + (p3.*PIANC.sog.^2) + (p4.*PIANC.sog) + p5 ;
PIANC.formula_Frmod_corrected = PIANC.formula_Ffilter - PIANC.corr_factor ;


close all;

figure;
g1=subplot(1,3,1);
hold on
grid on
scatter(PIANC.meas,PIANC.formula,10,'b','filled');
plot(x,y,'--k')
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
title(['PIANC'])% - window to select events ' num2str(event_window) ' min']);
ylim([0 0.5]);
xlim([0 0.5])

g2=subplot(1,3,2);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Ffilter,10,'b','filled');
plot(x,y,'--k')
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
title(['PIANC (Fr < 0.7)'])% - window to select events ' num2str(event_window) ' min']);
ylim([0 0.5]);
xlim([0 0.5])

g3=subplot(1,3,3);
hold on
grid on
scatter(PIANC.meas,PIANC.formula_Frmod_corrected,10,'b','filled');
plot(x,y,'--k')
ylabel('H_{max} empirical formulation (m)');
xlabel('H_{max} measurement (m)');
title(['PIANC corrected (Fr < 0.7)'])% - window to select events ' num2str(event_window) ' min']);
ylim([0 0.5]);
xlim([0 0.5])

print(['Scatter_AIS_measurements_PIANC_PIANCfiltered_PIANCmodified.png'],'-dpng');


