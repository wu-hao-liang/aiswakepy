% AIS data processing using various empirical formulations to determinate
% ship wake caracteristics
% CHAP
% DHI SG
% 17/02/2023
% WUHL
% Determine Bow Entry Length and Block Coefficient based on vessel type
% 22/02/2024

close all; clear all; clc;

addpath(genpath('C:\Users\chap\OneDrive - DHI\Documents\Toolboxes\MTOOLS\wafo26\'))
addpath(genpath('C:\Users\chap\OneDrive - DHI\Documents\Toolboxes\MTOOLS\m_tools_20201012'))
addpath(genpath('C:\Users\chap\OneDrive - DHI\Documents\Toolboxes\MTOOLS\DHIMatlabToolbox\DHIMatlabToolbox-Mz2020'))
addpath(genpath('C:\Users\chap\OneDrive - DHI\Documents\Toolboxes\WaveProcessing_ChapTool')) % toolbox for wave processing
addpath(genpath('\\sg-ncr04\projects\61802983 JI SSES\MATLAB\0_Scripts\ShipwakeCalculation\functions'))

%% Initializing

% folder directory
OutputFolder = '\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\';
PlotsVessels = '\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\AIS_Shipwake\Plot_vesselTrack'; mkdir(PlotsVessels);

% Info about measurement point
MeasLoc = [103.733335 1.265771]; % Coordinates for OSSI
MeasName = 'OSSI Merbau';

% AIS file csv edited (speed interp + fitered with numtime)
AIS_RawFile = '\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\0_Scripts\ShipwakeCalculation\data\DT_JI_ChannelEdited_ed2.csv';


% shp file of Singapore
SG_shp_file = '\\sg-gis\GIS\Template\GIS\Shapefiles & Geodatabase\Singapore\Shapefiles\Land\Historical Profile\v20230217\RD7550_CEx_SG_v20230217';
SG_shp = m_shaperead(SG_shp_file);
shp.lonlat = [];
for jj = 1:length(SG_shp.ncst);
    lonlat = SG_shp.ncst{jj};
    shp.lonlat = [shp.lonlat;lonlat];
end
%% options
plot_vessel_track = 1 ;
proc_AIS = 1 ;
xls_procAIS = 0;

if proc_AIS ==1
    %% Load data
    
    filename = AIS_RawFile;
    delimiter = {',','-','T',':'};
    startRow = 2;
    
    formatSpec = '%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    
    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
    end
    numericData = NaN(size(dataArray{1},1),size(dataArray,2));
    
    for col=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        rawData = dataArray{col};
        for row=1:size(rawData, 1)
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData(row), regexstr, 'names');
                numbers = result.numbers;
                invalidThousandsSeparator = false;
                if numbers.contains(',')
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(numbers, thousandsRegExp, 'once'))
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end
                if ~invalidThousandsSeparator
                    numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch
                raw{row, col} = rawData{row};
            end
        end
    end
    
    R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
    raw(R) = {NaN}; % Replace non-numeric cells
    
    mmsi = cell2mat(raw(:, 1));
    width = cell2mat(raw(:, 2));
    L = cell2mat(raw(:, 3));
    draught = cell2mat(raw(:, 4));
    Y = cell2mat(raw(:, 5));
    M = cell2mat(raw(:, 6));
    D = cell2mat(raw(:, 7));
    H = cell2mat(raw(:, 8));
    min = cell2mat(raw(:, 9));
    sec = cell2mat(raw(:, 10));
    time = datenum(Y,M,D,H,min,sec);
    lon = cell2mat(raw(:, 11));
    lat = cell2mat(raw(:, 12));
    sog = cell2mat(raw(:, 13));
    cog = cell2mat(raw(:, 14));
    waterdepth = cell2mat(raw(:, 15));
    typecargo = cell2mat(raw(:, 16));
    
    clearvars filename delimiter startRow formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp R Y M D H min sec;
    
    
    
    
    %% Calculate distance from vessel to measurement point
    
    dist = [];
    for i = 1:length(lat);
        Xv = lon(i);
        Yv = lat(i);
        Xm = MeasLoc(1);
        Ym = MeasLoc(2);
        d = distXY(Xv,Yv,Xm,Ym);
        dist = [dist;d];
    end
    clear Xv Yv Xm Ym d i
    
    AISraw.mmsi = mmsi;
    AISraw.width = width;
    AISraw.L = L;
    AISraw.draught = draught;
    AISraw.time = time;
    AISraw.lon = lon;
    AISraw.lat = lat;
    AISraw.sog = sog;
    AISraw.cog = cog;
    AISraw.waterdepth = waterdepth;
    AISraw.dist = dist;
    AISraw.typecargo = typecargo;
    clear mmsi  width L draught time lon lat sog cog waterdepth dist
    
    %% Dissociate each journey of separate vessels
    
    % attribute a number to separate each journey of separate vessels
    
    AISraw.vesselNo = nan(length(AISraw.time),1);
    No = 1;
    time_separation_journey = 1; % time in hours  to separate journey of each
    % vessel with same mmsi because are doing go and return to puteri harbour !
    
    for i = 1:length(AISraw.mmsi);
        % for i = 1:13;
        
        n = AISraw.vesselNo(i);
        
        if isnan(n) == 1;
            mmsi = AISraw.mmsi(i);
            time = AISraw.time(i);
            indx_time_mmsi = find(AISraw.time<= time + (time_separation_journey/24) & AISraw.mmsi == mmsi & isnan(AISraw.vesselNo) == 1);
            AISraw.vesselNo(indx_time_mmsi) = No;
            No = No+1;
        else
        end
    end
    
    disp(['Total of ' num2str(No-1) ' separated vessels journey recorded in AIS dataset']);
    
    clear No n i mmsi time indx_time_mmsi
    
    %% We remove every line with sog < 2knots
    
    sog_limit = 2;
    
    AIS.mmsi = [];
    AIS.width = [];
    AIS.L = [];
    AIS.draught = [];
    AIS.time = [];
    AIS.lon = [];
    AIS.lat = [];
    AIS.sog = [];
    AIS.cog = [];
    AIS.waterdepth = [];
    AIS.dist = [];
    AIS.vesselNo = [];
    AIS.typecargo = [];
    
    for k = 1:length(AISraw.time);
        if AISraw.sog(k) >= sog_limit & AISraw.draught(k) > 0 & AISraw.draught(k) < AISraw.waterdepth(k) - 1;
            AIS.mmsi = [AIS.mmsi;AISraw.mmsi(k)];
            AIS.width = [AIS.width;AISraw.width(k)];
            AIS.L = [AIS.L;AISraw.L(k)];
            AIS.draught = [AIS.draught;AISraw.draught(k)];
            AIS.time = [AIS.time;AISraw.time(k)];
            AIS.lon = [AIS.lon;AISraw.lon(k)];
            AIS.lat = [AIS.lat;AISraw.lat(k)];
            AIS.sog = [AIS.sog;AISraw.sog(k)];
            AIS.cog = [AIS.cog;AISraw.cog(k)];
            AIS.waterdepth = [AIS.waterdepth;AISraw.waterdepth(k)];
            AIS.dist = [AIS.dist;AISraw.dist(k)];
            AIS.vesselNo = [AIS.vesselNo;AISraw.vesselNo(k)];
            AIS.typecargo = [AIS.typecargo;AISraw.typecargo(k)];
        end
    end
    
    
    
    
    %% clear shp
    
%     figure('units','normalized','outerposition',[0 0 0.6 1])
%     hold on
%     plot(shp.lonlat(:,1),shp.lonlat(:,2),'k')
%     ylabel('Latitude');
%     xlabel('Longitude');
%     grid on
%     xlim(Xlim)
%     ylim(Ylim)
%     
    
    shp.lonlat(18888,:) = NaN;
    shp.lonlat(18583,:) = NaN;
    
    
    %% Now we work on each vessel journey to determinate the orthogonal point of the track
    
    cd(PlotsVessels)
    tot_No = max(AIS.vesselNo);
    AISproc = [];
    Xlim = [103.725 103.75] ;
    Ylim = [1.248 1.273];
    close all
    
%     q = 1;
%     
%     for i = 1:tot_No;
%         % for i = 6;
%         disp(['Plot vessel number ' num2str(i) '/' num2str(tot_No)]);
%         indx_track = find(AIS.vesselNo == i);
%         
%         if length(indx_track )>1; % if only one point then we remove the vessel journey
%             p = polyfit(AIS.lon(indx_track),AIS.lat(indx_track),1);
%             Xfit = Xlim;
%             Yfit = [(p(1)*Xfit(1) + p(2)) (p(1)*Xfit(2) + p(2))];
%             xx = Xfit(1):0.00001:Xfit(2);
%             
%             for j = 1:length(xx);
%                 yy(j) = p(1)*xx(j) + p(2);
%                 distLine(j) = distXY(MeasLoc(1),MeasLoc(2),xx(j),yy(j));
%             end
%             
%             indx_minDist = find(distLine == min(distLine));
%             x_ortho = xx(indx_minDist);
%             y_ortho = yy(indx_minDist);
%             
%             clear  x0 ii
%             
%             Xv = AIS.lon(indx_track);
%             Yv = AIS.lat(indx_track);
%             
%             % determinate the closest AIS data point from the orthogonal point
%             for m = 1:length(indx_track);
%                 d_vessel_ortho(m) = distXY(Xv(m),Yv(m),x_ortho,y_ortho);
%             end
%             
%             indx_closestV = find(d_vessel_ortho == min(d_vessel_ortho));
%             
%             indx_closest_vessel_ortho = indx_track(indx_closestV);
%             
%             
%             
%             
%             AISproc.mmsi(q,1) = AIS.mmsi(indx_track(1));
%             AISproc.width(q,1) = AIS.width(indx_track(1));
%             AISproc.L(q,1) = AIS.L(indx_track(1));
%             AISproc.draught(q,1) = AIS.draught(indx_track(1));
%             AISproc.lon(q,1) = x_ortho;
%             AISproc.lat(q,1) = y_ortho;
%             AISproc.sog(q,1) = AIS.sog(indx_closest_vessel_ortho);
%             AISproc.cog(q,1) = AIS.cog(indx_closest_vessel_ortho);
%             AISproc.waterdepth(q,1) = AIS.waterdepth(indx_closest_vessel_ortho);
%             AISproc.dist(q,1)= distLine(indx_minDist);
%             AISproc.vesselNo(q,1) = AIS.vesselNo(indx_track(1));
%             
%             
%             
%             % determination of the time
%             if AIS.lat(indx_track(1))>AIS.lat(indx_track(2)); % vessel going to the north
%                 if AISproc.lat(q,1) < AIS.lat(indx_track(2)); % if vessel located more north than the last point
%                     AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) + ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                 elseif AISproc.lat(q,1) > AIS.lat(indx_track(1)) ; % if vessel located more south than the first point
%                     AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) - ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                 else
%                     if AIS.lat(indx_closest_vessel_ortho) < AISproc.lat(q,1);
%                         AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) - ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                     else
%                         AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) + ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                     end
%                 end
%             else % vessel going to the south
%                 if AISproc.lat(q,1) < AIS.lat(indx_track(2)); % if vessel located more north than the last point
%                     AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) - ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                 elseif AISproc.lat(q,1) > AIS.lat(indx_track(1)) ; % if vessel located more south than the first point
%                     AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) + ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                 else
%                     if AIS.lat(indx_closest_vessel_ortho) < AISproc.lat(q,1);
%                         AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) + ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                     else
%                         AISproc.time(q,1) = AIS.time(indx_closest_vessel_ortho) - ((min(d_vessel_ortho)/(AIS.sog(indx_closest_vessel_ortho)*0.514444))/86400);
%                     end
%                 end
%             end
%             
%             
%             Xv = AIS.lon(indx_track);
%             Yv = AIS.lat(indx_track);
%             
%             
%             
% %             if plot_vessel_track ==1;
% %                 figure('units','normalized','outerposition',[0 0 0.6 1])
% %                 hold on
% %                 scatter(MeasLoc(1),MeasLoc(2),'r','filled','o');
% %                 scatter(xx, yy, 100, distLine, 'filled')
% %                 c = colorbar;
% %                 c.Label.String = 'Distance with measurement point (m)';
% %                 scatter(Xv,Yv,'MarkerEdgeColor','k','MarkerFaceColor','w');
% %                 scatter(x_ortho,y_ortho,'MarkerEdgeColor','k','MarkerFaceColor','k')
% %                 plot(Xv,Yv,'--k');
% %                 plot(shp.lonlat(:,1),shp.lonlat(:,2),'k')
% %                 title([num2str(AISproc.mmsi(q,1)) ' AIS vessel track - ' datestr(AISproc.time(q,1))]);
% %                 ylabel('Latitude');
% %                 xlabel('Longitude');
% %                 legend('Measurement location','Approximate journey and distance to mesaurement point','AIS track','Orthogonal point','Location','southeast');
% %                 xlim(Xlim)
% %                 ylim(Ylim)
% %                 grid on
% %                 print([num2str(AISproc.vesselNo(q,1)) '_' num2str(AISproc.mmsi(q,1)) '_AIStrack.png'],'-dpng');
% %             else
% %             end
%             
%             clear Xv Yv xx yy x_ortho y_ortho indx_closest_vessel_ortho indx_closestV indx_minDist indx_track d_vessel_ortho
%             
%             close all
%             q = q+1;
%         else
%         end
%     end
%     disp(['Total of ' num2str(q-1) ' vessels with speed > ' num2str(sog_limit) ' kn'])
%     
%     % Save matfile
%     save(['\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\MATFILES\1_PROCESSED\AISproc_OSSI.mat'],'-struct','AISproc');
%     

%     % save xlsx file
%     if xls_procAIS == 1;
%         xlsx_filename = '\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\4_Final\Shipwake\OSSI\AISproc_final.xlsx';
%         
%         xlswrite(xlsx_filename,AISproc.vesselNo,'Sheet1','A2');
%         xlswrite(xlsx_filename,AISproc.mmsi,'Sheet1','B2');
%         xlswrite(xlsx_filename,AISproc.width,'Sheet1','C2');
%         xlswrite(xlsx_filename,AISproc.L,'Sheet1','D2');
%         xlswrite(xlsx_filename,AISproc.draught,'Sheet1','E2');
%         xlswrite(xlsx_filename,AISproc.time,'Sheet1','F2');
%         xlswrite(xlsx_filename,AISproc.lon,'Sheet1','G2');
%         xlswrite(xlsx_filename,AISproc.lat,'Sheet1','H2');
%         xlswrite(xlsx_filename,AISproc.sog,'Sheet1','I2');
%         xlswrite(xlsx_filename,AISproc.cog,'Sheet1','J2');
%         xlswrite(xlsx_filename,AISproc.waterdepth,'Sheet1','K2');
%         xlswrite(xlsx_filename,AISproc.dist,'Sheet1','L2');
%     else
%     end
    
else
    AIS = load('\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\MATFILES\1_PROCESSED\AISproc_OSSI.mat');
end

clear tot_No sog_limit time_separation_journey AIS_RawFile ans c distLine i ...
    j jj k lonlat m p plot_vessel_track PlotsVessels proc_AIS q SG_shp_file ...
    Xfit Xlim Yfit Ylim xls_procAIS





%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% _______________________EMPIRICAL FORMULATIONS____________________________
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% KRIEBEL AND SEELIG (2005)
% Uniform and geometrically well defined channels

AIS.sog = AIS.sog*0.5144444; % knots to m/s

AIS.B_Le = AIS.typecargo;
AIS.Cb = AIS.typecargo;

for i = 1:length(AIS.typecargo);
    if ismember(AIS.typecargo(i), 80:89) == 1 % Tankers
        AIS.Cb(i)=0.8;
        AIS.B_Le(i)=1;
	elseif ismember(AIS.typecargo(i), [33, 70:79]) == 1 % General Cargo Vessel, Dredger
		AIS.Cb(i)=0.7;
        AIS.B_Le(i)=0.7;
    else                               % Ferries, Fast Ferries, Fishing boats, Tug, Heavy Lifter, Navy Frigate, Sailing, Unknown vessel
		AIS.Cb(i)=0.6;
        AIS.B_Le(i)=0.4;
    end
end
%AIS.Le = AIS.width/1.1; % BowEntry
%AIS.W = AIS.Le.*AIS.draught.*AIS.L.*0.8; % Displacement
AIS.Le = AIS.width./AIS.B_Le;
AIS.L_Le = AIS.L./AIS.Le;
AIS.W = AIS.width.*AIS.draught.*AIS.L.*0.95.*AIS.Cb;
AIS.BLratio = AIS.width./AIS.L ; % Beam length ratio
%AIS.beta = 1.+ 8.*(tanh(0.45.*((AIS.L./AIS.Le)-2))).^3;
%AIS.alpha = 2.35.*(1-(AIS.W./(AIS.L.*AIS.width.*AIS.draught)));
AIS.beta = 1.+ 8.*(tanh(0.45.*((AIS.L_Le)-2))).^3;
AIS.alpha = 2.35.*(1-AIS.Cb);
AIS.Fr = AIS.sog./sqrt(9.81.*AIS.L);
AIS.Fr_mod = AIS.Fr.*exp((AIS.alpha.*AIS.draught)./AIS.waterdepth);

%GHV2 = AIS.beta.*((AIS.Fr_mod-0.1).^2).*((AIS.width/2)./AIS.L).^(-1/3);
GHV2 = AIS.beta.*((AIS.Fr_mod-0.1).^2).*(AIS.dist./AIS.L).^(-1/3);

theta = 35.267*(1-exp(1-12+(12*(AIS.Fr)))); % Weggel & Sorensen (1986)
AIS.Cwave = AIS.sog./cosd(theta); % celerity of the shipwake

% change the time so it is coherent with the time to arrive at the
% measurement point
AIS.time_SailingLine = AIS.time;
for sh = 1:length(AIS.time);
    Cw = AIS.Cwave(sh);
    di = AIS.dist(sh);
    deltaTime = (di/Cw)/60/60/24; % in days
    AIS.time(sh) = AIS.time(sh) + deltaTime;
end

AIS.Kriebel.Hmax = (GHV2.*(AIS.sog.^2))./(9.81);
AIS.Twake = (2*pi/9.81).*AIS.Cwave;

% QAQC : remove spikes
indx_spk = find(AIS.Kriebel.Hmax > 4 | AIS.beta < 1);
AIS.Kriebel.Hmax(indx_spk) = NaN;
betaarg = AIS.beta.*((AIS.Fr_mod-0.1).^2) ;

% if Hmax is NaN, then T is also NaN
indx_nan = find(isnan(AIS.Kriebel.Hmax)==1);

% Filters
indx_remove = find(AIS.Fr_mod > 0.5 | AIS.Fr_mod < 0.1 | betaarg > 0.4);
AIS.Kriebel.Hmax(indx_remove) = NaN;
inx_spike = find(AIS.Kriebel.Hmax > 2); AIS.Kriebel.Hmax(inx_spike) = NaN;
clear indx_remove indx_spk indx_nan betaarg thetarad inx_spike

%% MAYNORD (2005)
% applicable to semi-planing and planning small craft

AIS.Fr_dis = AIS.sog./(sqrt(9.81.*(AIS.W.^(1/3)))) ; % Displacement Froude number
indx_infFrdis = find(AIS.Fr_dis == Inf);
AIS.Fr_dis(indx_infFrdis) = NaN;
depth_L = AIS.waterdepth./AIS.L;

Ccoeff = 0.82; % try 1 and 0.82

AIS.Maynord.Hmax = Ccoeff.*(AIS.Fr_dis.^(-0.58)).*((AIS.dist./(AIS.W.^(1/3))).^(-0.42)).*(AIS.W.^(1/3)); %
% results weird ! REPRENDRE ICI ET VERIFIER !!!!!

indx_remove = find(AIS.Fr_dis < 1.5 & AIS.Fr < 0.6 & depth_L < 0.35);
AIS.Maynord.Hmax(indx_remove) = NaN;

clear indx_remove

%% PIANC (1987)
% Vessels in inland waterways

A = 1; % 1 for tugs, patrol boats and loaded conventional inland motor boats
AIS.Fr_depth = AIS.sog./(sqrt(AIS.waterdepth.*9.81)); % depth Froude number

AIS.PIANC.Hmax = A.*AIS.waterdepth.*((AIS.dist./AIS.waterdepth).^(-0.33)).*(AIS.Fr_depth.^4);
inx_spike = find(AIS.PIANC.Hmax > 2); AIS.PIANC.Hmax(inx_spike) = NaN;

indx_remove = find(AIS.Fr > 0.7);
AIS.PIANC.Hmax(indx_remove) = NaN;

clear indx_remove 

indx_remove = find(AIS.Fr_depth > 0.7);
AIS.PIANC.Hmax(indx_remove) = NaN;


clear inx_spike A


%% Sorensen (1984)
% vessel Froude numbers from 0.2 to 0.8, which are common for most vessel operations


for i = 1:length(AIS.time);
    if AIS.Fr(i) < 0.55;
        beta = -0.225*(AIS.Fr(i))^(-0.699);
        delta = -0.118*(AIS.Fr(i))^(-0.356);
    else
        beta = -0.342;
        delta = -0.146;
    end
    
    a = -0.6/AIS.Fr(i);
    b = 0.75*(AIS.Fr(i)^(-1.125));
    c = (2.653*AIS.Fr(i)) -1.95;
    
    
    waterdepth_adim(i,1) = AIS.waterdepth(i)/(AIS.W(i)^0.33);
    n = beta*(waterdepth_adim(i,1)^delta);
    log_alpha = a + (b*log(waterdepth_adim(i,1))) + (c*(log(waterdepth_adim(i,1))^2));
    alpha = exp(log_alpha);
    dist_adim(i,1) = AIS.dist(i)/(AIS.W(i)^0.33);
    H_adim = alpha*(dist_adim(i,1)^n);
    AIS.Sorensen.Hmax(i,1) = alpha*(dist_adim(i,1)^n);
end

%% Bhowmik et al (1982)

AIS.Bhowmik.Hmax = ((0.133.*(AIS.sog./(sqrt(9.81.*AIS.draught))))).*AIS.draught;




%% CHAP fit

AIS.CHAPScaled.Hmax = ((0.19.*(AIS.sog./(sqrt(9.81.*AIS.draught))))-0.0387).*AIS.draught;
indx_remove = find(AIS.Fr >0.3 );
AIS.CHAPScaled.Hmax(indx_remove) = NaN;

% %% Bhowmik et al (1982) - SCALED (based on QQfit)
%
% AIS.BhowmikScaled.Hmax = ((((0.19.*(AIS.sog./(sqrt(9.81.*AIS.draught))))-0.0387).*AIS.draught)./0.71)-0.04;



%% Gates and Herbich (1977)

V = AIS.sog * 1.944; % To get ms/s to knots
Lf = AIS.L * 3.281; % To get length from m to feet
spd_L_ratio = V./sqrt(Lf); % multiply speed with 1.944 to get knots and Length with 3.281
N = 1 ; % cusp number 

for i = 1:length(AIS.time);
    if spd_L_ratio(i) > 1;
        Kw(i) = 1.133;
    else
        Kw(i) = (-6.905*spd_L_ratio(i))+7.595;
    end
end
Kw = Kw';
AIS.Gates.Hmax = (1.11.*((Kw.*AIS.width)./AIS.Le).*(AIS.sog.^2/(2*9.81)))/(((N*2)+1.5)^0.33);
% AIS.Gates.Hmax2 = (1.11.*((Kw.*AIS.width)./AIS.Le).*(AIS.sog.^2/(2*9.81)))/(((N*2)+1.5)^0.33);

indx_remove = find(AIS.Fr >0.7 );
AIS.Gates.Hmax(indx_remove) = NaN;

%% Blaauw et al. (1985)
% Model developed for large vessels moving in deep water

% coefficient that depends on the vessel hull type and condition
A1 = 0.8;
A2 = 0.35;
A3 = 0.25;

AIS.Blaauw.Hmax1 = A1.*AIS.waterdepth.*((AIS.dist./AIS.waterdepth).^(-0.33)).*((AIS.Fr_depth).^2.67);
AIS.Blaauw.Hmax2 = A2.*AIS.waterdepth.*((AIS.dist./AIS.waterdepth).^(-0.33)).*((AIS.Fr_depth).^2.67);
AIS.Blaauw.Hmax3 = A3.*AIS.waterdepth.*((AIS.dist./AIS.waterdepth).^(-0.33)).*((AIS.Fr_depth).^2.67);

indx_remove = find(AIS.Fr_depth >0.7 );
AIS.Blaauw.Hmax1(indx_remove) = NaN;
AIS.Blaauw.Hmax2(indx_remove) = NaN;
AIS.Blaauw.Hmax3(indx_remove) = NaN;
















%% Save matfile with empirical formulation results
save(['\\sg-ncr04\Projects\61802983 JI SSES\MATLAB\1_Data\MATFILES\1_PROCESSED\AISproc_OSSI_EmpiricalFormula_WUHL_B_Le.mat'],'-struct','AIS');


