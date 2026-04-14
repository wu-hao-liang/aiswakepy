import fiona
from shapely import Polygon
from shapely.geometry import Point,shape
import geopandas
import shapely.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from pathlib import Path
from rich.progress import track
import sys

MainFolder=r'C:\Projects\61803939 Metocean Study for HarbourFront Ferry Terminal (Sub)\DATA\AIS_analysis\2023'
#AIS_inputfile = Path(MainFolder) / Path(r'1_Data\AIS\2023\DTshipwake.csv')
AIS_inputfile = Path(MainFolder) / Path(r'processed\DTshipwake.csv')
AIS_outputfile = Path(MainFolder) / Path(r'processed\DTshipwake_Interp.csv')
#Shipwake_shapefile = Path(MainFolder) / Path(r'1_Data\AIS\Data\shp_shipwake\for_shipwake_smallok')
obstime_format = '%Y-%m-%dT%H:%M:%SZ'
start_datetime = np.datetime64("2023-01-01T00:00:00")
sog2mps = 1852/3600
deg2m = 111111
track_time_limit = 180.0 #consider as two traces for a ship when AIS is not received over 10min 
track_interp_interval = 10.0 #interpolate between two adjacent records for the same ship when the distance exceeds 100m
error_coords_limit = 10
error_speed_limit = 0.99

col_list = ['mmsi','width','length','draught','obstime','longitude','latitude','sog','cog','typecargo']#no Waterdepth column
ID_col_list = ['mmsi','typecargo']
index_list = ['mmsi','obstime']

class Trajectory:
    def __init__(self,mmsi,width,length,typecargo,draught,sog,cog,longitude,latitude,obstime):
        self.mmsi = mmsi
        self.width = width
        self.length = length
        self.typecargo = typecargo
        self.draught = draught
        self.sog = sog
        self.cog = cog
        self.speed = sog*sog2mps
        self.angle = cog/180*np.pi
        self.long = longitude
        self.lat = latitude
        self.x = longitude*deg2m
        self.y = latitude*deg2m
        self.obstime = obstime
        self.t = (obstime-start_datetime) / np.timedelta64(1,'s')
        self.dxdt = self.speed*np.sin(self.angle)
        self.dydt = self.speed*np.cos(self.angle)
        self.xspline = None
        self.yspline = None
        self.n = self.speed.size
    
    def AIS_synthesis(self):
        self.long = self.x/deg2m
        self.lat = self.y/deg2m
        self.sog = self.speed/sog2mps
        self.cog = self.angle/np.pi*180
        self.obstime = self.t * np.timedelta64(1,'s') + start_datetime

    def update(self,draught,speed,angle,x,y,t,dxdt,dydt):
        self.draught = draught
        self.speed = speed
        self.angle = angle
        self.x = x
        self.y = y
        self.t = t
        self.dxdt = dxdt
        self.dydt = dydt
        self.n = self.speed.size

    def clean_coords(self):
        if self.n < 3:
            return
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dl = np.sqrt(np.square(dx)+np.square(dy))
        dx2 = self.x[2:] - self.x[:-2] 
        dy2 = self.y[2:] - self.y[:-2]
        dl2 = dl[1:] + dl[:-1]
        flag_no_del = np.logical_not(dl2 > error_coords_limit*np.sqrt(np.square(dx2)+np.square(dy2))).tolist()
        flag_no_del = [True,*flag_no_del,True]
        self.update(draught = self.draught[flag_no_del],
                    speed = self.speed[flag_no_del],
                    angle = self.angle[flag_no_del],
                    x = self.x[flag_no_del],
                    y = self.y[flag_no_del],
                    t = self.t[flag_no_del],
                    dxdt = self.dxdt[flag_no_del],
                    dydt = self.dydt[flag_no_del])
        #self.AIS_synthesis()

    def clean_speed(self):
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dl = np.sqrt(np.square(dx)+np.square(dy))
        dl[dl==0] = np.finfo(np.float64).eps
        dt = np.diff(self.t)
        x_pre = self.x[:-1] + self.dxdt[:-1] * dt
        y_pre = self.y[:-1] + self.dydt[:-1] * dt
        error_pre = np.sqrt(np.square(self.x[1:]-x_pre) + np.square(self.y[1:]-y_pre))
        rel_err_pre = error_pre / dl
        x_post = self.x[1:] - self.dxdt[1:] * dt
        y_post = self.y[1:] - self.dydt[1:] * dt
        error_post = np.sqrt(np.square(self.x[:-1]-x_post) + np.square(self.y[:-1]-y_post))
        rel_err_post = error_post / dl
        if self.n < 3:
            flag_err_spd = np.logical_and(rel_err_pre[1:] > error_speed_limit,rel_err_post[:-1] > error_speed_limit)
            flag_pre = np.append(flag_err_spd,False)
            flag_post = np.insert(flag_err_spd,0,False)
            flag_replace = np.append(flag_post,False)
            dxdt_pre = dx[flag_pre]/dt[flag_pre]
            dxdt_post = dx[flag_post]/dt[flag_post]
            dydt_pre = dy[flag_pre]/dt[flag_pre]
            dydt_post = dy[flag_post]/dt[flag_post]
            w_pre = 1 / dl[flag_pre]
            w_post = 1 / dl[flag_post]
            self.dxdt[flag_replace] = (w_pre*dxdt_pre + w_post*dxdt_post)/(w_pre+w_post)
            self.dydt[flag_replace] = (w_pre*dydt_pre + w_post*dydt_post)/(w_pre+w_post)
            self.speed[flag_replace] = np.sqrt(np.square(self.dxdt[flag_replace])+np.square(self.dydt[flag_replace]))
            self.angle[flag_replace] = (np.arctan2(self.dxdt[flag_replace],self.dydt[flag_replace])+2*np.pi) % (2*np.pi)
        if rel_err_pre[0] > error_speed_limit:
            self.dxdt[0] = dx[0]/dt[0]
            self.dydt[0] = dy[0]/dt[0]
            self.speed[0] = np.sqrt(np.square(self.dxdt[0])+np.square(self.dydt[0]))
            self.angle[0] = (np.arctan2(self.dxdt[0],self.dydt[0])+2*np.pi) % (2*np.pi)
        if rel_err_post[-1] > error_speed_limit:
            self.dxdt[-1] = dx[-1]/dt[-1]
            self.dydt[-1] = dy[-1]/dt[-1]
            self.speed[-1] = np.sqrt(np.square(self.dxdt[-1])+np.square(self.dydt[-1]))
            self.angle[-1] = (np.arctan2(self.dxdt[-1],self.dydt[-1])+2*np.pi) % (2*np.pi)

        # for i,flag_err_spd in enumerate(rel_err_pre > error_speed_limit and rel_err_post > error_speed_limit):
        #     if flag_err_spd:
        #         dxdt_pre = dx[i]/dt[i]
        #         dxdt_post = dx[i+1]/dt[i+1]
        #         dydt_pre = dy[i]/dt[i]
        #         dydt_post = dy[i+1]/dt[i+1]
        #         w_pre = 1 / dl[i]
        #         w_post = 1 / dl[i+1]
        #         self.dxdt[i+1] = (w_pre*dxdt_pre + w_post*dxdt_post)/(w_pre+w_post)
        #         self.dydt[i+1] = (w_pre*dydt_pre + w_post*dydt_post)/(w_pre+w_post)
        #         self.speed[i+1] = np.sqrt(np.square(self.dxdt[i+1])+np.square(self.dydt[i+1]))
        #         self.angle[i+1] = np.arctan2(self.dxdt[i+1],self.dydt[i+1])
        
    def create_spline(self):
        self.clean_coords()
        self.clean_speed()
        self.xspline = interpolate.CubicHermiteSpline(x=self.t,y=self.x,dydx=self.dxdt)
        self.yspline = interpolate.CubicHermiteSpline(x=self.t,y=self.y,dydx=self.dydt)
    
    def interp_spline(self,interval):
        if self.xspline == None or self.yspline == None:
            self.create_spline()
        t_interp = np.append(np.arange(self.t[0],self.t[-1],interval,dtype="float64"),self.t[-1])
        t_change = np.abs(np.subtract(t_interp,self.t[:,np.newaxis]))
        nearst_t = np.argmin(t_change,axis=0)
        draught_interp = self.draught[nearst_t]
        x_interp = self.xspline(t_interp,nu=0)
        dxdt_interp = self.xspline(t_interp,nu=1)
        y_interp= self.yspline(t_interp,nu=0)
        dydt_interp = self.yspline(t_interp,nu=1)
        speed_interp = np.sqrt(np.square(dxdt_interp)+np.square(dydt_interp))
        angle_interp = (np.arctan2(dxdt_interp,dydt_interp)+2*np.pi) % (2*np.pi)
        # long_interp = x_interp / deg2m
        # lat_interp = y_interp / deg2m
        # if np.any(np.logical_and(np.logical_and(long_interp > 103.9605,long_interp < 103.98),np.logical_and(lat_interp > 1.42475,lat_interp < 1.427))):
        # #if np.any(np.logical_and(np.logical_and(long_interp > 103.934,long_interp < 103.935),np.logical_and(lat_interp > 1.4276,lat_interp < 1.4278))):
        # #if np.any(np.logical_and(np.logical_and(long_interp > 103.932,long_interp < 103.937),(lat_interp < 1.4278))):
        #     print(self.mmsi)
        #     plt.plot(self.long,self.lat,'go',linestyle='None', markersize = 4.0)
        #     plt.quiver(self.long, self.lat, self.dxdt, self.dydt, angles="xy", scale = 50, width = 0.001, pivot="tail")
        #     #plt.plot(long_interp,lat_interp,'bx',linestyle='-',markersize = 1.0)
        #     plt.plot(long_interp,lat_interp,linestyle='-',color='k')
        #     ax = plt.gca()
        #     for i, xy in enumerate(zip(self.long,self.lat)):
        #         ax.annotate(np.datetime_as_string(self.obstime[i]),xy=xy)
        #     for i, xy in enumerate(zip(self.long,self.lat)):
        #         ax.annotate(str(self.sog[i]),xy=xy)
        self.update(draught = draught_interp,
                    speed = speed_interp,
                    angle = angle_interp,
                    x = x_interp,
                    y = y_interp,
                    t = t_interp,
                    dxdt = dxdt_interp,
                    dydt = dydt_interp)
        #self.AIS_synthesis()
        #if np.any(np.logical_and(np.logical_and(self.long > 103.934,self.long < 103.935),np.logical_and(self.lat > 1.4276,self.lat < 1.4278))):
        #    plt.plot(self.long,self.lat,'bx',linestyle='-',markersize = 1.0)
        
    def to_pandas(self):
        self.AIS_synthesis()
        mmsi_pd = np.full(shape=self.n,fill_value=self.mmsi,dtype=np.int64)
        width_pd = np.full(shape=self.n,fill_value=self.width,dtype=np.float64)
        length_pd = np.full(shape=self.n,fill_value=self.length,dtype=np.float64)
        draught_pd = self.draught
        obstime_pd = self.obstime
        longitude_pd = self.long
        latitude_pd = self.lat
        sog_pd = self.sog
        cog_pd = self.cog
        typecargo_pd = np.full(shape=self.n,fill_value=self.typecargo,dtype=np.int64)
        data = [mmsi_pd,width_pd,length_pd,draught_pd,obstime_pd,longitude_pd,latitude_pd,sog_pd,cog_pd,typecargo_pd]
        #col_names = ['mmsi','width','length','draught','obstime','longitude','latitude','sog','cog','typecargo']
        df_output = pd.DataFrame.from_dict(dict(zip(col_list, data)))
        return(df_output)

print('Import AIS data from: \n' + str(AIS_inputfile))
print('...')

AIS_table = pd.read_csv(AIS_inputfile,header=0)
missing_col = set(col_list).difference(set(AIS_table.columns))
if not missing_col:
    AIS_table = AIS_table[col_list]
    #AIS_table = AIS_table.loc[:,col_list]
else:
    raise ValueError("Error: Columns missing: " + ", ".join(str(missing_col)) + " not in the input data file!")

if AIS_table.isnull().values.any():
    print("NaN found in data, removing")
    print('...')
    AIS_table.dropna()

AIS_table['obstime'] = pd.to_datetime(AIS_table['obstime'],format=obstime_format)
AIS_table[ID_col_list] = AIS_table[ID_col_list].astype(int)

duplicated_rows = AIS_table.duplicated(index_list,keep="first")
if duplicated_rows.values.sum() > 0:
    print("Duplicated \"MMSI + obstime\" found in data, removing laters")
    print('...')
    AIS_table = AIS_table.loc[~duplicated_rows.values]

print('Uniformize vessel information for every MMSI')
print('...')
MMSI_list = AIS_table['mmsi'].unique().tolist()
AIS_table = AIS_table.set_index(['mmsi','obstime'])
for mmsi in MMSI_list:
    mmsi_rows = AIS_table.xs(mmsi,level=0, axis=0)

    width_rows = mmsi_rows.groupby("width")
    count_width = width_rows["width"].count()
    if len(count_width.index) > 1:
        width_max = count_width.idxmax()
        AIS_table.loc[mmsi,"width"] = width_max
    
    length_rows = mmsi_rows.groupby("length")
    count_length = length_rows["length"].count()
    if len(count_length.index) > 1:
        length_max = count_length.idxmax()
        AIS_table.loc[mmsi,"length"] = length_max

    typecargo_rows = mmsi_rows.groupby("typecargo")
    count_typecargo = typecargo_rows["typecargo"].count()
    if len(count_typecargo.index) > 1:
        typecargo_max = count_typecargo.idxmax()
        AIS_table.loc[mmsi,"typecargo"] = typecargo_max

print(r'Remove AIS data with zero vessel width/legnth/draught')
print('...')
AIS_table = AIS_table[AIS_table.loc[:,"width"] > 0.0]
AIS_table = AIS_table[AIS_table.loc[:,"length"] > 0.0]
AIS_table = AIS_table[AIS_table.loc[:,"draught"] > 0.0]

sg_land = r'\\sg-gis\GIS\Template\GIS\Shapefiles & Geodatabase\Singapore\Shapefiles\Land\Historical Profile\v20250502\RD7550_CEx_SG_v20250502.shp'
my_land = r'\\sg-gis\GIS\Template\GIS\Shapefiles & Geodatabase\Malaysia\Shapefiles\Land\Historical Profile\v20240205\RD7550_CEx_MY_v20240205.shp'
proposed_jetty = r'\\sg-gis\GIS\GeoData\Singapore\618xxxxx\SG61803939\Data4Map\Proposed Jetty\SG61803939_ProposedJetty.shp'
with fiona.open(sg_land) as shp:
    for rec in shp:
        geom = shape(rec['geometry'])
        shapely.plotting.plot_polygon(geom,add_points=False,edgecolor='black',facecolor='#FFCC99')
with fiona.open(my_land) as shp:
    for rec in shp:
        geom = shape(rec['geometry'])
        shapely.plotting.plot_polygon(geom,add_points=False,edgecolor='black',facecolor='#FFCC99')
with fiona.open(proposed_jetty) as shp:
    for rec in shp:
        geom = shape(rec['geometry'])
        shapely.plotting.plot_polygon(geom,add_points=False,edgecolor='black',facecolor="#99FF00")
# with fiona.open(proposed_barrier) as shp:
#     for rec in shp:
#         geom = shape(rec['geometry'])
#         shapely.plotting.plot_line(geom,add_points=False,color='red',linewidth=1.0)

AIS_table = AIS_table.reset_index(drop=False).set_index(['mmsi','obstime'])
MMSI_list = AIS_table.index.levels[0].unique().tolist()
list_track_df = []
for mmsi in track(MMSI_list,description="Plotting Vessel Tracks..."):
    mmsi_rows = AIS_table.xs(key=mmsi, level=0, axis=0, drop_level=True)
    mmsi_rows = mmsi_rows.sort_index()
    dt = np.diff(mmsi_rows.index)
    seperate_track = np.where(dt > np.timedelta64(int(track_time_limit*1e9),'ns'))[0]+1

    for i,j in zip(np.insert(seperate_track,0,0),np.append(seperate_track,len(mmsi_rows.index))):
        track_rows = mmsi_rows[i:j]
        track = Trajectory(mmsi=mmsi,
                          width=track_rows["width"].values[0],
                          length=track_rows["length"].values[0],
                          typecargo=track_rows["typecargo"].values[0],
                          draught=track_rows["draught"].values,
                          sog=track_rows["sog"].values,
                          cog=track_rows["cog"].values,
                          longitude=track_rows["longitude"].values,
                          latitude=track_rows["latitude"].values,
                          obstime=track_rows.index.values)
        if track.n > 1 and not (np.any(np.logical_and(np.logical_and(track.long > 103.8198,track.long < 103.8213),np.logical_and(track.lat > 1.2624,track.lat < 1.2635)))):# and (np.any(track.long > 103.8198)):
        #if track.n > 1 and np.any(np.logical_and(np.logical_and(track.long > 103.817,track.long < 103.8198),np.logical_and(track.lat > 1.262,track.lat < 1.2635))):
            track.interp_spline(10)
            #plt.plot(track.long,track.lat,alpha=0.2,color=cm.viridis(track.typecargo/100))
        # if np.any(np.logical_and(np.logical_and(track.long > 103.68,track.long < 103.70),np.logical_and(track.lat > 1.405,track.lat < 1.43))):
        #     continue
        # elif np.any(np.logical_and(np.logical_and(track.long > 103.66,track.long < 103.67),np.logical_and(track.lat > 1.39,track.lat < 1.4))):
        #     continue
        # else:
            list_track_df.append(track.to_pandas())
            plt.plot(track.long,track.lat,alpha=0.2,color=cm.viridis(track.typecargo/100))
    # sys.stdout.write('\r')
    # sys.stdout.write("[%-20s] %d%%" % ('='*int((progress+1)/len(MMSI_list)*20), (progress+1)/len(MMSI_list)*100))
    # sys.stdout.flush()

# ax = plt.gca()
# ax.ticklabel_format(useOffset=False)
# ax.set_xlim(103.91,104.02)
# ax.set_ylim(1.408,1.44)
# plt.grid(visible=None)
# plt.show()


df_track_interp = pd.concat(list_track_df)
df_track_interp.to_csv(AIS_outputfile,date_format=obstime_format,index=False)
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
ax.set_xlim(103.817,103.823)
ax.set_ylim(1.261,1.2635)
plt.grid(visible=None)
plt.show()
