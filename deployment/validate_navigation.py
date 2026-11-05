"""执行 mesh -> NavMesh -> 路径 -> 移动约束 -> Gaussian RGB 的闭环验证。"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import habitat_sim as hs
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image
import imageio.v2 as imageio

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scene-dir',type=Path,default=Path(__file__).parent/'scene_crop');ap.add_argument('--output',type=Path,default=Path(__file__).parent/'results_crop');ap.add_argument('--bake-only',action='store_true');ap.add_argument('--skip-bake',action='store_true');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);nav=args.scene_dir/'wanda_crop.navmesh';report={}
 settings=hs.NavMeshSettings();settings.set_defaults();settings.cell_size=0.1;settings.cell_height=0.05;settings.agent_height=1.5;settings.agent_radius=0.25;settings.agent_max_climb=0.2;settings.agent_max_slope=35
 if not args.skip_bake:
  cfg=hs.SimulatorConfiguration();cfg.scene_id=str(args.scene_dir/'wanda_crop_mesh_yup.ply');cfg.create_renderer=False;cfg.load_semantic_mesh=False
  with hs.Simulator(hs.Configuration(cfg,[hs.agent.AgentConfiguration()])) as sim:
   start=time.time();assert sim.recompute_navmesh(sim.pathfinder,settings),'NavMesh bake failed';sim.pathfinder.save_nav_mesh(str(nav));report['bake_seconds']=time.time()-start
 pf=hs.PathFinder();assert pf.load_nav_mesh(str(nav));pf.seed(17);assert pf.navigable_area>10
 island=int(np.argmax([pf.island_area(i) for i in range(pf.num_islands)]));report.update(navmesh=str(nav),navigable_area_m2=pf.navigable_area,num_islands=pf.num_islands,largest_island=island,largest_island_area_m2=pf.island_area(island),bounds=[np.asarray(x).tolist() for x in pf.get_bounds()],agent_radius_m=0.25,agent_height_m=1.5)
 best=None
 for _ in range(150):
  a=pf.get_random_navigable_point(island_index=island);b=pf.get_random_navigable_point(island_index=island);path=hs.ShortestPath();path.requested_start=a;path.requested_end=b
  if pf.find_path(path) and 15<path.geodesic_distance<100:
   best=path;break
 assert best is not None,'No usable navigation route';points=np.array(best.points);np.save(args.output/'path.npy',points);report['geodesic_distance_m']=best.geodesic_distance
 # 沿 geodesic 折线移动，并用 try_step/is_navigable 检查 NavMesh 约束。
 trajectory=[points[0]]
 for a,b in zip(points[:-1],points[1:]):
  for t in np.linspace(0,1,max(2,int(np.linalg.norm(b-a)/0.25)+1))[1:]:
   wanted=a+(b-a)*t;actual=np.asarray(pf.try_step(trajectory[-1],wanted));assert np.linalg.norm(actual-wanted)<0.15,'NavMesh step blocked';assert pf.is_navigable(actual,0.15);trajectory.append(actual)
 trajectory=np.array(trajectory);np.save(args.output/'trajectory.npy',trajectory);report.update(navigation_steps=len(trajectory)-1,goal_error_m=float(np.linalg.norm(trajectory[-1]-points[-1])),all_steps_navigable=True)
 # 输出真实 NavMesh 三角形和实际轨迹，便于人工检查路径是否穿越空洞。
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.collections import PolyCollection
 tris=np.asarray(pf.build_navmesh_vertices(island)).reshape(-1,3,3);fig,ax=plt.subplots(figsize=(10,10));ax.add_collection(PolyCollection(tris[:,:,[0,2]],facecolors='#c9dfcf',edgecolors='none'));ax.plot(trajectory[:,0],trajectory[:,2],'r-',lw=2);ax.scatter(*trajectory[0,[0,2]],c='blue');ax.scatter(*trajectory[-1,[0,2]],c='red');ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('Habitat X (m)');ax.set_ylabel('Habitat Z (m)');fig.savefig(args.output/'navmesh_route.png',dpi=140);plt.close(fig)
 if not args.bake_only:
  cfg=hs.SimulatorConfiguration();cfg.scene_id=str(args.scene_dir/'wanda.stage_config.json');cfg.enable_physics=False;cfg.load_semantic_mesh=False;cfg.gpu_device_id=0
  sensor=hs.CameraSensorSpec();sensor.uuid='rgb';sensor.sensor_type=hs.SensorType.COLOR;sensor.resolution=[360,640];sensor.hfov=69.0;sensor.position=mn.Vector3(0,1.5,0);sensor.near=0.05;sensor.far=500.0
  agent=hs.agent.AgentConfiguration();agent.sensor_specifications=[sensor]
  times=[];stats=[]
  with hs.Simulator(hs.Configuration(cfg,[agent])) as sim:
   assert sim.pathfinder.load_nav_mesh(str(nav));assert hs.cuda_enabled,'CUDA disabled'
   with imageio.get_writer(str(args.output/'navigation_rgb.mp4'),fps=10,macro_block_size=2) as writer:
    for i,p in enumerate(trajectory):
     look=trajectory[min(i+3,len(trajectory)-1)]-trajectory[max(0,i-1)];look[1]=0
     yaw=math.atan2(-float(look[0]),-float(look[2]));state=hs.AgentState();state.position=p;state.rotation=quat_from_angle_axis(yaw,np.array([0,1,0]));sim.initialize_agent(0,state)
     t=time.perf_counter();rgb=sim.get_sensor_observations()['rgb'][:,:,:3];times.append(time.perf_counter()-t);writer.append_data(rgb)
     assert rgb.shape==(360,640,3);stats.append(float(rgb.std()))
     if i in [0,len(trajectory)//2,len(trajectory)-1]:Image.fromarray(rgb).save(args.output/f'rgb_{i:04d}.png')
     if i%20==0:print('Rendered',i,'/',len(trajectory),flush=True)
  assert np.median(stats)>10,'RGB frames have little variation'
  report.update(rgb_frames=len(times),render_mean_seconds=float(np.mean(times)),rgb_std_median=float(np.median(stats)),camera={'model':'D435 RGB nominal pinhole','width':640,'height':360,'hfov_deg':69,'fx':320/math.tan(math.radians(34.5)),'fy':320/math.tan(math.radians(34.5)),'cx':319.5,'cy':179.5,'height_m':1.5,'note':'Nominal RGB FoV, not physical-device calibrated intrinsics; no lens distortion.'})
 report['environment']={'habitat_gs_version':hs.__version__,'cuda_enabled':bool(hs.cuda_enabled),'gpu':'NVIDIA GeForce RTX 3090','cuda_toolkit':'12.4','conda_env':'cloudrobo-habitat-gs'}
 report['preparation']=json.loads((args.scene_dir/'preparation.json').read_text());(args.output/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
